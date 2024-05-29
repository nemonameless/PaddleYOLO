# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ..initializer import constant_
from ppdet.core.workspace import register
from ppdet.utils.check import _IS_NPU
from ..bbox_utils import batch_distance2bbox, bbox_iou, custom_ceil
from ..assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.backbones.csp_darknet import BaseConv
from ppdet.modeling.layers import MultiClassNMS

__all__ = ['YOLOv8Head', 'YOLOv8InsHead']


@register
class YOLOv8Head(nn.Layer):
    __shared__ = [
        'num_classes', 'eval_size', 'act', 'trt', 'exclude_nms',
        'exclude_post_process'
    ]
    __inject__ = ['assigner', 'nms']

    def __init__(self,
                 in_channels=[256, 512, 1024],
                 num_classes=80,
                 act='silu',
                 fpn_strides=[8, 16, 32],
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 reg_range=None,
                 use_varifocal_loss=False,
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 eval_size=None,
                 loss_weight={
                     'class': 0.5,
                     'iou': 7.5,
                     'dfl': 1.5,
                 },
                 trt=False,
                 exclude_nms=False,
                 exclude_post_process=False,
                 customized_c3=-1,
                 print_l1_loss=True):
        super(YOLOv8Head, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        if reg_range:
            self.reg_range = reg_range
        else:
            self.reg_range = (0, reg_max)  # not reg_max+1
        self.reg_channels = self.reg_range[1] - self.reg_range[0]
        self.use_varifocal_loss = use_varifocal_loss
        self.assigner = assigner
        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.eval_size = eval_size
        self.loss_weight = loss_weight
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process
        self.customized_c3 = customized_c3
        self.print_l1_loss = print_l1_loss

        # cls loss
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        # pred head
        c2 = max((16, in_channels[0] // 4, self.reg_max * 4))
        if self.customized_c3 < 0:
            c3 = max(in_channels[0], self.num_classes)
        else:
            c3 = self.customized_c3
        self.conv_reg = nn.LayerList()
        self.conv_cls = nn.LayerList()
        for in_c in self.in_channels:
            self.conv_reg.append(
                nn.Sequential(* [
                    BaseConv(
                        in_c, c2, 3, 1, act=act),
                    BaseConv(
                        c2, c2, 3, 1, act=act),
                    nn.Conv2D(
                        c2,
                        self.reg_max * 4,
                        1,
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
                ]))
            self.conv_cls.append(
                nn.Sequential(* [
                    BaseConv(
                        in_c, c3, 3, 1, act=act),
                    BaseConv(
                        c3, c3, 3, 1, act=act),
                    nn.Conv2D(
                        c3,
                        self.num_classes,
                        1,
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
                ]))
        self.proj = paddle.arange(self.reg_max).astype('float32')
        self._initialize_biases()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _initialize_biases(self):
        for a, b, s in zip(self.conv_reg, self.conv_cls, self.fpn_strides):
            constant_(a[-1].weight)
            constant_(a[-1].bias, 1.0)
            constant_(b[-1].weight)
            constant_(b[-1].bias, math.log(5 / self.num_classes / (640 / s)**2))

    def forward(self, feats, targets=None):
        if self.training:
            return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)

    def forward_train(self, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_logits_list, bbox_preds_list, bbox_dist_preds_list = [], [], []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            bbox_dist_preds = self.conv_reg[i](feat)
            cls_logit = self.conv_cls[i](feat)
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, l]).transpose([0, 3, 1, 2])  # [8, 6400, 4, 16]
            bbox_preds = F.softmax(bbox_dist_preds, axis=3).matmul(self.proj.reshape([-1, 1])).squeeze(-1) # [8, 6400, 4]

            cls_logits_list.append(cls_logit)
            bbox_preds_list.append(bbox_preds.transpose([0, 2, 1]).reshape([-1, 4, h, w]))
            bbox_dist_preds_list.append(bbox_dist_preds)

        return self.get_loss([
            cls_logits_list, bbox_preds_list, bbox_dist_preds_list, anchors, anchor_points,
            num_anchors_list, stride_tensor
        ], targets)

    def forward_eval(self, feats):
        anchor_points, stride_tensor = self._generate_anchors(feats)

        cls_logits_list, bbox_preds_list = [], []
        feats_shapes = []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            bbox_dist_preds = self.conv_reg[i](feat)
            cls_logit = self.conv_cls[i](feat)

            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, l]).transpose([0, 3, 1, 2])  # [8, 6400, 4, 16]
            bbox_preds = F.softmax(bbox_dist_preds, axis=3).matmul(self.proj.reshape([-1, 1])).squeeze(-1) # [8, 6400, 4]
            cls_logits_list.append(cls_logit)
            bbox_preds_list.append(bbox_preds.transpose([0, 2, 1]).reshape([-1, 4, h, w]))
            feats_shapes.append(l)

        pred_scores = [
            cls_score.transpose([0, 2, 3, 1]).reshape(
                [-1, size, self.num_classes])
            for size, cls_score in zip(feats_shapes, cls_logits_list)
        ]
        pred_dists = [
            bbox_pred.transpose([0, 2, 3, 1]).reshape([-1, size, 4])
            for size, bbox_pred in zip(feats_shapes, bbox_preds_list)
        ]
        pred_scores = F.sigmoid(paddle.concat(pred_scores, 1))
        pred_bboxes = paddle.concat(pred_dists, 1)

        return pred_scores, pred_bboxes, anchor_points, stride_tensor

    def _generate_anchors(self, feats=None, dtype='float32'):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = paddle.arange(end=w) + self.grid_cell_offset
            shift_y = paddle.arange(end=h) + self.grid_cell_offset
            shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
            anchor_point = paddle.cast(
                paddle.stack(
                    [shift_x, shift_y], axis=-1), dtype=dtype)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(paddle.full([h * w, 1], stride, dtype=dtype))
        anchor_points = paddle.concat(anchor_points)
        stride_tensor = paddle.concat(stride_tensor)
        return anchor_points, stride_tensor

    def _bbox2distance(self, points, bbox, reg_max=15, eps=0.01):
        x1y1, x2y2 = paddle.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return paddle.concat([lt, rb], -1).clip(0, reg_max - eps)

    def _df_loss(self, pred_dist, target, lower_bound=0):
        target_left = paddle.cast(target.floor(), 'int64')
        target_right = target_left + 1
        weight_left = target_right.astype('float32') - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist, target_left - lower_bound,
            reduction='none') * weight_left
        loss_right = F.cross_entropy(
            pred_dist, target_right - lower_bound,
            reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def get_loss(self, head_outs, gt_meta):
        cls_scores, bbox_preds, bbox_dist_preds, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs


        bs = cls_scores[0].shape[0]
        flatten_cls_preds = [
            cls_pred.transpose([0, 2, 3, 1]).reshape([bs, -1, self.num_classes])
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.transpose([0, 2, 3, 1]).reshape([bs, -1, 4])
            for bbox_pred in bbox_preds
        ]
        flatten_pred_dists = [
            bbox_pred_org.reshape([bs, -1, self.reg_max * 4])
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = paddle.concat(flatten_pred_dists, 1) # [8, 8400, 64]
        pred_scores = paddle.concat(flatten_cls_preds, 1) # [8, 8400, 80]
        pred_distri = paddle.concat(flatten_pred_bboxes, 1) # [8, 8400, 4]


        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = batch_distance2bbox(anchor_points_s, pred_distri) # xyxy [8, 8400, 4]
        pred_bboxes = pred_bboxes * stride_tensor # must *stride

        gt_labels = gt_meta['gt_class'] # [16, 51, 1]
        gt_bboxes = gt_meta['gt_bbox'] # xyxy max=640 # [16, 51, 4]
        pad_gt_mask = gt_meta['pad_gt_mask']

        assigned_labels, assigned_bboxes, assigned_scores = \
            self.assigner(
            F.sigmoid(pred_scores.detach()),
            pred_bboxes.detach(),
            anchor_points,
            num_anchors_list,
            gt_labels,
            gt_bboxes, # xyxy
            pad_gt_mask,
            bg_index=self.num_classes)
        # rescale bbox
        assigned_bboxes /= stride_tensor
        pred_bboxes /= stride_tensor

        # cls loss
        loss_cls = self.bce(pred_scores, assigned_scores).sum() # [16, 8400, 80]

        assigned_scores_sum = assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum /= paddle.distributed.get_world_size()
        assigned_scores_sum = paddle.clip(assigned_scores_sum, min=1.)
        loss_cls /= assigned_scores_sum

        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # ciou loss
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(
                pred_bboxes, bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)
            iou = bbox_iou( 
                pred_bboxes_pos.split(4, axis=-1),
                assigned_bboxes_pos.split(4, axis=-1),
                x1y1x2y2=True, # xyxy
                ciou=True,
                eps=1e-7)
            loss_iou = ((1.0 - iou) * bbox_weight).sum() / assigned_scores_sum

            if self.print_l1_loss:
                loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)
            else:
                loss_l1 = paddle.zeros([1])

            # dfl loss
            dist_mask = mask_positive.unsqueeze(-1).tile(
                [1, 1, self.reg_max * 4])
            pred_dist_pos = paddle.masked_select(
                flatten_dist_preds, dist_mask).reshape([-1, 4, self.reg_max])
            assigned_ltrb = self._bbox2distance(
                anchor_points_s,
                assigned_bboxes,
                reg_max=self.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = paddle.masked_select(
                assigned_ltrb, bbox_mask).reshape([-1, 4])

            loss_dfl = self._df_loss(pred_dist_pos,
                                     assigned_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
        else:
            loss_iou = flatten_dist_preds.sum() * 0.
            loss_dfl = flatten_dist_preds.sum() * 0.
            loss_l1 = flatten_dist_preds.sum() * 0.

        loss_cls *= self.loss_weight['class']
        loss_iou *= self.loss_weight['iou']
        loss_dfl *= self.loss_weight['dfl']
        loss_total = loss_cls + loss_iou + loss_dfl

        num_gpus = gt_meta.get('num_gpus', 8)
        total_bs = bs * num_gpus

        out_dict = {
            'loss': loss_total * total_bs,
            'loss_cls': loss_cls * total_bs,
            'loss_iou': loss_iou * total_bs,
            'loss_dfl': loss_dfl * total_bs,
        }
        if self.print_l1_loss:
            # just see convergence
            out_dict.update({'loss_l1': loss_l1 * total_bs})
        return out_dict

    def post_process(self,
                     head_outs,
                     im_shape,
                     scale_factor,
                     infer_shape=[640, 640]):
        pred_scores, pred_bboxes, anchor_points, stride_tensor = head_outs

        pred_bboxes = batch_distance2bbox(anchor_points, pred_bboxes)
        pred_bboxes *= stride_tensor

        if self.exclude_post_process:
            return paddle.concat(
                [pred_bboxes, pred_scores], axis=-1), None
        else:
            pred_scores = pred_scores.transpose([0, 2, 1])
            # scale bbox to origin
            scale_factor = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
            pred_bboxes /= scale_factor
            if self.exclude_nms:
                # `exclude_nms=True` just use in benchmark
                return pred_bboxes, pred_scores
            else:
                bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
                return bbox_pred, bbox_num


class MaskProto(nn.Layer):
    # YOLOv8 mask Proto module for instance segmentation models
    def __init__(self, ch_in, num_protos=256, num_masks=32, act='silu'):
        super().__init__()
        self.conv1 = BaseConv(ch_in, num_protos, 3, 1, act=act)
        self.upsample = nn.Conv2DTranspose(
            num_protos, num_protos, 2, 2, 0, bias_attr=True)
        self.conv2 = BaseConv(num_protos, num_protos, 3, 1, act=act)
        self.conv3 = BaseConv(num_protos, num_masks, 1, 1, act=act)

    def forward(self, x):
        return self.conv3(self.conv2(self.upsample(self.conv1(x))))


@register
class YOLOv8InsHead(nn.Layer):
    __shared__ = [
        'num_classes', 'eval_size', 'act', 'trt', 'exclude_nms',
        'exclude_post_process', 'with_mask', 'width_mult'
    ]
    __inject__ = ['assigner', 'nms']

    def __init__(self,
                 with_mask=True,
                 in_channels=[256, 512, 1024],
                 num_classes=80,
                 num_masks=32,
                 num_protos=256,
                 width_mult=1.0,
                 act='silu',
                 fpn_strides=[8, 16, 32],
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 reg_range=None,
                 use_varifocal_loss=False,
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 eval_size=[640, 640],
                 loss_weight={
                     'class': 0.5,
                     'iou': 7.5,
                     'dfl': 1.5,
                 },
                 trt=False,
                 exclude_nms=False,
                 exclude_post_process=False,
                 customized_c3=-1,
                 mask_thr_binary=0.5,
                 print_l1_loss=True):
        super(YOLOv8InsHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.with_mask = with_mask
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_masks = num_masks
        self.num_protos = int(num_protos * width_mult)
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        if reg_range:
            self.reg_range = reg_range
        else:
            self.reg_range = (0, reg_max)  # not reg_max+1
        self.reg_channels = self.reg_range[1] - self.reg_range[0]
        self.use_varifocal_loss = use_varifocal_loss
        self.assigner = assigner
        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.eval_size = eval_size
        self.loss_weight = loss_weight
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process
        self.customized_c3 = customized_c3
        self.print_l1_loss = print_l1_loss

        # cls loss
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        # pred head
        c2 = max((16, in_channels[0] // 4, self.reg_max * 4))
        if self.customized_c3 < 0:
            c3 = max(in_channels[0], self.num_classes)
        else:
            c3 = self.customized_c3
        c4 = max(in_channels[0] // 4, self.num_masks)
        self.conv_reg = nn.LayerList()
        self.conv_cls = nn.LayerList()
        self.conv_ins = nn.LayerList()
        for in_c in self.in_channels:
            self.conv_reg.append(
                nn.Sequential(* [
                    BaseConv(
                        in_c, c2, 3, 1, act=act),
                    BaseConv(
                        c2, c2, 3, 1, act=act),
                    nn.Conv2D(
                        c2,
                        self.reg_channels * 4,
                        1,
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
                ]))
            self.conv_cls.append(
                nn.Sequential(* [
                    BaseConv(
                        in_c, c3, 3, 1, act=act),
                    BaseConv(
                        c3, c3, 3, 1, act=act),
                    nn.Conv2D(
                        c3,
                        self.num_classes,
                        1,
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
                ]))
            self.conv_ins.append(
                nn.Sequential(* [
                    BaseConv(
                        in_c, c4, 3, 1, act=act),
                    BaseConv(
                        c4, c4, 3, 1, act=act),
                    nn.Conv2D(
                        c4,
                        self.num_masks,
                        1,
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
                ]))
        self.mask_thr_binary = mask_thr_binary
        self.proto = MaskProto(
            in_channels[0], self.num_protos, self.num_masks, act=act)
        self.proj = paddle.arange(self.reg_max).astype('float32')
        self._initialize_biases()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _initialize_biases(self):
        for a, b, s in zip(self.conv_reg, self.conv_cls, self.fpn_strides):
            constant_(a[-1].weight)
            constant_(a[-1].bias, 1.0)
            constant_(b[-1].weight)
            constant_(b[-1].bias, math.log(5 / self.num_classes / (640 / s)**2))

    def forward(self, feats, targets=None):
        if self.training:
            return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)

    def forward_train(self, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_logits_list, bbox_preds_list, bbox_dist_preds_list = [], [], []

        mask_feat = self.proto(feats[0])
        mask_coeff_list = []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            bbox_dist_preds = self.conv_reg[i](feat)
            cls_logit = self.conv_cls[i](feat)
            mask_pred = self.conv_ins[i](feat)

            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, l]).transpose([0, 3, 1, 2])
            bbox_preds = F.softmax(
                bbox_dist_preds,
                axis=3).matmul(self.proj.reshape([-1, 1])).squeeze(-1)

            cls_logits_list.append(cls_logit)
            mask_coeff_list.append(mask_pred)
            bbox_preds_list.append(
                bbox_preds.transpose([0, 2, 1]).reshape([-1, 4, h, w]))
            bbox_dist_preds_list.append(bbox_dist_preds)

        return self.get_loss([
            cls_logits_list, bbox_preds_list, bbox_dist_preds_list,
            mask_coeff_list, mask_feat, anchors, anchor_points,
            num_anchors_list, stride_tensor
        ], targets)

    def forward_eval(self, feats):
        mask_proto = self.proto(feats[0])
        anchor_points, stride_tensor = self._generate_anchors(feats)

        cls_logits_list, bbox_preds_list, mask_coeff_list = [], [], []
        feats_shapes = []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            bbox_dist_preds = self.conv_reg[i](feat)  # [1, 64, 80, 80] # 4*16
            cls_logit = self.conv_cls[i](feat)  # [1, 80, 80, 80]
            mask_coeff = self.conv_ins[i](feat)  # [1, 32, 80, 80]

            cls_logits_list.append(cls_logit)
            mask_coeff_list.append(mask_coeff)
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, l]).transpose([0, 3, 1, 2])
            bbox_preds = F.softmax(
                bbox_dist_preds,
                axis=3).matmul(self.proj.reshape([-1, 1])).squeeze(-1)
            bbox_preds_list.append(
                bbox_preds.transpose([0, 2, 1]).reshape([-1, 4, h, w]))

            feats_shapes.append(l)

        pred_scores = [
            cls_score.transpose([0, 2, 3, 1]).reshape(
                [-1, size, self.num_classes])
            for size, cls_score in zip(feats_shapes, cls_logits_list)
        ]
        pred_scores = F.sigmoid(paddle.concat(pred_scores, 1))

        pred_dists = [
            bbox_pred.transpose([0, 2, 3, 1]).reshape([-1, size, 4])
            for size, bbox_pred in zip(feats_shapes, bbox_preds_list)
        ]
        pred_bboxes = paddle.concat(pred_dists, 1)

        mask_coeff_list = [
            mask_coeff.transpose([0, 2, 3, 1]).reshape(
                [-1, size, self.num_masks])
            for size, mask_coeff in zip(feats_shapes, mask_coeff_list)
        ]
        pred_mask_coeffs = paddle.concat(mask_coeff_list, axis=1)

        return pred_scores, pred_bboxes, pred_mask_coeffs, mask_proto, anchor_points, stride_tensor

    def _generate_anchors(self, feats=None, dtype='float32'):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = paddle.arange(end=w) + self.grid_cell_offset
            shift_y = paddle.arange(end=h) + self.grid_cell_offset
            shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
            anchor_point = paddle.cast(
                paddle.stack(
                    [shift_x, shift_y], axis=-1), dtype=dtype)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(paddle.full([h * w, 1], stride, dtype=dtype))
        anchor_points = paddle.concat(anchor_points)
        stride_tensor = paddle.concat(stride_tensor)
        return anchor_points, stride_tensor

    def _bbox2distance(self, points, bbox, reg_max=15, eps=0.01):
        x1y1, x2y2 = paddle.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return paddle.concat([lt, rb], -1).clip(0, reg_max - eps)

    def _df_loss(self, pred_dist, target, lower_bound=0):
        target_left = paddle.cast(target.floor(), 'int64')
        target_right = target_left + 1
        weight_left = target_right.astype('float32') - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist, target_left - lower_bound,
            reduction='none') * weight_left
        loss_right = F.cross_entropy(
            pred_dist, target_right - lower_bound,
            reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def get_loss(self, head_outs, gt_meta):
        assert 'gt_bbox' in gt_meta and 'gt_class' in gt_meta
        assert 'gt_segm' in gt_meta

        cls_scores, bbox_preds, bbox_dist_preds, pred_mask_coeffs, mask_proto, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs

        bs = cls_scores[0].shape[0]
        imgsz = paddle.to_tensor(cls_scores[0].shape[2:]) * self.fpn_strides[
            0]  # image size (h,w)
        mask_h, mask_w = mask_proto.shape[-2:]
        flatten_cls_preds = [
            cls_pred.transpose([0, 2, 3, 1]).reshape(
                [bs, -1, self.num_classes]) for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.transpose([0, 2, 3, 1]).reshape([bs, -1, 4])
            for bbox_pred in bbox_preds
        ]
        flatten_pred_dists = [
            bbox_pred_org.reshape([bs, -1, self.reg_max * 4])
            for bbox_pred_org in bbox_dist_preds
        ]
        flatten_mask_coeff_preds = [
            mask_coeff.transpose([0, 2, 3, 1]).reshape(
                [bs, -1, self.num_masks]) for mask_coeff in pred_mask_coeffs
        ]

        flatten_dist_preds = paddle.concat(flatten_pred_dists, 1)
        pred_scores = paddle.concat(flatten_cls_preds, 1)
        pred_distri = paddle.concat(flatten_pred_bboxes, 1)
        pred_mask_coeff = paddle.concat(flatten_mask_coeff_preds, 1)

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = batch_distance2bbox(anchor_points_s, pred_distri)  # xyxy
        pred_bboxes = pred_bboxes * stride_tensor

        gt_labels = paddle.stack(gt_meta['gt_class'])
        gt_bboxes = paddle.stack(gt_meta['gt_bbox'])  # xyxy
        pad_gt_mask = paddle.stack(gt_meta['pad_gt_mask'])
        gt_segms = paddle.stack(gt_meta['gt_segm']).cast('float32')

        if tuple(gt_segms.shape[-2:]) != (mask_h, mask_w):  # downsample
            gt_segms = F.interpolate(
                gt_segms, (mask_h, mask_w),
                mode='nearest').reshape([bs, -1, mask_h * mask_w])

        assigned_labels, assigned_bboxes, assigned_scores, assigned_gt_index = \
            self.assigner(
            F.sigmoid(pred_scores.detach()),
            pred_bboxes.detach(),
            anchor_points,
            num_anchors_list,
            gt_labels,
            gt_bboxes, # xyxy
            pad_gt_mask,
            bg_index=self.num_classes,
            gt_segms=gt_segms)
        # rescale bbox
        assigned_bboxes /= stride_tensor
        pred_bboxes /= stride_tensor

        # assign segms for masks
        assigned_masks = paddle.gather(
            gt_segms.reshape([-1, mask_h * mask_w]),
            assigned_gt_index.flatten(),
            axis=0)
        assigned_masks = assigned_masks.reshape(
            [bs, assigned_gt_index.shape[1], mask_h * mask_w])

        # cls loss
        loss_cls = self.bce(pred_scores, assigned_scores).sum()

        assigned_scores_sum = assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum /= paddle.distributed.get_world_size()
        assigned_scores_sum = paddle.clip(assigned_scores_sum, min=1.)
        loss_cls /= assigned_scores_sum

        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # ciou loss
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)
            iou = bbox_iou(
                pred_bboxes_pos.split(
                    4, axis=-1),
                assigned_bboxes_pos.split(
                    4, axis=-1),
                x1y1x2y2=True,  # xyxy
                ciou=True,
                eps=1e-7)
            loss_iou = ((1.0 - iou) * bbox_weight).sum() / assigned_scores_sum

            if self.print_l1_loss:
                loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)
            else:
                loss_l1 = paddle.zeros([1])

            # dfl loss
            dist_mask = mask_positive.unsqueeze(-1).tile(
                [1, 1, self.reg_max * 4])
            pred_dist_pos = paddle.masked_select(
                flatten_dist_preds, dist_mask).reshape([-1, 4, self.reg_max])
            assigned_ltrb = self._bbox2distance(
                anchor_points_s,
                assigned_bboxes,
                reg_max=self.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = paddle.masked_select(
                assigned_ltrb, bbox_mask).reshape([-1, 4])

            loss_dfl = self._df_loss(pred_dist_pos,
                                     assigned_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum

            # mask loss
            loss_mask = self.calculate_segmentation_loss(
                mask_positive, assigned_gt_index, assigned_masks,
                assigned_bboxes * stride_tensor, mask_proto, pred_mask_coeff,
                imgsz)
            # [bs, 8400] [bs, 8400] [bs, 8400, 160 * 160] [bs, 8400, 4] [bs, 32, 160, 160] [bs, 8400, 32]
            loss_mask /= assigned_scores_sum
        else:
            loss_iou = flatten_dist_preds.sum() * 0.
            loss_dfl = flatten_dist_preds.sum() * 0.
            loss_mask = flatten_dist_preds.sum() * 0.
            loss_l1 = flatten_dist_preds.sum() * 0.

        loss_cls *= self.loss_weight['class']
        loss_iou *= self.loss_weight['iou']
        loss_dfl *= self.loss_weight['dfl']
        loss_mask *= self.loss_weight['iou']  # same as iou
        loss_total = loss_cls + loss_iou + loss_dfl + loss_mask

        num_gpus = gt_meta.get('num_gpus', 8)
        total_bs = bs * num_gpus

        out_dict = {
            'loss': loss_total * total_bs,
            'loss_cls': loss_cls * total_bs,
            'loss_iou': loss_iou * total_bs,
            'loss_dfl': loss_dfl * total_bs,
            'loss_mask': loss_mask * total_bs,
        }
        if self.print_l1_loss:
            # just see convergence
            out_dict.update({'loss_l1': loss_l1 * total_bs})
        return out_dict

    def calculate_segmentation_loss(self,
                                    fg_mask,
                                    target_gt_idx,
                                    masks,
                                    target_bboxes,
                                    proto,
                                    pred_masks,
                                    imgsz,
                                    overlap=True):
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (paddle.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (paddle.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (paddle.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (paddle.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (paddle.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (paddle.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (paddle.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (paddle.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (paddle.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = paddle.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = paddle.to_tensor([0.])

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        # [8, 8400, 4]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2).unsqueeze(
            -1)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * paddle.to_tensor(
            [mask_w, mask_h, mask_w, mask_h])

        for i, single_i in enumerate(
                zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea,
                    masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            #  [8400] [8400] [8400, 32] [32, 160, 160] [8400, 4] [8400, 1] [8400, 25600]
            if fg_mask_i.any():
                loss += self.single_mask_loss(
                    masks_i[fg_mask_i], pred_masks_i[fg_mask_i], proto_i,
                    mxyxy_i[fg_mask_i], marea_i[fg_mask_i])
                #  [10, 25600] [10, 32] [32, 160, 160] [10, 4] [10, 1]
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()
                # inf sums may lead to nan loss
        return loss

    @staticmethod
    def single_mask_loss(gt_mask, pred, proto, xyxy, area):
        """
        Compute the instance segmentation loss for a single image.
        Args:
            gt_mask (paddle.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (paddle.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (paddle.Tensor): Prototype masks of shape (32, H, W).
            xyxy (paddle.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (paddle.Tensor): Area of each ground truth bounding box of shape (n,).
        Returns:
            (paddle.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = paddle.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = paddle.einsum(
            'in,nhw->ihw', pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        nt = pred.shape[0]
        gt_mask = gt_mask.reshape([nt, *proto.shape[1:]])
        nmasks = 32
        pred_mask = (pred @proto.reshape([nmasks, -1])).reshape(
            [-1, *proto.shape[1:]])  # (n,32) @ (32,80,80) -> (n,80,80)
        loss = F.binary_cross_entropy_with_logits(
            pred_mask, gt_mask, reduction='none')
        return (crop_mask(loss, xyxy).mean(axis=(1, 2)) /
                area.squeeze(-1)).sum()

    def post_process(self,
                     head_outs,
                     im_shape,
                     scale_factor,
                     infer_shape=[640, 640],
                     rescale=True):
        assert not self.exclude_post_process or not self.exclude_nms
        pred_scores, pred_bboxes, pred_mask_coeffs, mask_feat, anchor_points, stride_tensor = head_outs

        pred_bboxes = batch_distance2bbox(anchor_points, pred_bboxes)
        pred_bboxes *= stride_tensor

        bbox_pred, bbox_num, keep_idxs = self.nms(
            pred_bboxes, pred_scores.transpose([0, 2, 1]))

        if self.with_mask and bbox_num.sum() > 0:
            mask_coeffs = paddle.gather(
                pred_mask_coeffs.reshape([-1, self.num_masks]), keep_idxs)

            mask_logits = process_mask_upsample(mask_feat[0], mask_coeffs,
                                                bbox_pred[:, 2:6], infer_shape)
            if rescale:
                ori_h, ori_w = im_shape[0] / scale_factor[0]
                mask_logits = F.interpolate(
                    mask_logits.unsqueeze(0),
                    size=[
                        custom_ceil(mask_logits.shape[-2] / scale_factor[0][0]),
                        custom_ceil(mask_logits.shape[-1] / scale_factor[0][1])
                    ],
                    mode='bilinear',
                    align_corners=False)[..., :int(ori_h), :int(ori_w)]
            masks = mask_logits.squeeze(0)
            mask_pred = masks > self.mask_thr_binary

            # scale bbox to origin
            scale_factor = scale_factor.flip(-1).tile([1, 2])
            bbox_pred[:, 2:6] /= scale_factor
        else:
            ori_h, ori_w = im_shape[0] / scale_factor[0]
            bbox_num = paddle.to_tensor([1])
            bbox_pred = paddle.zeros([bbox_num, 6])
            mask_pred = paddle.zeros([bbox_num, int(ori_h), int(ori_w)])

        if self.with_mask:
            return bbox_pred, bbox_num, mask_pred
        else:
            return bbox_pred, bbox_num


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.
    """
    assert x.shape[
        -1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
    y = paddle.empty_like(x) if isinstance(x, paddle.Tensor) else np.empty_like(
        x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box

    Args:
      masks (paddle.Tensor): [h, w, n] tensor of masks
      boxes (paddle.Tensor): [n, 4] tensor of bbox coordinates in relative point form

    Returns:
      (paddle.Tensor): The masks are being cropped to the bounding box.
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = paddle.chunk(boxes[:, :, None], 4, axis=1)
    r = paddle.arange(w, dtype=x1.dtype)[None, None, :]
    c = paddle.arange(h, dtype=y1.dtype)[None, :, None]
    if _IS_NPU: # bool tensor broadcast multiply is extreamly slow on npu, so we cast it to float32.
        m_dtype = masks.dtype
        return masks * ((r >= x1).cast(m_dtype) * (r < x2).cast(m_dtype) * (c >= y1).cast(m_dtype) * (c < y2).cast(m_dtype))
    else:
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask_upsample(protos, masks_in, bboxes, shape):
    """
    It takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher
    quality but is slower.

    Args:
      protos (paddle.Tensor): [mask_dim, mask_h, mask_w]
      masks_in (paddle.Tensor): [n, mask_dim], n is number of masks after nms
      bboxes (paddle.Tensor): [n, 4], n is number of masks after nms
      shape (tuple): the size of the input image (h,w)

    Returns:
      (paddle.Tensor): The upsampled masks.
    """
    c, mh, mw = protos.shape  # CHW
    masks = F.sigmoid(masks_in @protos.reshape([c, -1])).reshape([-1, mh, mw])
    masks = F.interpolate(
        masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks
