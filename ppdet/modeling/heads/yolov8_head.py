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
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ..initializer import constant_
from ppdet.core.workspace import register

from ..bbox_utils import batch_distance2bbox
from ..bbox_utils import bbox_iou
from ..assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.backbones.csp_darknet import BaseConv
from ppdet.modeling.layers import MultiClassNMS

__all__ = ['YOLOv8Head']


@register
class YOLOv8Head(nn.Layer):
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms',
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
        self.print_l1_loss = print_l1_loss

        # cls loss
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        # pred head
        c2 = max((16, in_channels[0] // 4, self.reg_max * 4))
        c3 = max(in_channels[0], self.num_classes)
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


        return feats_shapes, cls_logits_list, bbox_preds_list, anchor_points, stride_tensor

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

    def post_process(self, head_outs, im_shape, scale_factor):
        feats_shapes, pred_scores_list, pred_dist_list, anchor_points, stride_tensor = head_outs
        pred_scores = [
            cls_score.transpose([0, 2, 3, 1]).reshape([-1, size, self.num_classes])
            for size, cls_score in zip(feats_shapes, pred_scores_list)
        ]
        pred_dists = [
            bbox_pred.transpose([0, 2, 3, 1]).reshape([-1, size, 4])
            for size, bbox_pred in zip(feats_shapes, pred_dist_list)
        ]
        pred_scores = F.sigmoid(paddle.concat(pred_scores, 1)) # [8, 8400, 80]
        pred_bboxes = paddle.concat(pred_dists, 1) # [8, 8400, 4]

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
