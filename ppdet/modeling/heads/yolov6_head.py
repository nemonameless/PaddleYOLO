# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register

from ..bbox_utils import batch_distance2bbox
from ..losses import GIoULoss, SIoULoss
from ..initializer import bias_init_with_prob, constant_, normal_
from ..assigners.utils import generate_anchors_for_grid_cell
from ..backbones.yolov6_efficientrep import BaseConv, DPBlock
from ppdet.modeling.ops import get_static_shape
from ppdet.modeling.layers import MultiClassNMS
from ..bbox_utils import batch_distance2bbox, bbox_iou, custom_ceil

__all__ = [
    'EffiDeHead', 'EffiDeHead_distill_ns', 'EffiDeHead_fuseab',
    'Lite_EffideHead', 'EffiDeInsHead'
]


@register
class EffiDeHead(nn.Layer):
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms',
        'exclude_post_process', 'self_distill'
    ]
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(
            self,
            in_channels=[128, 256, 512],
            num_classes=80,
            fpn_strides=[8, 16, 32],
            grid_cell_scale=5.0,
            grid_cell_offset=0.5,
            anchors=1,
            reg_max=16,  # reg_max=0 if use_dfl is False
            use_dfl=True,  # False in n/s version, True in m/l version
            static_assigner_epoch=4,  # warmup_epoch
            static_assigner='ATSSAssigner',
            assigner='TaskAlignedAssigner',
            eval_size=None,
            iou_type='giou',  # 'siou' in n version
            loss_weight={
                'cls': 1.0,
                'iou': 2.5,
                'dfl': 0.5,  # used in m/l version 
                'cwd': 10.0,  # used when self_distill=True, in m/l version
            },
            nms='MultiClassNMS',
            trt=False,
            exclude_nms=False,
            exclude_post_process=False,
            self_distill=False,
            distill_weight={
                'cls': 1.0,
                'dfl': 1.0,
            },
            print_l1_loss=True):
        super(EffiDeHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.use_dfl = use_dfl

        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.anchors = anchors

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.eval_size = eval_size
        self.iou_loss = GIoULoss()
        assert iou_type in ['giou', 'siou'], "only support giou and siou loss."
        if iou_type == 'siou':
            self.iou_loss = SIoULoss()
        self.loss_weight = loss_weight

        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process
        self.print_l1_loss = print_l1_loss

        # for self-distillation
        self.self_distill = self_distill
        self.distill_weight = distill_weight

        # Init decouple head
        self.stems = nn.LayerList()
        self.cls_convs = nn.LayerList()
        self.cls_preds = nn.LayerList()
        self.reg_convs = nn.LayerList()
        self.reg_preds = nn.LayerList()

        bias_attr = ParamAttr(regularizer=L2Decay(0.0))
        reg_ch = self.reg_max + self.na
        cls_ch = self.num_classes * self.na
        for in_c in self.in_channels:
            self.stems.append(BaseConv(in_c, in_c, 1, 1))

            self.cls_convs.append(BaseConv(in_c, in_c, 3, 1))
            self.cls_preds.append(
                nn.Conv2D(
                    in_c, cls_ch, 1, bias_attr=bias_attr))

            self.reg_convs.append(BaseConv(in_c, in_c, 3, 1))
            self.reg_preds.append(
                nn.Conv2D(
                    in_c, 4 * reg_ch, 1, bias_attr=bias_attr))

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj_conv = nn.Conv2D(self.reg_max + 1, 1, 1, bias_attr=False)
        self.proj_conv.skip_quant = True

        self.proj = paddle.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj_conv.weight.set_value(
            self.proj.reshape([1, self.reg_max + 1, 1, 1]))
        self.proj_conv.weight.stop_gradient = True
        self.print_l1_loss = print_l1_loss
        self._initialize_biases()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _initialize_biases(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.cls_preds, self.reg_preds):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, 1.0)

        self.proj = paddle.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj_conv.weight.set_value(
            self.proj.reshape([1, self.reg_max + 1, 1, 1]))
        self.proj_conv.weight.stop_gradient = True

        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

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

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            feat = self.stems[i](feat)
            cls_x = feat
            reg_x = feat
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)
            # cls and reg
            cls_output = F.sigmoid(cls_output)
            cls_score_list.append(cls_output.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_output.flatten(2).transpose([0, 2, 1]))

        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)

        return self.get_loss([
            cls_score_list, reg_distri_list, anchors, anchor_points,
            num_anchors_list, stride_tensor
        ], targets)

    def forward_eval(self, feats):
        anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            feat = self.stems[i](feat)
            cls_x = feat
            reg_x = feat
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)

            if self.use_dfl:
                reg_output = reg_output.reshape(
                    [-1, 4, self.reg_max + 1, l]).transpose([0, 2, 1, 3])
                reg_output = self.proj_conv(F.softmax(reg_output, 1))

            # cls and reg
            cls_output = F.sigmoid(cls_output)
            cls_score_list.append(cls_output.reshape([-1, self.num_classes, l]))
            reg_dist_list.append(reg_output.reshape([-1, 4, l]))

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        reg_dist_list = paddle.concat(reg_dist_list, axis=-1)

        return cls_score_list, reg_dist_list, anchor_points, stride_tensor

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

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = F.binary_cross_entropy(
            pred_score, gt_score, weight=weight, reduction='sum')
        return loss

    def _bbox_decode(self, anchor_points, pred_dist):
        ### diff with PPYOLOEHead
        if self.use_dfl:
            b, l, _ = get_static_shape(pred_dist)
            pred_dist = F.softmax(
                pred_dist.reshape([b, l, 4, self.reg_max + 1])).matmul(
                    self.proj.reshape([-1, 1])).squeeze(-1)
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = paddle.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return paddle.concat([lt, rb], -1).clip(0, self.reg_max - 0.01)

    def _df_loss(self, pred_dist, target):
        target_left = paddle.cast(target, 'int64')
        target_right = target_left + 1
        weight_left = target_right.astype('float32') - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist, target_left, reduction='none') * weight_left
        loss_right = F.cross_entropy(
            pred_dist, target_right, reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # iou loss
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            # l1 loss just see the convergence, same in PPYOLOEHead
            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            # dfl loss ### diff with PPYOLOEHead
            if self.use_dfl:
                dist_mask = mask_positive.unsqueeze(-1).tile(
                    [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = paddle.masked_select(
                    pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                assigned_ltrb = self._bbox2distance(anchor_points,
                                                    assigned_bboxes)
                assigned_ltrb_pos = paddle.masked_select(
                    assigned_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos,
                                         assigned_ltrb_pos) * bbox_weight
                loss_dfl = loss_dfl.sum() / assigned_scores_sum
            else:
                loss_dfl = pred_dist.sum() * 0.
        else:
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
            loss_dfl = pred_dist.sum() * 0.
        return loss_l1, loss_iou, loss_dfl

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_distri, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor)
        else:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes)
        # rescale bbox
        assigned_bboxes /= stride_tensor

        # cls loss: varifocal_loss
        one_hot_label = F.one_hot(assigned_labels,
                                  self.num_classes + 1)[..., :-1]
        loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                        one_hot_label)
        assigned_scores_sum = assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum = paddle.clip(
                assigned_scores_sum / paddle.distributed.get_world_size(),
                min=1)
        loss_cls /= assigned_scores_sum

        # bbox loss
        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)

        if self.use_dfl:
            loss = self.loss_weight['cls'] * loss_cls + \
                self.loss_weight['iou'] * loss_iou + \
                self.loss_weight['dfl'] * loss_dfl
            num_gpus = gt_meta.get('num_gpus', 8)
            out_dict = {
                'loss': loss * num_gpus,
                'loss_cls': loss_cls,
                'loss_iou': loss_iou,
                'loss_dfl': loss_dfl,
            }
        else:
            loss = self.loss_weight['cls'] * loss_cls + \
                self.loss_weight['iou'] * loss_iou
            num_gpus = gt_meta.get('num_gpus', 8)
            out_dict = {
                'loss': loss * num_gpus,
                'loss_cls': loss_cls,
                'loss_iou': loss_iou,
            }

        if self.print_l1_loss:
            # just see convergence
            out_dict.update({'loss_l1': loss_l1})
        return out_dict

    def post_process(self,
                     head_outs,
                     im_shape,
                     scale_factor,
                     infer_shape=[640, 640],
                     rescale=True):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points,
                                          pred_dist.transpose([0, 2, 1]))
        pred_bboxes *= stride_tensor

        if self.exclude_post_process:
            return paddle.concat(
                [pred_bboxes, pred_scores.transpose([0, 2, 1])], axis=-1), None
        else:
            # scale bbox to origin
            scale_factor = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
            pred_bboxes /= scale_factor
            if self.exclude_nms:
                # `exclude_nms=True` just use in benchmark
                return pred_bboxes, pred_scores
            else:
                bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
                return bbox_pred, bbox_num


@register
class EffiDeHead_distill_ns(EffiDeHead):
    # add reg_preds_lrtb
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms',
        'exclude_post_process', 'self_distill'
    ]
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(
            self,
            in_channels=[128, 256, 512],
            num_classes=80,
            fpn_strides=[8, 16, 32],
            grid_cell_scale=5.0,
            grid_cell_offset=0.5,
            anchors=1,
            reg_max=16,  # reg_max=0 if use_dfl is False
            use_dfl=True,  # False in n/s version, True in m/l version
            static_assigner_epoch=4,  # warmup_epoch
            static_assigner='ATSSAssigner',
            assigner='TaskAlignedAssigner',
            eval_size=None,
            iou_type='giou',  # 'siou' in n version
            loss_weight={
                'cls': 1.0,
                'iou': 2.5,
                'dfl': 0.5,  # used in m/l version 
                'cwd': 10.0,  # used when self_distill=True, in m/l version
            },
            nms='MultiClassNMS',
            trt=False,
            exclude_nms=False,
            exclude_post_process=False,
            self_distill=False,
            distill_weight={
                'cls': 1.0,
                'dfl': 1.0,
            },
            print_l1_loss=True):
        super(EffiDeHead_distill_ns, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.use_dfl = use_dfl

        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.anchors = anchors

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.eval_size = eval_size
        self.iou_loss = GIoULoss()
        assert iou_type in ['giou', 'siou'], "only support giou and siou loss."
        if iou_type == 'siou':
            self.iou_loss = SIoULoss()
        self.loss_weight = loss_weight

        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process
        self.print_l1_loss = print_l1_loss

        # for self-distillation
        self.self_distill = self_distill
        self.distill_weight = distill_weight

        # Init decouple head
        self.stems = nn.LayerList()
        self.cls_convs = nn.LayerList()
        self.cls_preds = nn.LayerList()
        self.reg_convs = nn.LayerList()
        self.reg_preds = nn.LayerList()
        self.reg_preds_lrtb = nn.LayerList()

        bias_attr = ParamAttr(regularizer=L2Decay(0.0))
        reg_ch = self.reg_max + self.na
        for in_c in self.in_channels:
            self.stems.append(BaseConv(in_c, in_c, 1, 1))

            self.cls_convs.append(BaseConv(in_c, in_c, 3, 1))
            self.cls_preds.append(
                nn.Conv2D(
                    in_c, self.num_classes, 1, bias_attr=bias_attr))

            self.reg_convs.append(BaseConv(in_c, in_c, 3, 1))
            self.reg_preds.append(
                nn.Conv2D(
                    in_c, 4 * reg_ch, 1, bias_attr=bias_attr))

            self.reg_preds_lrtb.append(
                nn.Conv2D(
                    in_c, 4 * self.na, 1, bias_attr=bias_attr))

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj_conv = nn.Conv2D(self.reg_max + 1, 1, 1, bias_attr=False)
        self.proj_conv.skip_quant = True

        self.proj = paddle.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj_conv.weight.set_value(
            self.proj.reshape([1, self.reg_max + 1, 1, 1]))
        self.proj_conv.weight.stop_gradient = True

        self.print_l1_loss = print_l1_loss
        self._initialize_biases()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _initialize_biases(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.cls_preds, self.reg_preds):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, 1.0)

        self.proj = paddle.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj_conv.weight.set_value(
            self.proj.reshape([1, self.reg_max + 1, 1, 1]))
        self.proj_conv.weight.stop_gradient = True

        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

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

        cls_score_list, reg_distri_list, reg_lrtb_list = [], [], []
        for i, feat in enumerate(feats):
            feat = self.stems[i](feat)
            cls_x = feat
            reg_x = feat
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)
            reg_output_lrtb = self.reg_preds_lrtb[i](reg_feat)
            # cls and reg
            cls_output = F.sigmoid(cls_output)
            cls_score_list.append(cls_output.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_output.flatten(2).transpose([0, 2, 1]))
            reg_lrtb_list.append(
                reg_output_lrtb.flatten(2).transpose([0, 2, 1]))

        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)
        reg_lrtb_list = paddle.concat(reg_lrtb_list, axis=1)

        return self.get_loss([
            cls_score_list, reg_distri_list, reg_lrtb_list, anchors,
            anchor_points, num_anchors_list, stride_tensor
        ], targets)

    def forward_eval(self, feats):
        anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_lrtb_list = [], []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            feat = self.stems[i](feat)
            cls_x = feat
            reg_x = feat
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)
            reg_output_lrtb = self.reg_preds_lrtb[i](reg_feat)
            # cls and reg_lrtb 
            cls_output = F.sigmoid(cls_output)
            cls_score_list.append(cls_output.reshape([-1, self.num_classes, l]))
            reg_lrtb_list.append(reg_output_lrtb.reshape([-1, 4, l]))

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        reg_lrtb_list = paddle.concat(reg_lrtb_list, axis=-1)

        return cls_score_list, reg_lrtb_list, anchor_points, stride_tensor

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_distri, pred_ltbrs, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor)
        else:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes)
        # rescale bbox
        assigned_bboxes /= stride_tensor

        # cls loss: varifocal_loss
        one_hot_label = F.one_hot(assigned_labels,
                                  self.num_classes + 1)[..., :-1]
        loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                        one_hot_label)
        assigned_scores_sum = assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum = paddle.clip(
                assigned_scores_sum / paddle.distributed.get_world_size(),
                min=1)
        loss_cls /= assigned_scores_sum

        # bbox loss
        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)

        if self.use_dfl:
            loss = self.loss_weight['cls'] * loss_cls + \
                self.loss_weight['iou'] * loss_iou + \
                self.loss_weight['dfl'] * loss_dfl
            num_gpus = gt_meta.get('num_gpus', 8)
            out_dict = {
                'loss': loss * num_gpus,
                'loss_cls': loss_cls,
                'loss_iou': loss_iou,
                'loss_dfl': loss_dfl,
            }
        else:
            loss = self.loss_weight['cls'] * loss_cls + \
                self.loss_weight['iou'] * loss_iou
            num_gpus = gt_meta.get('num_gpus', 8)
            out_dict = {
                'loss': loss * num_gpus,
                'loss_cls': loss_cls,
                'loss_iou': loss_iou,
            }

        if self.print_l1_loss:
            # just see convergence
            out_dict.update({'loss_l1': loss_l1})
        return out_dict


@register
class EffiDeHead_fuseab(EffiDeHead):
    # add cls_preds_af/reg_preds_af and cls_preds_ab/reg_preds_ab
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms',
        'exclude_post_process', 'self_distill'
    ]
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(
            self,
            in_channels=[128, 256, 512],
            num_classes=80,
            fpn_strides=[8, 16, 32],
            grid_cell_scale=5.0,
            grid_cell_offset=0.5,
            anchors=[[10, 13, 19, 19, 33, 23], [30, 61, 59, 59, 59, 119],
                     [116, 90, 185, 185, 373, 326]],
            reg_max=16,  # reg_max=0 if use_dfl is False
            use_dfl=True,  # False in n/s version, True in m/l version
            static_assigner_epoch=4,  # warmup_epoch
            static_assigner='ATSSAssigner',
            assigner='TaskAlignedAssigner',
            assigner_ab='TaskAlignedAssigner',
            eval_size=None,
            iou_type='giou',  # 'siou' in n version
            loss_weight={
                'cls': 1.0,
                'iou': 2.5,
                'dfl': 0.5,  # used in m/l version 
                'cwd': 10.0,  # used when self_distill=True, in m/l version
            },
            nms='MultiClassNMS',
            trt=False,
            exclude_nms=False,
            exclude_post_process=False,
            self_distill=False,
            distill_weight={
                'cls': 1.0,
                'dfl': 1.0,
            },
            print_l1_loss=True):
        super(EffiDeHead_fuseab, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.use_dfl = use_dfl

        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.anchors = anchors

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.assigner_ab = assigner_ab
        self.eval_size = eval_size
        self.iou_loss = GIoULoss()
        assert iou_type in ['giou', 'siou'], "only support giou and siou loss."
        if iou_type == 'siou':
            self.iou_loss = SIoULoss()
        self.loss_weight = loss_weight

        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process
        self.print_l1_loss = print_l1_loss

        self.anchors_init = (
            paddle.to_tensor(anchors) /
            paddle.to_tensor(self.fpn_strides)[:, None]).reshape(
                [3, self.na, 2])

        # for self-distillation
        self.self_distill = self_distill
        self.distill_weight = distill_weight

        # Init decouple head
        self.stems = nn.LayerList()
        self.cls_convs = nn.LayerList()
        self.cls_preds = nn.LayerList()
        self.reg_convs = nn.LayerList()
        self.reg_preds = nn.LayerList()
        self.reg_preds_lrtb = nn.LayerList()

        self.cls_preds_af = nn.LayerList()
        self.reg_preds_af = nn.LayerList()
        self.cls_preds_ab = nn.LayerList()
        self.reg_preds_ab = nn.LayerList()

        bias_attr = ParamAttr(regularizer=L2Decay(0.0))
        reg_ch = self.reg_max + self.na
        for in_c in self.in_channels:
            self.stems.append(BaseConv(in_c, in_c, 1, 1))

            self.cls_convs.append(BaseConv(in_c, in_c, 3, 1))
            self.cls_preds.append(
                nn.Conv2D(
                    in_c, self.num_classes, 1, bias_attr=bias_attr))

            self.reg_convs.append(BaseConv(in_c, in_c, 3, 1))
            self.reg_preds.append(
                nn.Conv2D(
                    in_c, 4 * reg_ch, 1, bias_attr=bias_attr))

            self.reg_preds_lrtb.append(
                nn.Conv2D(
                    in_c, 4 * self.na, 1, bias_attr=bias_attr))

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj_conv = nn.Conv2D(self.reg_max + 1, 1, 1, bias_attr=False)
        self.proj_conv.skip_quant = True

        self.proj = paddle.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj_conv.weight.set_value(
            self.proj.reshape([1, self.reg_max + 1, 1, 1]))
        self.proj_conv.weight.stop_gradient = True
        self.print_l1_loss = print_l1_loss
        self._initialize_biases()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _initialize_biases(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.cls_preds, self.reg_preds):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, 1.0)

        self.proj = paddle.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj_conv.weight.set_value(
            self.proj.reshape([1, self.reg_max + 1, 1, 1]))
        self.proj_conv.weight.stop_gradient = True

        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

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

        cls_score_list, reg_distri_list, reg_lrtb_list = [], [], []
        for i, feat in enumerate(feats):
            feat = self.stems[i](feat)
            cls_x = feat
            reg_x = feat
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)
            reg_output_lrtb = self.reg_preds_lrtb[i](reg_feat)
            # cls and reg
            cls_output = F.sigmoid(cls_output)
            cls_score_list.append(cls_output.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_output.flatten(2).transpose([0, 2, 1]))
            reg_lrtb_list.append(
                reg_output_lrtb.flatten(2).transpose([0, 2, 1]))

        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)
        reg_lrtb_list = paddle.concat(reg_lrtb_list, axis=1)

        return self.get_loss([
            cls_score_list, reg_distri_list, reg_lrtb_list, anchors,
            anchor_points, num_anchors_list, stride_tensor
        ], targets)

    def forward_eval(self, feats):
        anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_lrtb_list = [], []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            feat = self.stems[i](feat)
            cls_x = feat
            reg_x = feat
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            # reg_output = self.reg_preds[i](reg_feat)
            reg_output_lrtb = self.reg_preds_lrtb[i](reg_feat)
            # cls and reg_lrtb 
            cls_output = F.sigmoid(cls_output)
            cls_score_list.append(cls_output.reshape([-1, self.num_classes, l]))
            reg_lrtb_list.append(reg_output_lrtb.reshape([-1, 4, l]))

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        reg_lrtb_list = paddle.concat(reg_lrtb_list, axis=-1)

        return cls_score_list, reg_lrtb_list, anchor_points, stride_tensor

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_distri, pred_ltbrs, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor)
        else:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes)
        # rescale bbox
        assigned_bboxes /= stride_tensor

        # cls loss: varifocal_loss
        one_hot_label = F.one_hot(assigned_labels,
                                  self.num_classes + 1)[..., :-1]
        loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                        one_hot_label)
        assigned_scores_sum = assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum = paddle.clip(
                assigned_scores_sum / paddle.distributed.get_world_size(),
                min=1)
        loss_cls /= assigned_scores_sum

        # bbox loss
        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)

        if self.use_dfl:
            loss = self.loss_weight['cls'] * loss_cls + \
                self.loss_weight['iou'] * loss_iou + \
                self.loss_weight['dfl'] * loss_dfl
            num_gpus = gt_meta.get('num_gpus', 8)
            out_dict = {
                'loss': loss * num_gpus,
                'loss_cls': loss_cls,
                'loss_iou': loss_iou,
                'loss_dfl': loss_dfl,
            }
        else:
            loss = self.loss_weight['cls'] * loss_cls + \
                self.loss_weight['iou'] * loss_iou
            num_gpus = gt_meta.get('num_gpus', 8)
            out_dict = {
                'loss': loss * num_gpus,
                'loss_cls': loss_cls,
                'loss_iou': loss_iou,
            }

        if self.print_l1_loss:
            # just see convergence
            out_dict.update({'loss_l1': loss_l1})
        return out_dict


@register
class Lite_EffideHead(nn.Layer):
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms', 'exclude_post_process'
    ]
    __inject__ = ['static_assigner', 'assigner', 'nms']
    '''Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    '''

    def __init__(
            self,
            in_channels=[96, 96, 96, 96],
            num_classes=80,
            fpn_strides=[8, 16, 32, 64],
            grid_cell_scale=5.0,
            grid_cell_offset=0.5,
            anchors=1,
            reg_max=0,
            use_dfl=False,
            static_assigner_epoch=4,  # warmup_epoch
            static_assigner='ATSSAssigner',
            assigner='TaskAlignedAssigner',
            eval_size=None,
            iou_type='siou',
            loss_weight={
                'cls': 1.0,
                'iou': 2.5,
            },
            nms='MultiClassNMS',
            trt=False,
            exclude_nms=False,
            exclude_post_process=False,
            print_l1_loss=True):
        super().__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.use_dfl = use_dfl

        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.anchors = anchors

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.eval_size = eval_size
        self.iou_loss = SIoULoss()
        self.loss_weight = loss_weight

        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process

        self.grid = [paddle.zeros([1])] * len(fpn_strides)
        self.prior_prob = 1e-2
        stride = [8, 16, 32] if len(fpn_strides) == 3 else [
            8, 16, 32, 64
        ]  # strides computed during build
        self.stride = paddle.to_tensor(stride)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        # Init decouple head
        self.stems = nn.LayerList()
        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()
        self.cls_preds = nn.LayerList()
        self.reg_preds = nn.LayerList()

        # Efficient decoupled head layers
        bias_attr = ParamAttr(regularizer=L2Decay(0.0))
        self.reg_ch = self.reg_max + self.na
        self.cls_ch = self.num_classes * self.na
        for in_c in self.in_channels:
            self.stems.append(DPBlock(in_c, in_c, 5, 1))

            self.cls_convs.append(DPBlock(in_c, in_c, 5, 1))
            self.cls_preds.append(
                nn.Conv2D(
                    in_c, self.cls_ch, 1, bias_attr=bias_attr))

            self.reg_convs.append(DPBlock(in_c, in_c, 5, 1))
            self.reg_preds.append(
                nn.Conv2D(
                    in_c, 4 * self.reg_ch, 1, bias_attr=bias_attr))

        if self.use_dfl and self.reg_max > 0:
            self.proj_conv = nn.Conv2D(self.reg_max + 1, 1, 1, bias_attr=False)
            self.proj_conv.skip_quant = True
            self.proj = paddle.linspace(0, self.reg_max, self.reg_max + 1)
            self.proj_conv.weight.set_value(
                self.proj.reshape([1, self.reg_max + 1, 1, 1]))
            self.proj_conv.weight.stop_gradient = True
        self.print_l1_loss = print_l1_loss
        self._initialize_biases()

    def _initialize_biases(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.cls_preds, self.reg_preds):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, 1.0)

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

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            bs, _, h, w = feat.shape
            l = h * w
            feat = self.stems[i](feat)
            cls_x = feat
            reg_x = feat
            cls_output = self.cls_preds[i](self.cls_convs[i](cls_x))
            reg_output = self.reg_preds[i](self.reg_convs[i](reg_x))

            cls_output = F.sigmoid(cls_output)
            cls_score_list.append(cls_output.reshape([-1, self.num_classes, l]))
            reg_distri_list.append(reg_output.reshape([-1, 4 * self.reg_ch, l]))

        cls_scores = paddle.concat(cls_score_list, axis=-1).transpose([0, 2, 1])
        reg_dists = paddle.concat(reg_distri_list, axis=-1).transpose([0, 2, 1])

        return self.get_loss([
            cls_scores, reg_dists, anchors, anchor_points, num_anchors_list,
            stride_tensor
        ], targets)

    def forward_eval(self, feats):
        anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            feat = self.stems[i](feat)
            cls_x = feat
            reg_x = feat
            cls_output = self.cls_preds[i](self.cls_convs[i](cls_x))
            reg_output = self.reg_preds[i](self.reg_convs[i](reg_x))

            cls_output = F.sigmoid(cls_output)
            cls_score_list.append(cls_output.reshape([-1, self.num_classes, l]))
            reg_dist_list.append(reg_output.reshape([-1, 4, l]))

        cls_scores = paddle.concat(cls_score_list, axis=-1).transpose([0, 2, 1])
        reg_dists = paddle.concat(reg_dist_list, axis=-1).transpose([0, 2, 1])
        return cls_scores, reg_dists, anchor_points, stride_tensor

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

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = F.binary_cross_entropy(
            pred_score, gt_score, weight=weight, reduction='sum')
        return loss

    def _bbox_decode(self, anchor_points, pred_dist):
        ### diff with PPYOLOEHead
        if self.use_dfl:
            b, l, _ = get_static_shape(pred_dist)
            pred_dist = F.softmax(
                pred_dist.reshape([b, l, 4, self.reg_max + 1])).matmul(
                    self.proj.reshape([-1, 1])).squeeze(-1)
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = paddle.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return paddle.concat([lt, rb], -1).clip(0, self.reg_max - 0.01)

    def _df_loss(self, pred_dist, target):
        target_left = paddle.cast(target, 'int64')
        target_right = target_left + 1
        weight_left = target_right.astype('float32') - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist, target_left, reduction='none') * weight_left
        loss_right = F.cross_entropy(
            pred_dist, target_right, reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # iou loss
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            # l1 loss just see the convergence, same in PPYOLOEHead
            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            # dfl loss ### diff with PPYOLOEHead
            if self.use_dfl:
                dist_mask = mask_positive.unsqueeze(-1).tile(
                    [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = paddle.masked_select(
                    pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                assigned_ltrb = self._bbox2distance(anchor_points,
                                                    assigned_bboxes)
                assigned_ltrb_pos = paddle.masked_select(
                    assigned_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos,
                                         assigned_ltrb_pos) * bbox_weight
                loss_dfl = loss_dfl.sum() / assigned_scores_sum
            else:
                loss_dfl = pred_dist.sum() * 0.
        else:
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
            loss_dfl = pred_dist.sum() * 0.
        return loss_l1, loss_iou, loss_dfl

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_distri, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor)
        else:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes)
        # rescale bbox
        assigned_bboxes /= stride_tensor

        # cls loss: varifocal_loss
        one_hot_label = F.one_hot(assigned_labels,
                                  self.num_classes + 1)[..., :-1]
        loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                        one_hot_label)
        assigned_scores_sum = assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum = paddle.clip(
                assigned_scores_sum / paddle.distributed.get_world_size(),
                min=1)
        loss_cls /= assigned_scores_sum

        # bbox loss, no need loss_dfl
        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)

        loss = self.loss_weight['cls'] * loss_cls + \
                self.loss_weight['iou'] * loss_iou
        num_gpus = gt_meta.get('num_gpus', 8)
        out_dict = {
            'loss': loss * num_gpus,
            'loss_cls': loss_cls * num_gpus,
            'loss_iou': loss_iou * num_gpus
        }
        if self.print_l1_loss:
            # just see convergence
            out_dict.update({'loss_l1': loss_l1 * num_gpus})
        return out_dict

    def post_process(self,
                     head_outs,
                     im_shape,
                     scale_factor,
                     infer_shape=[640, 640],
                     rescale=True):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist)
        pred_bboxes *= stride_tensor

        if self.exclude_post_process:
            return paddle.concat([pred_bboxes, pred_scores], axis=-1), None
        else:
            # scale bbox to origin
            scale_factor = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
            pred_bboxes /= scale_factor
            if self.exclude_nms:
                # `exclude_nms=True` just use in benchmark
                return pred_bboxes, pred_scores.transpose([0, 2, 1])
            else:
                bbox_pred, bbox_num, _ = self.nms(
                    pred_bboxes, pred_scores.transpose([0, 2, 1]))
                return bbox_pred, bbox_num


class MaskProto(nn.Layer):
    # YOLOv6 mask Proto module for instance segmentation models
    def __init__(self,
                 ch_in,
                 num_protos=256,
                 num_masks=32,
                 act='silu',
                 scale_factor=2):
        super().__init__()
        self.conv1 = BaseConv(ch_in, num_protos, 3, 1, act=act)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv2 = BaseConv(num_protos, num_protos, 3, 1, act=act)
        self.conv3 = BaseConv(num_protos, num_masks, 1, 1, act=act)

    def forward(self, x):
        return self.conv3(self.conv2(self.upsample(self.conv1(x))))


@register
class EffiDeInsHead(nn.Layer):
    __shared__ = [
        'num_classes', 'eval_size', 'act', 'trt', 'exclude_nms',
        'exclude_post_process', 'with_mask', 'width_mult'
    ]
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(
            self,
            with_mask=True,
            in_channels=[128, 256, 512],
            num_classes=80,
            issolo=False,
            num_masks=32,
            num_protos=256,
            width_mult=1.0,
            act='silu',
            fpn_strides=[8, 16, 32],
            grid_cell_scale=5.0,
            grid_cell_offset=0.5,
            anchors=1,
            reg_max=16,  # reg_max=0 if use_dfl is False
            use_dfl=True,  # False in n/s version, True in m/l version
            static_assigner_epoch=0,  # warmup_epoch
            static_assigner='ATSSAssigner',
            assigner='TaskAlignedAssigner',
            nms='MultiClassNMS',
            eval_size=[640, 640],
            iou_type='giou',  # 'siou' in n version
            loss_weight={
                'cls': 1.0,
                'iou': 2.5,
                'dfl': 0.5,  # used in m/l version 
            },
            trt=False,
            exclude_nms=False,
            exclude_post_process=False,
            stack_protos=1,
            mask_thr_binary=0.5,
            print_l1_loss=True):
        super(EffiDeInsHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.with_mask = with_mask
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.use_dfl = use_dfl

        self.issolo = issolo
        self.num_masks = num_masks
        self.width_mult = width_mult
        self.act = act
        self.num_protos = int(num_protos * width_mult)
        self.stack_protos = stack_protos
        self.mask_thr_binary = mask_thr_binary

        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.anchors = anchors

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.eval_size = eval_size
        self.iou_loss = GIoULoss()
        assert iou_type in ['giou', 'siou'], "only support giou and siou loss."
        if iou_type == 'siou':
            self.iou_loss = SIoULoss()
        self.loss_weight = loss_weight

        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process
        self.print_l1_loss = print_l1_loss

        # Init decouple head
        self.stems = nn.LayerList()
        self.cls_convs = nn.LayerList()
        self.cls_preds = nn.LayerList()
        self.reg_convs = nn.LayerList()
        self.reg_preds = nn.LayerList()
        self.seg_convs = nn.LayerList()
        self.seg_preds = nn.LayerList()

        bias_attr = ParamAttr(regularizer=L2Decay(0.0))
        self.reg_ch = self.reg_max + self.na
        self.cls_ch = self.num_classes * self.na
        for i, in_c in enumerate(self.in_channels):
            self.stems.append(BaseConv(in_c, in_c, 1, 1))

            self.cls_convs.append(BaseConv(in_c, in_c, 3, 1))
            self.cls_preds.append(
                nn.Conv2D(
                    in_c, self.cls_ch, 1, bias_attr=bias_attr))

            self.reg_convs.append(BaseConv(in_c, in_c, 3, 1))
            self.reg_preds.append(
                nn.Conv2D(
                    in_c, 4 * self.reg_ch, 1, bias_attr=bias_attr))

            self.seg_convs.append(BaseConv(in_c, in_c, 3, 1))
            if not self.issolo:
                self.seg_preds.append(
                    nn.Conv2D(
                        in_c, self.num_masks, 1, bias_attr=bias_attr))
            else:
                self.seg_preds.append(
                    nn.Conv2D(
                        in_c, self.num_masks + 2 + 1, 1, bias_attr=bias_attr))

        self.seg_proto = nn.LayerList()
        if self.issolo:
            self.seg_proto.append(
                MaskProto(
                    self.in_channels[0],
                    self.num_protos,
                    self.num_masks,
                    act='silu'))
            self.seg_proto.append(
                MaskProto(
                    self.in_channels[1],
                    self.num_protos,
                    self.num_masks,
                    act='silu',
                    scale_factor=4))
            self.seg_proto.append(
                MaskProto(
                    self.in_channels[2],
                    self.num_protos,
                    self.num_masks,
                    act='silu',
                    scale_factor=8))
        else:
            self.seg_proto.append(
                MaskProto(
                    self.in_channels[0],
                    self.num_protos,
                    self.num_masks,
                    act='silu'))

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        if self.use_dfl and self.reg_max > 0:
            self.proj_conv = nn.Conv2D(self.reg_max + 1, 1, 1, bias_attr=False)
            self.proj_conv.skip_quant = True
            self.proj = paddle.linspace(0, self.reg_max, self.reg_max + 1)
            self.proj_conv.weight.set_value(
                self.proj.reshape([1, self.reg_max + 1, 1, 1]))
            self.proj_conv.weight.stop_gradient = True
        self.print_l1_loss = print_l1_loss
        self._initialize_biases()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _initialize_biases(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.cls_preds, self.reg_preds):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, 1.0)

        if self.use_dfl and self.reg_max > 0:
            self.proj = paddle.linspace(0, self.reg_max, self.reg_max + 1)
            self.proj_conv.weight.set_value(
                self.proj.reshape([1, self.reg_max + 1, 1, 1]))
            self.proj_conv.weight.stop_gradient = True

        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

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

        seg_feats = []
        seg_mask = self.seg_proto[0](feats[0])
        seg_feats.append(seg_mask)  # [[1, 32, 160, 160]]

        cls_score_list, reg_distri_list, seg_conf_list = [], [], []
        for i, feat in enumerate(feats):
            bs, _, h, w = feat.shape
            l = h * w
            feat = self.stems[i](feat)
            cls_x = feat
            reg_x = feat
            seg_x = feat
            cls_output = self.cls_preds[i](self.cls_convs[i](cls_x))
            reg_output = self.reg_preds[i](self.reg_convs[i](reg_x))
            seg_output = self.seg_preds[i](self.seg_convs[i](seg_x))

            # cls and reg
            cls_output = F.sigmoid(cls_output)
            cls_score_list.append(cls_output.reshape([-1, self.num_classes, l]))
            reg_distri_list.append(reg_output.reshape([-1, 4 * self.reg_ch, l]))
            seg_conf_list.append(seg_output.reshape([-1, self.num_masks, l]))

        cls_scores = paddle.concat(cls_score_list, axis=-1).transpose([0, 2, 1])
        reg_dists = paddle.concat(reg_distri_list, axis=-1).transpose([0, 2, 1])
        seg_confs = paddle.concat(seg_conf_list, axis=-1).transpose([0, 2, 1])

        return self.get_loss([
            cls_scores, reg_dists, seg_confs, seg_feats[0], anchors,
            anchor_points, num_anchors_list, stride_tensor
        ], targets)

    def forward_eval(self, feats):
        seg_feats = []
        seg_mask = self.seg_proto[0](feats[0])
        seg_feats.append(seg_mask)  # [[1, 32, 160, 160]]

        anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list, seg_conf_list = [], [], []
        for i, feat in enumerate(feats):
            bs, _, h, w = feat.shape
            l = h * w
            feat = self.stems[i](feat)
            cls_x = feat
            reg_x = feat
            seg_x = feat
            cls_output = self.cls_preds[i](self.cls_convs[i](cls_x))
            reg_output = self.reg_preds[i](self.reg_convs[i](reg_x))
            seg_output = self.seg_preds[i](self.seg_convs[i](seg_x))

            if self.use_dfl:
                reg_output = reg_output.reshape(
                    [-1, 4, self.reg_max + 1, l]).transpose([0, 2, 1, 3])
                reg_output = self.proj_conv(F.softmax(reg_output, 1))

            proto_no = paddle.ones([bs, 1, l])  #* (2-i)

            # cls and reg
            cls_output = F.sigmoid(cls_output)
            cls_score_list.append(cls_output.reshape([-1, self.num_classes, l]))
            reg_dist_list.append(reg_output.reshape([-1, 4, l]))
            seg_conf_list.append(
                paddle.concat(
                    [proto_no, seg_output.reshape([-1, self.num_masks, l])], 1))

        cls_scores = paddle.concat(cls_score_list, axis=-1).transpose([0, 2, 1])
        reg_dists = paddle.concat(reg_dist_list, axis=-1).transpose([0, 2, 1])
        seg_confs = paddle.concat(seg_conf_list, axis=-1).transpose([0, 2, 1])
        # [1, 8400, 80]  [1, 8400, 4]  [1, 8400, 33]

        return cls_scores, reg_dists, seg_confs, seg_feats, anchor_points, stride_tensor
        # pred_bboxes = batch_distance2bbox(anchor_points, reg_dists)
        # pred_bboxes *= stride_tensor
        # predictions = paddle.concat([pred_bboxes, paddle.ones([bs, pred_bboxes.shape[1], 1]), cls_scores], axis=-1)
        # return predictions, seg_feats, seg_confs

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

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = F.binary_cross_entropy(
            pred_score, gt_score, weight=weight, reduction='sum')
        return loss

    def _bbox_decode(self, anchor_points, pred_dist):
        ### diff with PPYOLOEHead
        if self.use_dfl:
            b, l, _ = get_static_shape(pred_dist)
            pred_dist = F.softmax(
                pred_dist.reshape([b, l, 4, self.reg_max + 1])).matmul(
                    self.proj.reshape([-1, 1])).squeeze(-1)
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = paddle.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return paddle.concat([lt, rb], -1).clip(0, self.reg_max - 0.01)

    def _df_loss(self, pred_dist, target):
        target_left = paddle.cast(target, 'int64')
        target_right = target_left + 1
        weight_left = target_right.astype('float32') - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist, target_left, reduction='none') * weight_left
        loss_right = F.cross_entropy(
            pred_dist, target_right, reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # iou loss
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            # l1 loss just see the convergence, same in PPYOLOEHead
            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            # dfl loss ### diff with PPYOLOEHead
            if self.use_dfl:
                dist_mask = mask_positive.unsqueeze(-1).tile(
                    [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = paddle.masked_select(
                    pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                assigned_ltrb = self._bbox2distance(anchor_points,
                                                    assigned_bboxes)
                assigned_ltrb_pos = paddle.masked_select(
                    assigned_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos,
                                         assigned_ltrb_pos) * bbox_weight
                loss_dfl = loss_dfl.sum() / assigned_scores_sum
            else:
                loss_dfl = pred_dist.sum() * 0.
        else:
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
            loss_dfl = pred_dist.sum() * 0.
        return loss_l1, loss_iou, loss_dfl

    def get_loss(self, head_outs, gt_meta):
        assert 'gt_bbox' in gt_meta and 'gt_class' in gt_meta
        assert 'gt_segm' in gt_meta

        pred_scores, pred_distri, pred_mask_coeff, mask_proto, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs

        bs = pred_scores.shape[0]
        imgsz = paddle.to_tensor(
            [640, 640]
        )  # paddle.to_tensor(pred_scores[0].shape[2:]) * self.fpn_strides[0]
        # image size (h,w)
        mask_h, mask_w = mask_proto.shape[-2:]

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = paddle.stack(gt_meta['gt_class'])
        gt_bboxes = paddle.stack(gt_meta['gt_bbox'])  # xyxy
        pad_gt_mask = paddle.stack(gt_meta['pad_gt_mask'])
        gt_segms = paddle.stack(gt_meta['gt_segm']).cast('float32')

        if tuple(gt_segms.shape[-2:]) != (mask_h, mask_w):  # downsample
            gt_segms = F.interpolate(
                gt_segms, (mask_h, mask_w),
                mode='nearest').reshape([bs, -1, mask_h * mask_w])

        # label assignment, only TAL
        if 1:
            assigned_labels, assigned_bboxes, assigned_scores, assigned_gt_index = \
                self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes,
                gt_segms=gt_segms)
        # rescale bbox
        assigned_bboxes /= stride_tensor
        #pred_bboxes /= stride_tensor ###

        # assign segms for masks
        assigned_masks = paddle.gather(
            gt_segms.reshape([-1, mask_h * mask_w]),
            assigned_gt_index.flatten(),
            axis=0)
        assigned_masks = assigned_masks.reshape(
            [bs, assigned_gt_index.shape[1], mask_h * mask_w])

        # cls loss: varifocal_loss, not bce
        one_hot_label = F.one_hot(assigned_labels,
                                  self.num_classes + 1)[..., :-1]
        loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                        one_hot_label)
        assigned_scores_sum = assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum /= paddle.distributed.get_world_size()
        assigned_scores_sum = paddle.clip(assigned_scores_sum, min=1.)
        loss_cls /= assigned_scores_sum

        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        if num_pos > 0:
            # siou/giou loss
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
                giou=True,
                eps=1e-7)
            loss_iou = ((1.0 - iou) * bbox_weight).sum() / assigned_scores_sum

            if self.print_l1_loss:
                loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)
            else:
                loss_l1 = paddle.zeros([1])

            # dfl loss ### diff with PPYOLOEHead
            if self.use_dfl:
                dist_mask = mask_positive.unsqueeze(-1).tile(
                    [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = paddle.masked_select(
                    pred_distri, dist_mask).reshape([-1, 4, self.reg_max + 1])
                assigned_ltrb = self._bbox2distance(anchor_points,
                                                    assigned_bboxes)
                assigned_ltrb_pos = paddle.masked_select(
                    assigned_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos,
                                         assigned_ltrb_pos) * bbox_weight
                loss_dfl = loss_dfl.sum() / assigned_scores_sum
            else:
                loss_dfl = pred_distri.sum() * 0.

            # mask loss
            mask_positive = (assigned_labels != self.num_classes)
            loss_mask = self.calculate_segmentation_loss(
                mask_positive, assigned_gt_index, assigned_masks,
                assigned_bboxes * stride_tensor, mask_proto, pred_mask_coeff,
                imgsz)
            # [bs, 8400] [bs, 8400] [bs, 8400, 160 * 160] [bs, 8400, 4] [bs, 32, 160, 160] [bs, 8400, 32]
            loss_mask /= assigned_scores_sum
        else:
            loss_iou = paddle.to_tensor([0])
            loss_dfl = paddle.to_tensor([0])
            loss_mask = paddle.to_tensor([0])
            loss_l1 = paddle.to_tensor([0])

        loss_cls *= self.loss_weight['cls']
        loss_iou *= self.loss_weight['iou']
        loss_mask *= self.loss_weight['iou']  # same as iou
        loss_total = loss_cls + loss_iou + loss_dfl + loss_mask
        num_gpus = gt_meta.get('num_gpus', 8)
        out_dict = {
            'loss': loss_total * num_gpus,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_mask': loss_mask,
        }
        if self.use_dfl:
            loss_dfl *= self.loss_weight['dfl']
            out_dict['loss'] += loss_dfl * num_gpus
            out_dict.update({'loss_dfl': loss_dfl})

        if self.print_l1_loss:
            # just see convergence
            out_dict.update({'loss_l1': loss_l1})
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

    def post_process_align(self,
                           head_outs,
                           im_shape,
                           scale_factor,
                           infer_shape=[640, 640],
                           gt_labels=None,
                           gt_bboxes=None,
                           gt_masks=None):
        assert not self.exclude_post_process or not self.exclude_nms

        cls_scores, reg_dists, seg_confs, seg_feats, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points, reg_dists)
        pred_bboxes *= stride_tensor
        bs = cls_scores.shape[0]
        predictions = paddle.concat(
            [
                pred_bboxes, paddle.ones([bs, pred_bboxes.shape[1], 1]),
                cls_scores
            ],
            axis=-1)

        conf_thres, iou_thres, max_det = 0.01, 0.65, 300  # 0.03
        loutputs = non_max_suppression_seg(
            [predictions, seg_feats, seg_confs],
            conf_thres,
            iou_thres,
            classes=None,
            agnostic=False,
            multi_label=False,
            max_det=max_det)

        if len(loutputs) > 0:
            segments = []
            segconf = [loutputs[li][..., 0:]
                       for li in range(len(loutputs))]  # [[N, 39]]
            bbox_pred = [loutputs[li][..., :6]
                         for li in range(len(loutputs))]  # [[N, 6]]
            protos = seg_feats[0]  # [1, 32, 160, 160]
            #if not self.issolo:
            segments = [
                self.handle_proto_test(
                    [protos[li].reshape([1, *(protos[li].shape[-3:])])],
                    segconf[li], infer_shape) for li in range(len(loutputs))
            ]
            if 0:  #gt_masks:
                #import copy
                #eval_outputs = copy.deepcopy([x.detach().cpu() for x in bbox_pred])
                pred, pred_masks = bbox_pred[0], segments[0]
                iouv = paddle.linspace(0.5, 0.95, 10)
                niou = iouv.numel()
                correct_masks = paddle.zeros([len(pred), niou])  # init
                correct = paddle.zeros(len(pred), niou)  # init
                predn = pred.clone()
                self.scale_coords(infer_shape, predn[:, :4], im_shape[0][0],
                                  im_shape[0][1])  # native-space pred

                # target boxes
                # tbox = xywh2xyxy(gt_bboxes)
                # tbox[:, [0, 2]] *= imgs[si].shape[1:][1]
                # tbox[:, [1, 3]] *= imgs[si].shape[1:][0]
                tbox = gt_bboxes.clone()

                self.scale_coords(infer_shape, tbox, im_shape[0][0],
                                  im_shape[0][1])  # native-space labels
                labelsn = paddle.concat((gt_labels, tbox),
                                        1)  # native-space labels
                bbox_pred = process_batch(predn, labelsn, iouv)
                mask_pred = process_batch(
                    predn,
                    labelsn,
                    iouv,
                    pred_masks,
                    gt_masks,
                    overlap=False,
                    masks=True)
                bbox_num = paddle.to_tensor([bbox_pred.shape[0]])

            else:
                mask_logits = segments[0].cast('float32')  # [300, 640, 640]
                ori_h, ori_w = im_shape[0] / scale_factor[0]

                #masks = self.rescale_mask(infer_shape, mask_logits, (ori_h, ori_w))
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
                bbox_pred = bbox_pred[0]
                bbox_pred = paddle.concat(
                    [bbox_pred[:, 5:6], bbox_pred[:, 4:5], bbox_pred[:, :4]],
                    axis=1)
                bbox_pred[:, 2:6] /= scale_factor
                bbox_num = paddle.to_tensor([bbox_pred.shape[0]])
        else:
            ori_h, ori_w = im_shape[0] / scale_factor[0]
            bbox_num = paddle.to_tensor([1])
            bbox_pred = paddle.zeros([bbox_num, 6])
            mask_pred = paddle.zeros([bbox_num, int(ori_h), int(ori_w)])

        if self.with_mask:
            return bbox_pred, bbox_num, mask_pred
        else:
            return bbox_pred, bbox_num

    @staticmethod
    def rescale_mask(ori_shape, masks, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0],
                    ori_shape[1] / target_shape[1])
        padding = int((ori_shape[1] - target_shape[1] * ratio) / 2), int(
            (ori_shape[0] - target_shape[0] * ratio) / 2)

        masks = masks[:, 0:ori_shape[0] - padding[1] * 2, padding[0]:ori_shape[
            1] - padding[0]]
        #masks = masks[:, padding[1]: ori_shape[0]- padding[1], padding[0]: ori_shape[1] - padding[0]]
        # masks = masks.transpose([1, 2, 0])
        # masks = cv2.resize(masks, target_shape[:2][::-1])
        # if len(masks.shape) == 2:
        #     masks = masks.reshape([*masks.shape, 1])
        return masks

    def handle_proto_test(self, proto_list, oconfs, imgshape):
        conf = oconfs[..., 6:]
        if conf.shape[0] == 0:
            return None

        xyxy = oconfs[..., :4]
        confs = conf[..., 1:]
        proto = proto_list[0]
        s = proto.shape[-2:]
        seg = ((confs
                @proto.reshape([proto.shape[0], proto.shape[1], -1])).reshape(
                    [proto.shape[0], confs.shape[0], *s]))
        seg = F.sigmoid(seg)
        masks = F.interpolate(
            seg, imgshape, mode='bilinear', align_corners=False)[0]
        masks = crop_mask(masks, xyxy)  # > 0.5
        return masks  #.unsqueeze(0)

    def post_process(self,
                     head_outs,
                     im_shape,
                     scale_factor,
                     infer_shape=[640, 640],
                     gt_labels=None,
                     gt_bboxes=None,
                     gt_masks=None,
                     rescale=True):
        assert not self.exclude_post_process or not self.exclude_nms
        pred_scores, pred_bboxes, seg_conf, seg_feats, anchor_points, stride_tensor = head_outs
        seg_conf = seg_conf[:, :, 1:]
        pred_bboxes = batch_distance2bbox(anchor_points, pred_bboxes)
        pred_bboxes *= stride_tensor
        # [1, 8400, 4]

        bbox_pred, bbox_num, keep_idxs = self.nms(
            pred_bboxes, pred_scores.transpose([0, 2, 1]))

        if self.with_mask and bbox_num.sum() > 0:
            mask_coeffs = paddle.gather(
                seg_conf.reshape([-1, self.num_masks]), keep_idxs)

            mask_logits = process_mask_upsample(seg_feats[0][0], mask_coeffs,
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


def non_max_suppression_seg(predictions,
                            conf_thres=0.25,
                            iou_thres=0.45,
                            classes=None,
                            agnostic=False,
                            multi_label=False,
                            max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
    Args:
        prediction: (tensor), with shape [N, 5 + num_classes], N is the number of bboxes.
        conf_thres: (float) confidence threshold.
        iou_thres: (float) iou threshold.
        classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
        agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
        multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
        max_det:(int), max number of output bboxes.

    Returns:
         list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
    """
    prediction = predictions[
        0]  # [1, 8400, 85] [[1, 32, 160, 160]] [1, 8400, 33]
    #import pdb; pdb.set_trace()
    confs = predictions[2]  # (bs, which_proto, fs)
    prediction = paddle.concat(
        [prediction, confs], axis=2)  # (bs, l ,5 + num_classes + 33)

    num_classes = prediction.shape[2] - 5 - 33  # number of classes
    pred_candidates = paddle.logical_and(
        prediction[..., 4] > conf_thres,
        paddle.max(prediction[..., 5:5 + num_classes], axis=-1)[0] >
        conf_thres)  # candidates
    # Check the parameters.
    assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'
    assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'

    # Function settings.
    max_wh = 4096  # maximum box width and height
    max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
    multi_label &= num_classes > 1  # multiple labels per box

    output = [paddle.zeros((0, 6 + 33))] * prediction.shape[0]
    for img_idx, x in enumerate(prediction):  # image index, image inference
        x = x[pred_candidates[img_idx]]  # confidence
        # [138, 118]

        # If no box remains, skip the next process.
        if not x.shape[0]:
            continue

        # confidence multiply the objectness
        x[:, 5:5 + num_classes] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # (center x, center y, width, height) to (x1, y1, x2, y2)
        #box = xywh2xyxy(x[:, :4])
        box = x[:, :4]
        segconf = x[:, 5 + num_classes:]

        # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
        if multi_label:
            box_idx, class_idx = (x[:, 5:5 + num_classes] > conf_thres).nonzero(
                as_tuple=False).T
            x = paddle.concat(
                (box[box_idx], x[box_idx, class_idx + 5, None],
                 class_idx[:, None].cast('float32'), segconf[box_idx]), 1)
        else:  # Only keep the class with highest scores.
            conf = x[:, 5:5 + num_classes].max(1, keepdim=True)
            class_idx = x[:, 5:5 + num_classes].argmax(1, keepdim=True)
            x = paddle.concat((box, conf, class_idx.cast('float32'), segconf),
                              1)[conf.reshape([-1]) > conf_thres]

        # Filter by class, only keep boxes whose category is in classes.
        # if classes is not None:
        #     x = x[(x[:, 5:6] == paddle.to_tensor(classes)).any(1)]

        # Check shape
        num_box = x.shape[0]  # number of boxes
        if not num_box:  # no boxes kept.
            continue
        elif num_box > max_nms:  # excess max boxes' number.
            x = x[x[:, 4].argsort(
                descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :
                          4] + class_offset, x[:,
                                               4]  # boxes (offset by class), scores
        keep_box_idx = paddle.vision.ops.nms(boxes, iou_thres,
                                             scores=scores)  # NMS
        if keep_box_idx.shape[0] > max_det:  # limit detections
            keep_box_idx = keep_box_idx[:max_det]

        output[img_idx] = x[keep_box_idx]

    return output  # [[21, 39]]


def xywh2xyxy(x):
    '''Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right.'''
    y = x.clone()  #if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


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


def process_batch(detections,
                  labels,
                  iouv,
                  pred_masks=None,
                  gt_masks=None,
                  overlap=False,
                  masks=False):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    if masks:
        if overlap:
            nl = len(labels)
            index = paddle.arange(nl).reshape([nl, 1, 1]) + 1
            gt_masks = gt_masks.repeat(nl, 1,
                                       1)  # shape(1,640,640) -> (n,640,640)
            gt_masks = paddle.where(gt_masks == index, 1.0, 0.0)
        if gt_masks.shape[1:] != pred_masks.shape[1:]:
            gt_masks = F.interpolate(
                gt_masks[None].cast('float32'),
                pred_masks.shape[1:],
                mode='bilinear',
                align_corners=False)[0]
            gt_masks = gt_masks.gt_(0.5)
        iou = mask_iou(
            gt_masks.reshape([gt_masks.shape[0], -1]).cast('float32'),
            pred_masks.reshape([pred_masks.shape[0], -1]))
    else:  # boxes
        iou = box_iou(labels[:, 1:], detections[:, :4])

    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = paddle.where((iou >= iouv[i]) &
                         correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = paddle.concat(
                (paddle.stack(x, 1), iou[x[0], x[1]][:, None]),
                1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(
                    matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(
                    matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return paddle.to_tensor(correct, dtype='bool')


def mask_iou(mask1, mask2, eps=1e-7):
    """
    mask1: [N, n] m1 means number of predicted objects
    mask2: [M, n] m2 means number of gt objects
    Note: n means image_w x image_h

    return: masks iou, [N, M]
    """
    mask1 = mask1.cast('float32')
    intersection = paddle.matmul(mask1, mask2.t()).clip(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]
             ) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(
        2, 2)
    inter = (paddle.minimum(a2, b2) - paddle.maximum(a1, b1)).clip(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
