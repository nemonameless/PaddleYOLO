# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from ppdet.core.workspace import register
from ppdet.modeling.backbones.yolov7_elannet import ImplicitA, ImplicitM
from ppdet.modeling.layers import MultiClassNMS

from ppdet.modeling.bbox_utils import batch_distance2bbox
from ppdet.modeling.bbox_utils import bbox_iou
from ppdet.modeling.assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.backbones.csp_darknet import BaseConv
from ppdet.modeling.ops import get_static_shape
from ppdet.modeling.layers import MultiClassNMS

__all__ = ['YOLOv7Head', 'YOLOv7uHead']


@register
class YOLOv7Head(nn.Layer):
    __shared__ = [
        'num_classes', 'data_format', 'use_aux', 'use_implicit', 'trt',
        'exclude_nms', 'exclude_post_process'
    ]
    __inject__ = ['loss', 'nms']

    def __init__(self,
                 num_classes=80,
                 in_channels=[256, 512, 1024],
                 anchors=[[12, 16], [19, 36], [40, 28], [36, 75], [76, 55],
                          [72, 146], [142, 110], [192, 243], [459, 401]],
                 anchor_masks=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                 stride=[8, 16, 32],
                 use_aux=False,
                 use_implicit=False,
                 loss='YOLOv7Loss',
                 data_format='NCHW',
                 nms='MultiClassNMS',
                 trt=False,
                 exclude_post_process=False,
                 exclude_nms=False):
        """
        Head for YOLOv7

        Args:
            num_classes (int): number of foreground classes
            in_channels (int): channels of input features
            anchors (list): anchors
            anchor_masks (list): anchor masks
            stride (list): strides
            use_aux (bool): whether to use Aux Head, only in P6 models
            use_implicit (bool): whether to use ImplicitA and ImplicitM
            loss (object): YOLOv7Loss instance
            data_format (str): nms format, NCHW or NHWC
            nms (object): MultiClassNMS instance
            trt (bool): whether to use trt infer
            exclude_nms (bool): whether to use exclude_nms for speed test
        """
        super(YOLOv7Head, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.parse_anchor(anchors, anchor_masks)
        self.anchors = paddle.to_tensor(self.anchors, dtype='int32')
        self.anchor_levels = len(self.anchors)

        self.stride = stride
        self.use_aux = use_aux
        self.use_implicit = use_implicit
        self.loss = loss
        self.data_format = data_format
        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process

        self.num_anchor = len(self.anchors[0])  # self.na
        self.num_out_ch = self.num_classes + 5  # self.no

        self.yolo_outputs = []
        if self.use_aux:
            self.yolo_outputs_aux = []
        if self.use_implicit:
            self.ia, self.im = [], []
        self.num_levels = len(self.anchors)
        for i in range(self.num_levels):
            num_filters = self.num_anchor * self.num_out_ch
            name = 'yolo_output.{}'.format(i)
            conv = nn.Conv2D(
                in_channels=self.in_channels[i],
                out_channels=num_filters,
                kernel_size=1,
                stride=1,
                padding=0,
                data_format=data_format,
                bias_attr=ParamAttr(regularizer=L2Decay(0.)))
            conv.skip_quant = True
            yolo_output = self.add_sublayer(name, conv)
            self.yolo_outputs.append(yolo_output)

            if self.use_aux:
                name_aux = 'yolo_output_aux.{}'.format(i)
                conv_aux = nn.Conv2D(
                    in_channels=self.in_channels[i + self.num_levels],
                    out_channels=num_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    data_format=data_format,
                    bias_attr=ParamAttr(regularizer=L2Decay(0.)))
                conv_aux.skip_quant = True
                yolo_output_aux = self.add_sublayer(name_aux, conv_aux)
                self.yolo_outputs_aux.append(yolo_output_aux)

            if self.use_implicit:
                ia = ImplicitA(self.in_channels[i])
                yolo_output_ia = self.add_sublayer(
                    'yolo_output_ia.{}'.format(i), ia)
                self.ia.append(yolo_output_ia)

                im = ImplicitM(num_filters)
                yolo_output_im = self.add_sublayer(
                    'yolo_output_im.{}'.format(i), im)
                self.im.append(yolo_output_im)

        self._initialize_biases()

    def fuse(self):
        if self.use_implicit:
            # fuse ImplicitA and Convolution
            for i in range(len(self.yolo_outputs)):
                c1, c2, _, _ = self.yolo_outputs[
                    i].weight.shape  # [255, 256, 1, 1]
                c1_, c2_, _, _ = self.ia[i].ia.shape  # [1, 256, 1, 1]
                cc = paddle.matmul(self.yolo_outputs[i].weight.reshape(
                    [c1, c2]), self.ia[i].ia.reshape([c2_, c1_])).squeeze(1)
                self.yolo_outputs[i].bias.set_value(self.yolo_outputs[i].bias +
                                                    cc)

            # fuse ImplicitM and Convolution
            for i in range(len(self.yolo_outputs)):
                c1, c2, _, _ = self.im[i].im.shape  # [1, 255, 1, 1]
                self.yolo_outputs[i].bias.set_value(self.yolo_outputs[i].bias *
                                                    self.im[i].im.reshape([c2]))
                self.yolo_outputs[i].weight.set_value(
                    self.yolo_outputs[i].weight * paddle.transpose(
                        self.im[i].im, [1, 0, 2, 3]))

    def _initialize_biases(self):
        # initialize biases, see https://arxiv.org/abs/1708.02002 section 3.3
        for i, conv in enumerate(self.yolo_outputs):
            b = conv.bias.numpy().reshape([3, -1])  # [255] to [3,85]
            b[:, 4] += math.log(8 / (640 / self.stride[i])**2)
            b[:, 5:self.num_classes + 5] += math.log(0.6 / (self.num_classes - 0.999999))
            conv.bias.set_value(b.reshape([-1]))

        if self.use_aux:
            for i, conv in enumerate(self.yolo_outputs_aux):
                b = conv.bias.numpy().reshape([3, -1])  # [255] to [3,85]
                b[:, 4] += math.log(8 / (640 / self.stride[i])**2)
                b[:, 5:] += math.log(0.6 / (self.num_classes - 0.999999))
                conv.bias.set_value(b.reshape([-1]))

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def parse_anchor(self, anchors, anchor_masks):
        self.anchors = [[anchors[i] for i in mask] for mask in anchor_masks]
        self.mask_anchors = []
        anchor_num = len(anchors)
        for masks in anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def forward(self, feats, targets=None):
        yolo_outputs = []
        if self.training and self.use_aux:
            yolo_outputs_aux = []
        for i in range(self.num_levels):
            if self.training and self.use_implicit:
                yolo_output = self.im[i](self.yolo_outputs[i](self.ia[i](feats[
                    i])))
            else:
                yolo_output = self.yolo_outputs[i](feats[i])
            if self.data_format == 'NHWC':
                yolo_output = paddle.transpose(yolo_output, [0, 3, 1, 2])
            yolo_outputs.append(yolo_output)

            if self.training and self.use_aux:
                yolo_output_aux = self.yolo_outputs_aux[i](feats[
                    i + self.num_levels])
                yolo_outputs_aux.append(yolo_output_aux)

        if self.training:
            if self.use_aux:
                return self.loss(yolo_outputs + yolo_outputs_aux, targets,
                                 self.anchors)
            else:
                return self.loss(yolo_outputs, targets, self.anchors)
        else:
            return yolo_outputs

    def make_grid(self, nx, ny, anchor):
        yv, xv = paddle.meshgrid([
            paddle.arange(
                ny, dtype='int32'), paddle.arange(
                    nx, dtype='int32')
        ])
        grid = paddle.stack((xv, yv), axis=2).reshape([1, 1, ny, nx, 2])
        anchor_grid = anchor.reshape([1, self.num_anchor, 1, 1, 2])
        return grid, anchor_grid

    def postprocessing_by_level(self, head_out, stride, anchor, ny, nx):
        grid, anchor_grid = self.make_grid(nx, ny, anchor)
        out = F.sigmoid(head_out)
        xy = (out[..., 0:2] * 2. - 0.5 + grid) * stride
        wh = (out[..., 2:4] * 2)**2 * anchor_grid
        lt_xy = (xy - wh / 2.)
        rb_xy = (xy + wh / 2.)
        bboxes = paddle.concat((lt_xy, rb_xy), axis=-1)
        scores = out[..., 5:] * out[..., 4].unsqueeze(-1)
        return bboxes, scores

    def post_process(self, head_outs, img_shape, scale_factor):
        bbox_list, score_list = [], []

        for i, head_out in enumerate(head_outs):
            _, _, ny, nx = head_out.shape
            head_out = head_out.reshape(
                [-1, self.num_anchor, self.num_out_ch, ny, nx]).transpose(
                    [0, 1, 3, 4, 2])
            # head_out.shape [bs, self.num_anchor, ny, nx, self.num_out_ch]

            bbox, score = self.postprocessing_by_level(head_out, self.stride[i],
                                                       self.anchors[i], ny, nx)
            bbox = bbox.reshape([-1, self.num_anchor * ny * nx, 4])
            score = score.reshape(
                [-1, self.num_anchor * ny * nx, self.num_classes]).transpose(
                    [0, 2, 1])
            bbox_list.append(bbox)
            score_list.append(score)
        pred_bboxes = paddle.concat(bbox_list, axis=1)
        pred_scores = paddle.concat(score_list, axis=-1)

        if self.exclude_post_process:
            return paddle.concat(
                [pred_bboxes, pred_scores.transpose([0, 2, 1])], axis=-1)
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
class YOLOv7uHead(nn.Layer):
    # YOLOv7 Anchor-Free Head, like YOLOv8Head + use_implicit
    __shared__ = [
        'num_classes', 'eval_size', 'use_implicit', 'trt', 'exclude_nms',
        'exclude_post_process', 'use_shared_conv'
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
                 use_implicit=False,
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 },
                 trt=False,
                 exclude_nms=False,
                 exclude_post_process=False,
                 use_shared_conv=True,
                 print_l1_loss=True):
        super(YOLOv7uHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
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
        self.use_shared_conv = use_shared_conv
        self.use_implicit = use_implicit

        # cls loss
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=paddle.to_tensor([1.0]), reduction="mean")

        # pred head
        c2 = max((16, in_channels[0] // 4, self.reg_channels * 4))
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

        if self.use_implicit:
            self.ia2 = nn.LayerList()
            self.ia3 = nn.LayerList()
            self.im2 = nn.LayerList()
            self.im3 = nn.LayerList()
            for in_c in self.in_channels:
                self.ia2.append(ImplicitA(in_c))
                self.ia3.append(ImplicitA(in_c))
                self.im2.append(ImplicitM(self.reg_channels * 4))
                self.im3.append(ImplicitM(self.num_classes))

        # projection conv
        self.dfl_conv = nn.Conv2D(self.reg_channels, 1, 1, bias_attr=False)
        self.dfl_conv.skip_quant = True
        self.proj = paddle.linspace(0, self.reg_channels - 1, self.reg_channels)
        self.dfl_conv.weight.set_value(
            self.proj.reshape([1, self.reg_channels, 1, 1]))
        self.dfl_conv.weight.stop_gradient = True

        # self._init_bias()

    def fuse(self):
        pass

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_bias(self):
        for a, b, s in zip(self.conv_reg, self.conv_cls, self.fpn_strides):
            a[-1].bias.set_value(1.0)  # box
            b[-1].bias[:self.num_classes] = math.log(5 / self.num_classes /
                                                     (640 / s)**2)

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
            reg_distri = self.im2[i](self.conv_reg[i](self.ia2[i](feat)))
            cls_logit = self.im3[i]((self.conv_cls[i](self.ia3[i](feat))))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))
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
            # reg_dist = self.conv_reg[i](feat)
            # cls_logit = self.conv_cls[i](feat)
            reg_dist = self.im2[i](self.conv_reg[i](self.ia2[i](feat)))
            cls_logit = self.im3[i]((self.conv_cls[i](self.ia3[i](feat))))

            reg_dist = reg_dist.reshape(
                [-1, 4, self.reg_channels, l]).transpose(
                    [0, 2, 3, 1])  # Note diff
            if self.use_shared_conv:
                reg_dist = self.dfl_conv(F.softmax(reg_dist, axis=1)).squeeze(1)
                # [bs, l, 4]
            else:
                reg_dist = F.softmax(reg_dist, axis=1)
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([-1, self.num_classes, l]))
            reg_dist_list.append(reg_dist)

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        if self.use_shared_conv:
            reg_dist_list = paddle.concat(reg_dist_list, axis=1)
        else:
            reg_dist_list = paddle.concat(reg_dist_list, axis=2)
            reg_dist_list = self.dfl_conv(reg_dist_list).squeeze(1)

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
        _, l, _ = get_static_shape(pred_dist)
        pred_dist = F.softmax(pred_dist.reshape([-1, l, 4, self.reg_channels]))
        pred_dist = self.dfl_conv(pred_dist.transpose([0, 3, 1, 2])).squeeze(1)
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = paddle.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return paddle.concat([lt, rb], -1).clip(self.reg_range[0],
                                                self.reg_range[1] - 1 - 0.01)

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

    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)

            # loss_l1 just see if train well
            if self.print_l1_loss:
                loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)
            else:
                loss_l1 = paddle.zeros([1])

            # ciou loss
            iou = bbox_iou(
                pred_bboxes_pos, assigned_bboxes_pos, x1y1x2y2=False, ciou=True)
            loss_iou = ((1.0 - iou) * bbox_weight).sum() / assigned_scores_sum

            dist_mask = mask_positive.unsqueeze(-1).tile(
                [1, 1, self.reg_channels * 4])
            pred_dist_pos = paddle.masked_select(
                pred_dist, dist_mask).reshape([-1, 4, self.reg_channels])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes)
            assigned_ltrb_pos = paddle.masked_select(
                assigned_ltrb, bbox_mask).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos, assigned_ltrb_pos,
                                     self.reg_range[0]) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
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
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels,
                                      self.num_classes + 1)[..., :-1]
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self.bce(pred_scores, assigned_scores)

        assigned_scores_sum = assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum /= paddle.distributed.get_world_size()
        assigned_scores_sum = paddle.clip(assigned_scores_sum, min=1.)
        # loss_cls /= assigned_scores_sum

        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl
        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
        }
        if self.print_l1_loss:
            # just see convergence
            out_dict.update({'loss_l1': loss_l1})
        return out_dict

    def post_process(self, head_outs, im_shape, scale_factor):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist)
        pred_bboxes *= stride_tensor

        if self.exclude_post_process:
            return paddle.concat(
                [pred_bboxes, pred_scores.transpose([0, 2, 1])], axis=-1)
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
