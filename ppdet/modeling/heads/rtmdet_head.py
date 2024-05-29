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

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register

from ..bbox_utils import batch_distance2bbox, custom_ceil
from ..losses import GIoULoss, QualityFocalLoss, IouLoss
from ..initializer import bias_init_with_prob, constant_
from ppdet.modeling.backbones.csp_darknet import BaseConv
from ppdet.modeling.assigners.simota_assigner import SimOTAAssigner  #, DynamicSoftLabelAssigner
from ppdet.modeling.layers import MultiClassNMS
from paddle import ParamAttr
from paddle.nn.initializer import Normal

__all__ = ['RTMDetHead', 'RTMDetInsHead']


@register
class RTMDetHead(nn.Layer):
    __shared__ = [
        'num_classes', 'width_mult', 'trt', 'exclude_nms',
        'exclude_post_process'
    ]
    __inject__ = ['assigner', 'nms']

    def __init__(
            self,
            num_classes=80,
            width_mult=1.0,
            in_channels=[1024, 512, 256],
            feat_channels=256,
            stacked_convs=2,
            pred_kernel_size=1,
            act='swish',
            fpn_strides=(32, 16, 8),
            share_conv=True,
            exp_on_reg=False,
            assigner='SimOTAAssigner',  # just placeholder
            grid_cell_offset=0.,
            nms='MultiClassNMS',
            loss_weight={
                'cls': 1.0,
                'box': 2.0,
            },
            trt=False,
            exclude_nms=False,
            exclude_post_process=False):
        super(RTMDetHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self._dtype = paddle.framework.get_default_dtype()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.pred_kernel_size = pred_kernel_size
        self.stacked_convs = stacked_convs
        self.feat_channels = int(feat_channels * width_mult)
        self.share_conv = share_conv
        self.exp_on_reg = exp_on_reg
        self.grid_cell_offset = grid_cell_offset

        self.loss_cls = QualityFocalLoss()
        self.loss_box = IouLoss(loss_weight=1.0, giou=True)
        self.loss_weight = loss_weight
        self.assigner = assigner

        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process

        # head
        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()
        self.cls_preds = nn.LayerList()
        self.reg_preds = nn.LayerList()
        for idx in range(len(self.fpn_strides)):
            cls_convs = nn.LayerList()
            reg_convs = nn.LayerList()
            for i in range(self.stacked_convs):
                chn = self.in_channels[idx] if i == 0 else self.feat_channels
                cls_convs.append(BaseConv(chn, self.feat_channels, 3, 1))
                reg_convs.append(BaseConv(chn, self.feat_channels, 3, 1))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

            self.cls_preds.append(
                nn.Conv2D(
                    self.feat_channels,
                    self.num_classes,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2,
                    weight_attr=ParamAttr(initializer=Normal(
                        mean=0., std=0.01)),
                    bias_attr=True))
            self.reg_preds.append(
                nn.Conv2D(
                    self.feat_channels,
                    4,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2,
                    weight_attr=ParamAttr(initializer=Normal(
                        mean=0., std=0.01)),
                    bias_attr=True))

        self.share_conv = False  # TODO in deploy
        if self.share_conv:
            for n in range(len(self.fpn_strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv
        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.cls_preds, self.reg_preds):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, 1.0)

    def forward_train(self, feats, targets):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"
        feat_sizes = [[f.shape[-2], f.shape[-1]] for f in feats]

        cls_score_list, reg_distri_list = [], []
        for idx, x in enumerate(feats):
            _, _, h, w = x.shape
            cls_feat = x
            reg_feat = x
            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_logit = self.cls_preds[idx](cls_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)
            if self.exp_on_reg:
                reg_dist = self.reg_preds[idx](reg_feat).exp()
            else:
                reg_dist = self.reg_preds[idx](reg_feat)
            reg_dist = reg_dist * self.fpn_strides[idx]
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_dist.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)

        anchor_points, stride_tensor = self._generate_anchor_point(
            feat_sizes, self.fpn_strides, 0.)

        raise NotImplementedError('RTMDet training not implemented yet.')

        return self.get_loss(
            [cls_score_list, reg_distri_list, anchor_points,
             stride_tensor], targets)

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

    def forward_eval(self, feats):
        anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        for idx, x in enumerate(feats):
            _, _, h, w = x.shape
            l = h * w
            cls_feat = x
            reg_feat = x
            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_logit = self.cls_preds[idx](cls_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)
            if self.exp_on_reg:
                reg_dist = self.reg_preds[idx](reg_feat).exp()
            else:
                reg_dist = self.reg_preds[idx](reg_feat)
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([-1, self.num_classes, l]))
            reg_dist_list.append(reg_dist.reshape([-1, 4, l]))

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        reg_dist_list = paddle.concat(reg_dist_list, axis=-1)

        return cls_score_list, reg_dist_list, anchor_points, stride_tensor

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        if self.training:
            return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)

    def _generate_anchor_point(self, feat_sizes, strides, offset=0.):
        anchor_points, stride_tensor = [], []
        num_anchors_list = []
        for feat_size, stride in zip(feat_sizes, strides):
            h, w = feat_size
            x = (paddle.arange(w) + offset) * stride
            y = (paddle.arange(h) + offset) * stride
            y, x = paddle.meshgrid(y, x)
            anchor_points.append(paddle.stack([x, y], axis=-1).reshape([-1, 2]))
            stride_tensor.append(
                paddle.full(
                    [len(anchor_points[-1]), 1], stride, dtype=self._dtype))
            num_anchors_list.append(len(anchor_points[-1]))
        anchor_points = paddle.concat(anchor_points).astype(self._dtype)
        anchor_points.stop_gradient = True
        stride_tensor = paddle.concat(stride_tensor)
        stride_tensor.stop_gradient = True
        return anchor_points, stride_tensor  #, num_anchors_list

    def get_loss(self, head_outs, targets):
        pred_cls, pred_bboxes, anchor_points, stride_tensor = head_outs
        raise NotImplementedError('RTMDet training not implemented yet.')

        gt_labels = targets['gt_class']
        gt_bboxes = targets['gt_bbox']

        loss_cls = paddle.zeros([1])
        loss_iou = paddle.zeros([1])
        loss = self.loss_weight['cls'] * loss_cls + \
               self.loss_weight['box'] * loss_iou
        return {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_box': loss_iou,
        }

    def post_process(self, head_outs, im_shape, scale_factor):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points,
                                          pred_dist.transpose([0, 2, 1]))
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


class MaskFeatModule(nn.Layer):
    def __init__(self,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 num_levels=3,
                 num_prototypes=8,
                 act='silu'):
        super().__init__()
        self.num_levels = num_levels
        self.fusion_conv = nn.Conv2D(num_levels * in_channels, in_channels, 1)
        convs = []
        for i in range(stacked_convs):
            in_c = in_channels if i == 0 else feat_channels
            convs.append(BaseConv(in_c, feat_channels, 3, 1, act=act))
        self.stacked_convs = nn.Sequential(*convs)
        self.projection = nn.Conv2D(
            feat_channels, num_prototypes, kernel_size=1)

    def forward(self, features):
        fusion_feats = [features[0]]
        size = features[0].shape[-2:]
        for i in range(1, self.num_levels):
            f = F.interpolate(features[i], size=size, mode='bilinear')
            fusion_feats.append(f)
        fusion_feats = paddle.concat(fusion_feats, axis=1)
        fusion_feats = self.fusion_conv(fusion_feats)
        # pred mask feats
        mask_features = self.stacked_convs(fusion_feats)
        mask_features = self.projection(mask_features)
        return mask_features


@register
class RTMDetInsHead(nn.Layer):
    __shared__ = [
        'num_classes', 'width_mult', 'eval_size', 'act', 'trt', 'exclude_nms',
        'exclude_post_process', 'with_mask'
    ]
    __inject__ = ['assigner', 'nms']

    def __init__(
            self,
            num_classes=80,
            width_mult=1.0,
            in_channels=[256, 256, 256],
            feat_channels=256,
            stacked_convs=2,
            pred_kernel_size=1,
            act='silu',
            fpn_strides=(32, 16, 8),
            eval_size=[640, 640],
            share_conv=True,
            exp_on_reg=False,
            assigner='SimOTAAssigner',  # just placeholder
            grid_cell_offset=0.,
            nms='MultiClassNMS',
            loss_weight={
                'cls': 1.0,
                'box': 2.0,
            },
            with_mask=True,
            num_prototypes=8,
            dyconv_channels=8,
            num_dyconvs=3,
            mask_loss_stride=4,
            loss_mask=dict(
                type='DiceLoss', loss_weight=2.0, eps=5e-6, reduction='mean'),
            mask_thr_binary=0.5,
            trt=False,
            exclude_nms=False,
            exclude_post_process=False):
        super(RTMDetInsHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self._dtype = paddle.framework.get_default_dtype()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.eval_size = eval_size
        self.pred_kernel_size = pred_kernel_size
        self.stacked_convs = stacked_convs
        self.feat_channels = int(feat_channels * width_mult)
        self.share_conv = share_conv
        self.exp_on_reg = exp_on_reg
        self.grid_cell_offset = grid_cell_offset

        self.loss_cls = QualityFocalLoss()
        self.loss_box = IouLoss(loss_weight=1.0, giou=True)
        self.loss_weight = loss_weight
        self.assigner = assigner

        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process

        # mask added
        self.with_mask = with_mask
        self.num_prototypes = num_prototypes
        self.num_dyconvs = num_dyconvs
        self.dyconv_channels = dyconv_channels
        self.mask_loss_stride = mask_loss_stride
        self.mask_thr_binary = mask_thr_binary
        weight_nums, bias_nums = [], []
        for i in range(self.num_dyconvs):
            if i == 0:
                weight_nums.append(
                    # mask prototype and coordinate features
                    (self.num_prototypes + 2) * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels * 1)
            elif i == self.num_dyconvs - 1:
                weight_nums.append(self.dyconv_channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dyconv_channels * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels * 1)
        self.weight_nums = weight_nums  # [80, 8, 64]
        self.bias_nums = bias_nums  # [8, 1, 8]
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        # head
        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()
        self.ker_convs = nn.LayerList()
        self.cls_preds = nn.LayerList()
        self.reg_preds = nn.LayerList()
        self.ker_preds = nn.LayerList()
        for idx in range(len(self.fpn_strides)):
            cls_convs = nn.LayerList()
            reg_convs = nn.LayerList()
            ker_convs = nn.LayerList()
            for i in range(self.stacked_convs):
                chn = self.in_channels[idx] if i == 0 else self.feat_channels
                cls_convs.append(
                    BaseConv(
                        chn, self.feat_channels, 3, 1, act=act))
                reg_convs.append(
                    BaseConv(
                        chn, self.feat_channels, 3, 1, act=act))
                ker_convs.append(
                    BaseConv(
                        chn, self.feat_channels, 3, 1, act=act))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
            self.ker_convs.append(ker_convs)

            self.cls_preds.append(
                nn.Conv2D(
                    self.feat_channels,
                    self.num_classes,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2,
                    weight_attr=ParamAttr(initializer=Normal(
                        mean=0., std=0.01)),
                    bias_attr=True))
            self.reg_preds.append(
                nn.Conv2D(
                    self.feat_channels,
                    4,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2,
                    weight_attr=ParamAttr(initializer=Normal(
                        mean=0., std=0.01)),
                    bias_attr=True))
            self.ker_preds.append(
                nn.Conv2D(
                    self.feat_channels,
                    self.num_gen_params,  # 169
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2,
                    weight_attr=ParamAttr(initializer=Normal(
                        mean=0., std=0.01)),
                    bias_attr=True))

        self.mask_head = MaskFeatModule(
            in_channels=self.in_channels[0],
            feat_channels=self.feat_channels,
            stacked_convs=4,
            num_levels=len(fpn_strides),
            num_prototypes=self.num_prototypes,
            act=act)

        self.share_conv = False  # TODO: export and deploy
        if self.share_conv:
            for n in range(len(self.fpn_strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv
        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_, ker_ in zip(self.cls_preds, self.reg_preds,
                                    self.ker_preds):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, 1.0)
            constant_(ker_.weight)
            constant_(ker_.bias, 1.0)

    def forward_train(self, feats, targets):
        mask_feat = self.mask_head(feats)

        feat_sizes = [[f.shape[-2], f.shape[-1]] for f in feats]
        cls_score_list, reg_distri_list, ker_pred_list = [], [], []
        for idx, x in enumerate(feats):
            _, _, h, w = x.shape
            cls_feat = x
            reg_feat = x
            ker_feat = x
            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_logit = self.cls_preds[idx](cls_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)
            reg_dist = self.reg_preds[idx](reg_feat)
            reg_dist = reg_dist * self.fpn_strides[idx]

            for ker_layer in self.ker_convs[idx]:
                ker_feat = ker_layer(ker_feat)
            ker_pred = self.ker_preds[idx](ker_feat)

            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_dist.flatten(2).transpose([0, 2, 1]))
            ker_pred_list.append(ker_pred.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)
        ker_pred_list = paddle.concat(ker_pred_list, axis=1)

        anchor_points, stride_tensor = self._generate_anchor_point(
            feat_sizes, self.fpn_strides, 0.)

        raise NotImplementedError('RTMDet training not implemented yet.')

        return self.get_loss([
            cls_score_list, reg_distri_list, ker_pred_list, mask_feat,
            anchor_points, stride_tensor
        ], targets)

    def _generate_anchors(self, fpn_strides, feats=None, dtype='float32'):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(fpn_strides):
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

    def forward_eval(self, feats):
        mask_feat = self.mask_head(feats)

        anchor_points, stride_tensor = self._generate_anchors(self.fpn_strides,
                                                              feats)
        cls_score_list, reg_dist_list, ker_pred_list = [], [], []
        for idx, x in enumerate(feats):
            _, _, h, w = x.shape
            l = h * w
            cls_feat = x
            reg_feat = x
            ker_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_logit = self.cls_preds[idx](cls_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)
            reg_dist = F.relu(self.reg_preds[idx](reg_feat))

            for ker_layer in self.ker_convs[idx]:
                ker_feat = ker_layer(ker_feat)
            ker_pred = self.ker_preds[idx](ker_feat)

            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([-1, self.num_classes, l]))
            reg_dist_list.append(reg_dist.reshape([-1, 4, l]))
            ker_pred_list.append(ker_pred.reshape([-1, self.num_gen_params, l]))

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        reg_dist_list = paddle.concat(reg_dist_list, axis=-1)
        ker_pred_list = paddle.concat(ker_pred_list, axis=-1)

        return cls_score_list, reg_dist_list, ker_pred_list, mask_feat, anchor_points, stride_tensor

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        if self.training:
            return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)

    def _generate_anchor_point(self, feat_sizes, strides, offset=0.):
        anchor_points, stride_tensor = [], []
        num_anchors_list = []
        for feat_size, stride in zip(feat_sizes, strides):
            h, w = feat_size
            x = (paddle.arange(w) + offset) * stride
            y = (paddle.arange(h) + offset) * stride
            y, x = paddle.meshgrid(y, x)
            anchor_points.append(paddle.stack([x, y], axis=-1).reshape([-1, 2]))
            stride_tensor.append(
                paddle.full(
                    [len(anchor_points[-1]), 1], stride, dtype=self._dtype))
            num_anchors_list.append(len(anchor_points[-1]))
        anchor_points = paddle.concat(anchor_points).astype(self._dtype)
        anchor_points.stop_gradient = True
        stride_tensor = paddle.concat(stride_tensor)
        stride_tensor.stop_gradient = True
        return anchor_points, stride_tensor

    def get_loss(self, head_outs, targets):
        pred_cls, pred_bboxes, pred_kernels, mask_feat, anchor_points, stride_tensor = head_outs
        raise NotImplementedError('RTMDet-Ins training not implemented yet.')

        gt_labels = targets['gt_class']
        gt_bboxes = targets['gt_bbox']

        loss_cls = paddle.zeros([1])
        loss_iou = paddle.zeros([1])
        loss = self.loss_weight['cls'] * loss_cls + \
               self.loss_weight['box'] * loss_iou
        return {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_box': loss_iou,
        }

    def post_process(self, head_outs, im_shape, scale_factor, rescale=True):
        assert not self.exclude_post_process or not self.exclude_nms

        pred_scores, pred_dist, ker_preds, mask_feats, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points,
                                          pred_dist.transpose([0, 2, 1]))
        pred_bboxes *= stride_tensor
        # scale bbox to origin
        scale_factor_bbox = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor_bbox

        bbox_pred, bbox_num, keep_idxs = self.nms(pred_bboxes, pred_scores)

        if self.with_mask and bbox_num.sum() > 0:
            ker_preds = ker_preds.transpose([0, 2, 1])  ### Note
            ker_preds_keep = paddle.gather(
                ker_preds.reshape([-1, self.num_gen_params]), keep_idxs)

            anchor_points_keep = paddle.gather(anchor_points, keep_idxs)
            stride_tensor_keep = paddle.gather(stride_tensor, keep_idxs)
            mask_logits = self.mask_post_process(mask_feats, ker_preds_keep,
                                                 anchor_points_keep,
                                                 stride_tensor_keep)

            mask_logits = F.interpolate(
                mask_logits.unsqueeze(0), scale_factor=8, mode='bilinear')
            if rescale:
                ori_h, ori_w = im_shape[0] / scale_factor[0]
                mask_logits = F.interpolate(
                    mask_logits,
                    size=[
                        custom_ceil(mask_logits.shape[-2] / scale_factor[0][0]),
                        custom_ceil(mask_logits.shape[-1] / scale_factor[0][1])
                    ],
                    mode='bilinear',
                    align_corners=False)[..., :int(ori_h), :int(ori_w)]
            masks = F.sigmoid(mask_logits).squeeze(0)
            mask_pred = masks > self.mask_thr_binary
        else:
            ori_h, ori_w = im_shape[0] / scale_factor[0]
            mask_pred = paddle.zeros([bbox_num, int(ori_h), int(ori_w)])

        if self.with_mask:
            return bbox_pred, bbox_num, mask_pred
        else:
            return bbox_pred, bbox_num

    def mask_post_process(self, mask_feat, kernels, anchor_points,
                          stride_tensor):
        _, c, h, w = mask_feat.shape  # [1, 8, 80, 80]
        num_inst = 100  # kernels.shape[0] # [100, 169]
        coord, coord_stride = self._generate_anchors(fpn_strides=[8])
        coord = (coord * coord_stride).reshape([1, -1, 2])

        anchor_points = (anchor_points * stride_tensor).reshape([-1, 1, 2])
        stride_tensor = stride_tensor.reshape([-1, 1, 1])
        relative_coord = (anchor_points - coord).transpose([0, 2, 1]) / (
            stride_tensor * 8)
        relative_coord = relative_coord.reshape([num_inst, 2, h, w])
        mask_feat = paddle.concat(
            [relative_coord, mask_feat.tile([num_inst, 1, 1, 1])], axis=1)
        weights, biases = self.parse_dynamic_params(kernels, num_inst)

        n_layers = len(weights)
        x = mask_feat.reshape([-1, num_inst * (c + 2), h, w])
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, weight, bias=bias, stride=1, padding=0, groups=num_inst)
            if i < n_layers - 1:
                x = F.relu(x)
        x = x.reshape([num_inst, h, w])
        return x

    def parse_dynamic_params(self, flatten_kernels, n_inst):
        """split kernel head prediction to conv weight and bias."""
        n_layers = len(self.weight_nums)
        params_splits = list(
            paddle.split(
                flatten_kernels, self.weight_nums + self.bias_nums, axis=1))
        weight_splits = params_splits[:n_layers]  # [100, 80] [100, 64] [100, 8]
        bias_splits = params_splits[n_layers:]  # [100, 8] [100, 8] [100, 1]
        for i in range(n_layers):
            w_dim = weight_splits[i].shape[1]
            if i < n_layers - 1:
                weight_splits[i] = weight_splits[i].reshape([
                    n_inst * self.dyconv_channels,
                    int(w_dim / self.dyconv_channels), 1, 1
                ])
                bias_splits[i] = bias_splits[i].reshape(
                    [n_inst * self.dyconv_channels])
            else:
                weight_splits[i] = weight_splits[i].reshape(
                    [n_inst, w_dim, 1, 1])
                bias_splits[i] = bias_splits[i].reshape([n_inst])

        return weight_splits, bias_splits
