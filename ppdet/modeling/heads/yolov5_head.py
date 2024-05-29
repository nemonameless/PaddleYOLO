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
from ppdet.modeling.backbones.csp_darknet import BaseConv
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register
from ppdet.modeling.layers import MultiClassNMS
from ..bbox_utils import custom_ceil

__all__ = ['YOLOv5Head', 'YOLOv5InsHead']


@register
class YOLOv5Head(nn.Layer):
    __shared__ = [
        'num_classes', 'data_format', 'trt', 'exclude_nms',
        'exclude_post_process'
    ]
    __inject__ = ['loss', 'nms']

    def __init__(self,
                 num_classes=80,
                 in_channels=[256, 512, 1024],
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                 stride=[8, 16, 32],
                 loss='YOLOv5Loss',
                 data_format='NCHW',
                 nms='MultiClassNMS',
                 trt=False,
                 exclude_post_process=False,
                 exclude_nms=False):
        """
        Head for YOLOv5

        Args:
            num_classes (int): number of foreground classes
            in_channels (int): channels of input features
            anchors (list): anchors
            anchor_masks (list): anchor masks
            stride (list): strides
            loss (object): YOLOv5Loss instance
            data_format (str): nms format, NCHW or NHWC
            loss (object): YOLOv5Loss instance
        """
        super(YOLOv5Head, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.parse_anchor(anchors, anchor_masks)
        self.anchors = paddle.to_tensor(self.anchors, dtype='float32')
        self.anchor_levels = len(self.anchors)

        self.stride = stride
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
        for i in range(len(self.anchors)):
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

        self._initialize_biases()

    def _initialize_biases(self):
        # initialize biases into Detect()
        # https://arxiv.org/abs/1708.02002 section 3.3
        for i, conv in enumerate(self.yolo_outputs):
            b = conv.bias.numpy().reshape([3, -1])
            b[:, 4] += math.log(8 / (640 / self.stride[i])**2)
            b[:, 5:self.num_classes + 5] += math.log(0.6 / (
                self.num_classes - 0.999999))
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
        assert len(feats) == len(self.anchors)
        yolo_outputs = []
        for i, feat in enumerate(feats):
            yolo_output = self.yolo_outputs[i](feat)
            if self.data_format == 'NHWC':
                yolo_output = paddle.transpose(yolo_output, [0, 3, 1, 2])
            yolo_outputs.append(yolo_output)

        if self.training:
            return self.loss(yolo_outputs, targets, self.anchors)
        else:
            return yolo_outputs

    def make_grid(self, nx, ny, anchor):
        yv, xv = paddle.meshgrid([
            paddle.arange(
                ny, dtype='float32'), paddle.arange(
                    nx, dtype='float32')
        ])

        grid = paddle.stack(
            (xv, yv), axis=2).expand([1, self.num_anchor, ny, nx, 2])
        anchor_grid = anchor.reshape([1, self.num_anchor, 1, 1, 2]).expand(
            (1, self.num_anchor, ny, nx, 2))
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

    def post_process(self,
                     head_outs,
                     im_shape,
                     scale_factor,
                     infer_shape=[640, 640]):
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


class MaskProto(nn.Layer):
    # YOLOv5 mask Proto module for instance segmentation models
    def __init__(self, ch_in, num_protos=256, num_masks=32, act='silu'):
        super().__init__()
        self.conv1 = BaseConv(ch_in, num_protos, 3, 1, act=act)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = BaseConv(num_protos, num_protos, 3, 1, act=act)
        self.conv3 = BaseConv(num_protos, num_masks, 1, 1, act=act)

    def forward(self, x):
        return self.conv3(self.conv2(self.upsample(self.conv1(x))))


@register
class YOLOv5InsHead(nn.Layer):
    __shared__ = [
        'num_classes', 'width_mult', 'act', 'trt', 'exclude_nms',
        'exclude_post_process', 'with_mask'
    ]
    __inject__ = ['loss', 'nms']

    def __init__(self,
                 with_mask=True,
                 num_classes=80,
                 num_masks=32,
                 num_protos=256,
                 width_mult=1.0,
                 in_channels=[256, 512, 1024],
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                 act='silu',
                 stride=[8, 16, 32],
                 loss='YOLOv5InsLoss',
                 nms='MultiClassNMS',
                 mask_thr_binary=0.5,
                 trt=False,
                 exclude_post_process=False,
                 exclude_nms=False):
        """
        Head for YOLOv5 Ins

        Args:
            num_classes (int): number of foreground classes
            in_channels (int): channels of input features
            anchors (list): anchors
            anchor_masks (list): anchor masks
            stride (list): strides
            loss (object): YOLOv5Loss instance
            loss (object): YOLOv5Loss instance
        """
        super(YOLOv5InsHead, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.with_mask = with_mask
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_masks = num_masks
        self.num_protos = int(num_protos * width_mult)

        self.parse_anchor(anchors, anchor_masks)
        self.anchors = paddle.to_tensor(self.anchors, dtype='float32')
        self.anchor_levels = len(self.anchors)

        self.stride = stride
        self.loss = loss
        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process

        self.num_anchor = len(self.anchors[0])  # self.na
        self.num_out_ch = self.num_classes + 5 + self.num_masks  # self.no

        self.yolo_outputs = []
        for i in range(len(self.anchors)):
            num_filters = self.num_anchor * self.num_out_ch
            name = 'yolo_output.{}'.format(i)
            conv = nn.Conv2D(
                in_channels=self.in_channels[i],
                out_channels=num_filters,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=ParamAttr(regularizer=L2Decay(0.)))
            conv.skip_quant = True
            yolo_output = self.add_sublayer(name, conv)
            self.yolo_outputs.append(yolo_output)

        self.mask_thr_binary = mask_thr_binary
        self.proto = MaskProto(
            in_channels[0], self.num_protos, self.num_masks, act=act)

        self._initialize_biases()

    def _initialize_biases(self):
        # initialize biases into Detect()
        # https://arxiv.org/abs/1708.02002 section 3.3
        for i, conv in enumerate(self.yolo_outputs):
            b = conv.bias.numpy().reshape([3, -1])
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
        mask_feats = self.proto(feats[0])

        yolo_outputs = []
        for i, feat in enumerate(feats):
            yolo_output = self.yolo_outputs[i](feat)
            yolo_outputs.append(yolo_output)

        yolo_outputs.append(mask_feats)

        if self.training:
            return self.loss(yolo_outputs, targets, self.anchors)
        else:
            return yolo_outputs

    def make_grid(self, nx, ny, anchor):
        yv, xv = paddle.meshgrid([
            paddle.arange(
                ny, dtype='float32'), paddle.arange(
                    nx, dtype='float32')
        ])

        grid = paddle.stack(
            (xv, yv), axis=2).expand([1, self.num_anchor, ny, nx, 2])
        anchor_grid = anchor.reshape([1, self.num_anchor, 1, 1, 2]).expand(
            (1, self.num_anchor, ny, nx, 2))
        return grid, anchor_grid

    def postprocessing_by_level(self, head_out, stride, anchor, ny, nx):
        grid, anchor_grid = self.make_grid(nx, ny, anchor)
        out = F.sigmoid(head_out[..., :-self.num_masks])
        xy = (out[..., 0:2] * 2. - 0.5 + grid) * stride
        wh = (out[..., 2:4] * 2)**2 * anchor_grid
        lt_xy = (xy - wh / 2.)
        rb_xy = (xy + wh / 2.)
        bboxes = paddle.concat((lt_xy, rb_xy), axis=-1)
        scores = out[..., 5:self.num_classes + 5] * out[..., 4].unsqueeze(-1)
        masks = head_out[..., -self.num_masks:]
        return bboxes, scores, masks

    def post_process(self,
                     head_outs,
                     im_shape,
                     scale_factor,
                     infer_shape=[640, 640],
                     rescale=True):
        assert not self.exclude_post_process or not self.exclude_nms

        mask_feats = head_outs[-1]

        bbox_list, score_list, mask_list = [], [], []
        for i, head_out in enumerate(head_outs[:-1]):
            _, _, ny, nx = head_out.shape
            head_out = head_out.reshape(
                [-1, self.num_anchor, self.num_out_ch, ny, nx]).transpose(
                    [0, 1, 3, 4, 2])
            # head_out.shape [bs, self.num_anchor, ny, nx, self.num_out_ch]

            bbox, score, mask = self.postprocessing_by_level(
                head_out, self.stride[i], self.anchors[i], ny, nx)
            bbox = bbox.reshape([-1, self.num_anchor * ny * nx, 4])
            score = score.reshape(
                [-1, self.num_anchor * ny * nx, self.num_classes]).transpose(
                    [0, 2, 1])
            mask = mask.reshape([-1, self.num_anchor * ny * nx, self.num_masks])

            bbox_list.append(bbox)
            score_list.append(score)
            mask_list.append(mask)

        pred_bboxes = paddle.concat(bbox_list, axis=1)
        pred_scores = paddle.concat(score_list, axis=-1)
        pred_masks = paddle.concat(mask_list, axis=1)

        bbox_pred, bbox_num, keep_idxs = self.nms(pred_bboxes, pred_scores)

        if self.with_mask and bbox_num.sum() > 0:
            mask_coeffs = paddle.gather(
                pred_masks.reshape([-1, self.num_masks]), keep_idxs)

            mask_logits = process_mask(
                mask_feats[0],
                mask_coeffs,
                bbox_pred[:, 2:6],
                shape=infer_shape,
                upsample=True)  # HWC
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
        else:
            ori_h, ori_w = im_shape[0] / scale_factor[0]
            mask_pred = paddle.zeros([bbox_num, int(ori_h), int(ori_w)])

        # scale bbox to origin
        scale_factor = scale_factor.flip(-1).tile([1, 2])
        bbox_pred[:, 2:6] /= scale_factor

        if self.with_mask:
            return bbox_pred, bbox_num, mask_pred
        else:
            return bbox_pred, bbox_num


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
    # x1 shape(n,1,1) # [146, 428, 640]
    r = paddle.arange(w, dtype=x1.dtype)[None, None, :]
    # rows shape(1,w,1) # [1, 1, 640]
    c = paddle.arange(h, dtype=y1.dtype)[None, :, None]
    # cols shape(h,1,1) # [1, 428, 1]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(protos, masks_in, bboxes, shape, upsample=True):
    """
    It takes the output of the mask head, and applies the mask to the bounding boxes. This is faster but produces
    downsampled quality of mask

    Args:
      protos (paddle.Tensor): [mask_dim, mask_h, mask_w]
      masks_in (paddle.Tensor): [n, mask_dim], n is number of masks after nms
      bboxes (paddle.Tensor): [n, 4], n is number of masks after nms
      shape (paddle): the size of the input image (h,w)

    Returns:
      (paddle.Tensor): The processed masks.
    """

    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = F.sigmoid(masks_in @protos.reshape([c, -1])).reshape([-1, mh, mw])

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(
            masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    return masks
