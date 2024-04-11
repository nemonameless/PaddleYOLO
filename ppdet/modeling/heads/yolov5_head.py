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
from ppdet.modeling.layers import MultiClassNMS
import numpy as np
from ..losses import IouLoss

__all__ = ['YOLOv5Head', 'YOLOv5Head_rect']


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

    def post_process(self, head_outs, im_shape, scale_factor):
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
class YOLOv5Head_rect(nn.Layer):
    # ONLY uesd for rect eval
    __shared__ = ['num_classes', 'trt', 'exclude_nms', 'exclude_post_process']
    __inject__ = ['loss', 'nms']

    def __init__(self,
                 num_classes=80,
                 in_channels=[256, 512, 1024],
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                 stride=[8, 16, 32],
                 loss='YOLOv5Loss',
                 nms='MultiClassNMS',
                 multi_label=True,
                 trt=False,
                 exclude_post_process=False,
                 exclude_nms=False):
        super(YOLOv5Head_rect, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.num_classes = num_classes
        self.in_channels = in_channels

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
        if self.exclude_nms or self.exclude_post_process:
            multi_label = False
        self.multi_label = multi_label and self.num_classes > 1
        self.categories = [i for i in range(self.num_classes)]

        self.num_anchor = len(self.anchors[0])  # self.na
        self.num_levels = len(self.anchors[1])
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
                bias_attr=ParamAttr(regularizer=L2Decay(0.)))
            conv.skip_quant = True
            yolo_output = self.add_sublayer(name, conv)
            self.yolo_outputs.append(yolo_output)

        self.loss_cls = nn.BCEWithLogitsLoss(
            pos_weight=paddle.to_tensor([1.0]), reduction="mean")
        self.loss_obj = nn.BCEWithLogitsLoss(
            pos_weight=paddle.to_tensor([1.0]), reduction="mean")
        self.loss_bbox = IouLoss() #### bug code

        box_weight = 0.05
        obj_weight = 1.0
        cls_weght = 0.5
        self.loss_weights = {
            'box': box_weight,
            'obj': obj_weight,
            'cls': cls_weght,
        }
        self.obj_level_weights = [4.0, 1.0, 0.4]  # balance
        self.prior_match_thr = 4.0  # anchor_t
        self.near_neighbor_thr = 0.5  # bias
        self.grid_offset = paddle.to_tensor([
            [0, 0],  # center
            [1, 0],  # left
            [0, 1],  # up
            [-1, 0],  # right
            [0, -1],  # bottom
        ])
        self._initialize_biases()

    def _initialize_biases(self):
        # initialize biases into Detect()
        # https://arxiv.org/abs/1708.02002 section 3.3
        for i, conv in enumerate(self.yolo_outputs):
            b = conv.bias.numpy().reshape([self.num_anchor, -1])
            b[:, 4] += math.log(8 / (640 / self.stride[i])**2)
            b[:, 5:5 + self.num_classes] += math.log(0.6 / (
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
        cls_scores, bbox_preds, objectnesses = [], [], []
        featmap_sizes = []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            featmap_sizes.append([h, w])
            yolo_output = self.yolo_outputs[i](feat) # [b, 3 * (80+5), h, w]
            yolo_output = yolo_output.reshape([-1, self.num_anchor, self.num_out_ch, h, w]) # [b, 3, 85, h, w]
            # cls_scores.append(yolo_output[:, :, 5:, ...].reshape([-1, self.num_classes, h, w]))
            # bbox_preds.append(yolo_output[:, :, :4, ...].reshape([-1, 4, h, w]))
            # objectnesses.append(yolo_output[:, :, 4:5, ...].reshape([-1, 1, h, w]))

            cls_scores.append(yolo_output[:, :, 5:, ...].reshape([-1, self.num_anchor * self.num_classes, h, w])) # [1, 240, 56, 84]
            bbox_preds.append(yolo_output[:, :, :4, ...].reshape([-1, self.num_anchor * 4, h, w])) # [1, 12, 56, 84]
            objectnesses.append(yolo_output[:, :, 4:5, ...].reshape([-1, self.num_anchor * 1, h, w])) # [1, 3, 56, 84]

            #cls_scores.append(yolo_output[:, :, 5:, ...].transpose([0, 1, 3, 4, 2]).reshape([-1, self.num_anchor * h * w, self.num_classes]))
            # bbox_preds.append(yolo_output[:, :, :4, ...].transpose([0, 1, 3, 4, 2]).reshape([-1, self.num_anchor * h * w, 4]))
            # objectnesses.append(yolo_output[:, :, 4:5, ...].transpose([0, 1, 3, 4, 2]).reshape([-1, self.num_anchor * h * w, 1]))

        # for x in bbox_pred:
        #     print(x.shape, x.sum())
        if self.training:
            return self.get_loss([cls_scores, bbox_preds, objectnesses], targets, self.anchors)
        else:
            # flatten_cls_scores = [
            #     cls_score.transpose([0, 2, 3, 1]).reshape([-1, hw[0] * hw[1] * self.num_anchor, self.num_classes]) # [1, 3*6174, 80]
            #     for cls_score, hw in zip(cls_scores, featmap_sizes)
            # ]
            bs = feats[0].shape[0]
            flatten_cls_scores = [
                cls_score.transpose([0, 2, 3, 1]).reshape([bs, -1, self.num_classes]) # [1, 3*6174, 80]
                for cls_score, hw in zip(cls_scores, featmap_sizes)
            ]
            flatten_cls_scores = F.sigmoid(paddle.concat(flatten_cls_scores, 1))

            # flatten_bbox_preds = [
            #     bbox_pred.transpose([0, 2, 3, 1]).reshape([-1, hw[0] * hw[1] * self.num_anchor, 4])
            #     for bbox_pred, hw in zip(bbox_preds, featmap_sizes)
            # ]
            flatten_bbox_preds = [
                bbox_pred.transpose([0, 2, 3, 1]).reshape([bs, -1, 4])
                for bbox_pred, hw in zip(bbox_preds, featmap_sizes)
            ]
            flatten_bbox_preds = paddle.concat(flatten_bbox_preds, 1)

            # flatten_objectness = [
            #     objectness.transpose([0, 2, 3, 1]).reshape([-1, hw[0] * hw[1] * self.num_anchor, 1])
            #     for objectness, hw in zip(objectnesses, featmap_sizes)
            # ]
            flatten_objectness = [
                objectness.transpose([0, 2, 3, 1]).reshape([bs, -1])
                for objectness, hw in zip(objectnesses, featmap_sizes)
            ]
            flatten_objectness = F.sigmoid(paddle.concat(flatten_objectness, 1))

            # print(' flatten_cls_scores .shape', flatten_cls_scores.sum(), flatten_cls_scores.shape)
            # print(' flatten_bbox_preds .shape', flatten_bbox_preds.sum(), flatten_bbox_preds.shape)
            # print(' flatten_objectness .shape', flatten_objectness.sum(), flatten_objectness.shape)
            # flatten_cls_scores = paddle.to_tensor(np.load('flatten_cls_scores.npy'))
            # flatten_bbox_preds = paddle.to_tensor(np.load('flatten_bbox_preds.npy'))
            # flatten_objectness = paddle.to_tensor(np.load('flatten_objectness.npy')) ###

            self.mlvl_priors = self.grid_priors(
                featmap_sizes,
                self.anchors,
                self.stride,
                dtype=cls_scores[0].dtype)
            self.featmap_sizes = featmap_sizes
            flatten_priors = paddle.concat(self.mlvl_priors) # [18522, 4]  sum()=20744640.  sum(0)=[5901021., 3771138., 6545763., 4526718.]

            mlvl_strides = [
                paddle.full([(hw[0] * hw [1]) * self.num_anchor], stride) for
                hw, stride in zip(featmap_sizes, self.stride)
            ]
            flatten_stride = paddle.concat(mlvl_strides) # [18522]
            flatten_decoded_bboxes = self.decode(
                flatten_priors[None], flatten_bbox_preds, flatten_stride)
                # [1, 18522, 4]  [18522, 4]   [18522]
            #flatten_decoded_bboxes = paddle.to_tensor(np.load('flatten_decoded_bboxes.npy'))
            # print(' flatten_decoded_bboxes .shape', flatten_decoded_bboxes.sum(), flatten_decoded_bboxes.shape)
            # [1, 18522, 4]
            return [flatten_cls_scores, flatten_decoded_bboxes, flatten_objectness]

    def grid_priors(self, featmap_sizes, anchors, strides, dtype='float32'):
        multi_level_anchors = []
        for hw, anchor, stride in zip(featmap_sizes, anchors, strides):
            feat_h, feat_w = hw[0], hw[1]
            # shift_yy, shift_xx = paddle.meshgrid([
            #     paddle.arange(0, ny, dtype='float32') * stride, 
            #     paddle.arange(0, nx, dtype='float32') * stride,
            # ])
            shift_y = paddle.arange(0, feat_h, dtype='float32') * stride # [56]
            shift_x = paddle.arange(0, feat_w, dtype='float32') * stride # [84]
            shift_yy, shift_xx = paddle.meshgrid(shift_y, shift_x)
            shifts = paddle.stack([shift_xx, shift_yy, shift_xx, shift_yy], -1).reshape([-1, 4]) # [4704, 4]

            base_anchors = []
            for base_size in anchor:
                w, h = base_size
                x_center, y_center = stride / 2., stride / 2.
                base_anchor = paddle.to_tensor([
                    x_center - 0.5 * w, y_center - 0.5 * h, x_center + 0.5 * w,
                    y_center + 0.5 * h
                ])
                base_anchors.append(base_anchor)
            base_anchors = paddle.stack(base_anchors, 0).squeeze(-1) # [3, 4, 1]->[3, 4]
            # grid = paddle.stack(
            #     (xv, yv), axis=2).expand([1, self.num_anchor, ny, nx, 2])
            # anchor_grid = anchor.reshape([1, self.num_anchor, 1, 1, 2]).expand(
            #     (1, self.num_anchor, ny, nx, 2))
            all_anchors = base_anchors[None, :, :] + shifts[:, None, :] # [1, 3, 4] + [4704, 1, 4]
            all_anchors = all_anchors.reshape([-1, 4])
            multi_level_anchors.append(all_anchors)
        return multi_level_anchors

    def decode(self, priors, pred_bboxes, stride):
        # grid, anchor_grid = self.make_grid(nx, ny, anchor) # [1, 3, 80, 80, 2] [1, 3, 80, 80, 2]
        pred_bboxes = F.sigmoid(pred_bboxes)

        x_center = (priors[..., 0] + priors[..., 2]) * 0.5
        y_center = (priors[..., 1] + priors[..., 3]) * 0.5
        w = priors[..., 2] - priors[..., 0]
        h = priors[..., 3] - priors[..., 1]

        # The anchor of mmdet has been offset by 0.5
        x_center_pred = (pred_bboxes[..., 0] - 0.5) * 2 * stride + x_center
        y_center_pred = (pred_bboxes[..., 1] - 0.5) * 2 * stride + y_center
        w_pred = (pred_bboxes[..., 2] * 2)**2 * w
        h_pred = (pred_bboxes[..., 3] * 2)**2 * h

        decoded_bboxes = paddle.stack(
            (x_center_pred - w_pred / 2, y_center_pred - h_pred / 2,
             x_center_pred + w_pred / 2, y_center_pred + h_pred / 2),
            axis=-1)
        # [1, 18522, 4]
        return decoded_bboxes

    def post_process(self, head_outs, img_meta, rescale=True): #img_shape, scale_factor, im0_shape=[427, 640], pad_param=None, rescale=True):
        cls_scores, bbox_preds, objectnesses = head_outs
        score_thr = self.nms.score_threshold # 0.001
        nms_pre = self.nms.nms_top_k # 30000
        
        if score_thr > 0:
            conf_inds = objectnesses > score_thr # [1, 18522]
            # cls_scores = paddle.masked_select(cls_scores, conf_inds.unsqueeze(-1).tile([1, 1, self.num_classes])).reshape([-1, self.num_classes])
            # bbox_preds = paddle.masked_select(bbox_preds, conf_inds.unsqueeze(-1).tile([1, 1, 4])).reshape([-1, 4])
            # objectnesses = paddle.masked_select(objectnesses, conf_inds).reshape([-1, 1])
            cls_scores = cls_scores[conf_inds] #paddle.masked_select(cls_scores, conf_inds.tile([1, 1, self.num_classes])).reshape([-1, self.num_classes])
            bbox_preds = bbox_preds[conf_inds] #paddle.masked_select(bbox_preds, conf_inds.tile([1, 1, 4])).reshape([-1, 4])
            objectnesses = objectnesses[conf_inds] #paddle.masked_select(objectnesses, conf_inds).reshape([-1, 1])
        cls_scores *= objectnesses[:, None]

        if self.multi_label is False:
            labels = cls_scores.argmax(1, keepdim=True)
            scores = cls_scores.max(1, keepdim=True)
            pred_scores, _, keep_idxs, results = filter_scores_and_topk(
                cls_scores,
                score_thr,
                nms_pre,
                results=dict(labels=labels[:, 0]))
            pred_labels = results['labels']
        else:
            pred_scores, pred_labels, keep_idxs, _ = filter_scores_and_topk(
                cls_scores, score_thr, nms_pre)
        pred_bboxes = bbox_preds[keep_idxs] # [2631, 4]

        if rescale:
            pad_param = img_meta['pad_param'].numpy()[0] #[10., 11., 16., 16.]
            if pad_param is not None:
                pred_bboxes -= paddle.to_tensor([int(pad_param[2]), int(pad_param[0]), int(pad_param[2]), int(pad_param[0])])

            scale_y, scale_x = paddle.split(img_meta['scale_factor'][0], 2, axis=-1)
            scale_factor = paddle.concat([scale_x, scale_y, scale_x, scale_y], -1)
            pred_bboxes /= scale_factor

        keep_top_k = min(self.nms.keep_top_k, pred_bboxes.shape[0])
        out_idxs = paddle.vision.ops.nms(boxes=pred_bboxes,
                                        iou_threshold=self.nms.nms_threshold,
                                        scores=pred_scores,
                                        category_idxs=pred_labels,
                                        categories=self.categories,
                                        top_k=keep_top_k)
        # {'multi_label': True,
        # 'nms_pre': 30000,
        # 'score_thr': 0.001,
        # 'nms': {'type': 'nms', 'iou_threshold': 0.65},
        # 'max_per_img': 300}
        # print(out_idxs)
        if len(out_idxs):
            bboxes = pred_bboxes[out_idxs]
            if len(out_idxs) == 1:
                bboxes = bboxes.reshape([1, 4])
            scores = pred_scores[out_idxs].unsqueeze(-1).astype('float32')
            labels = pred_labels[out_idxs].unsqueeze(-1).astype('float32')
            im0_shape = img_meta['im0_shape'].numpy()[0]

            bboxes[:, 0::2].clip(0, im0_shape[1]) # 640
            bboxes[:, 1::2].clip(0, im0_shape[0]) # 427
            results = paddle.concat([labels, scores, bboxes], axis=1)
            results_num = paddle.shape(results)[0:1]
        else:
            results = paddle.to_tensor(
                np.array(
                    [[-1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype='float32'))
            results_num = paddle.zeros([1])
        return results, results_num

    def get_loss(self, head_outs, gt_meta, anchors):
        cls_scores, bbox_preds, objectnesses = head_outs

        gt_nums = gt_meta['pad_gt_mask'].sum(1).squeeze(-1)
        nt = gt_nums.sum().astype('int32')
        anchors = anchors
        na = anchors.shape[1]  # not len(anchors)
        ai = paddle.tile(paddle.arange(na,dtype=np.float32).reshape([na, 1]), [1, nt])

        batch_size = bbox_preds[0].shape[0]
        gt_labels = []
        for idx in range(batch_size):
            gt_num = gt_nums[idx].astype("int32")
            if gt_num == 0: continue
            gt_bbox = gt_meta['gt_bbox'][idx][:gt_num]
            gt_class = gt_meta['gt_class'][idx][:gt_num] * 1.0
            img_idx = paddle.repeat_interleave(paddle.to_tensor(idx),gt_num,axis=0)[None,:].astype(paddle.float32).T
            gt_labels.append(paddle.concat((img_idx, gt_class, gt_bbox),axis=-1))
        if (len(gt_labels)):
            gt_labels = paddle.concat(gt_labels)
        else:
            gt_labels = paddle.zeros([0, 6])

        batch_targets_normed = paddle.concat((paddle.tile(
            paddle.unsqueeze(gt_labels, 0), [na, 1, 1]), ai[:, :, None]), 2)
        #batch_targets_normed = paddle.to_tensor(np.load('batch_targets_normed.npy'))

        # for x in cls_scores: print(x.shape, x.sum())
        # for x in bbox_preds: print(x.shape, x.sum())
        # for x in objectnesses: print(x.shape, x.sum())

        loss_cls = paddle.zeros([1])
        loss_box = paddle.zeros([1])
        loss_obj = paddle.zeros([1])
        scaled_factor = paddle.ones([7])

        for i in range(self.num_levels):
            batch_size, _, h, w = bbox_preds[i].shape
            target_obj = paddle.zeros_like(objectnesses[i])

            # empty gt bboxes
            if batch_targets_normed.shape[1] == 0:
                loss_box += bbox_preds[i].sum() * 0
                loss_cls += cls_scores[i].sum() * 0
                loss_obj += self.loss_obj(
                    objectnesses[i], target_obj) * self.obj_level_weights[i]
                continue

            priors_base_sizes_i = anchors[i] / self.stride[i]
            # feature map scale whwh
            scaled_factor[2:6] = paddle.to_tensor(
                bbox_preds[i].shape, dtype='float32')[[3, 2, 3, 2]]
            # Scale batch_targets from range 0-1 to range 0-features_maps size.
            # (num_anchor, num_bboxes, 7)
            batch_targets_scaled = batch_targets_normed * scaled_factor

            # 2. Shape match
            wh_ratio = batch_targets_scaled[..., 4:6] / priors_base_sizes_i[:, None]
            match_inds = paddle.maximum(wh_ratio, 1 / wh_ratio).max(2) < self.prior_match_thr
            batch_targets_scaled = batch_targets_scaled[match_inds] # [24, 7]

            # no gt bbox matches anchor
            if batch_targets_scaled.shape[0] == 0:
                loss_box += bbox_preds[i].sum() * 0
                loss_cls += cls_scores[i].sum() * 0
                loss_obj += self.loss_obj(
                    objectnesses[i], target_obj) * self.obj_level_weights[i]
                continue
            # 3. Positive samples with additional neighbors

            # check the left, up, right, bottom sides of the
            # targets grid, and determine whether assigned
            # them as positive samples as well.
            batch_targets_cxcy = batch_targets_scaled[:, 2:4]
            grid_xy = scaled_factor[[2, 3]] - batch_targets_cxcy
            left, up = ((batch_targets_cxcy % 1 < self.near_neighbor_thr) &
                        (batch_targets_cxcy > 1)).T.astype(paddle.int64)
            right, bottom = ((grid_xy % 1 < self.near_neighbor_thr) &
                             (grid_xy > 1)).T.astype(paddle.int64)
            offset_inds = paddle.stack(
                (paddle.ones_like(left), left, up, right, bottom)).astype(paddle.bool)
            batch_targets_scaled = batch_targets_scaled.tile(
                [5, 1, 1])[offset_inds] # [24, 7] -> [72, 7]
            retained_offsets = self.grid_offset[:, None].tile([1, offset_inds.shape[1], 1])[offset_inds]
            # [72, 2] # # [5, 2] -> [72, 7]

            # prepare pred results and positive sample indexes to
            # calculate class loss and bbox lo
            #_chunk_targets = paddle.chunk(batch_targets_scaled, chunks=4, axis=1)
            img_class_inds = batch_targets_scaled[:, :2]
            grid_xy = batch_targets_scaled[:, 2:4]
            grid_wh = batch_targets_scaled[:, 4:6]
            priors_inds = batch_targets_scaled[:, 6:].astype(paddle.int64).reshape([-1])

            # priors_inds = priors_inds.reshape([-1])
            img_inds, class_inds = img_class_inds.astype(paddle.int64).T

            grid_xy_long = (grid_xy -
                            retained_offsets * self.near_neighbor_thr).astype(paddle.int64)
            grid_x_inds, grid_y_inds = grid_xy_long.T
            bboxes_targets = paddle.concat((grid_xy - grid_xy_long, grid_wh), 1) # [72, 4]

            # 4. Calculate loss
            # bbox loss
            # [8, 3, 4, 80, 80]
            try:
                bbox_pred_= bbox_preds[i].reshape([batch_size, self.num_anchor, -1, h, w]).transpose(
                    (0, 1, 3, 4, 2))
                mask = paddle.stack([img_inds, priors_inds, grid_y_inds, grid_x_inds], 1)
                #retained_bbox_pred = bbox_pred_[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]
                retained_bbox_pred = bbox_pred_.gather_nd(mask) # [72, 4]
            except:
                print('yolov5head_rect get_loss ')
                import pdb; pdb.set_trace()
            priors_base_sizes_i = priors_base_sizes_i[priors_inds] # [72, 2]
            decoded_bbox_pred = self._decode_bbox_to_xywh(
                retained_bbox_pred, priors_base_sizes_i)
            loss_box_i, iou = self.loss_bbox(
                decoded_bbox_pred.split(4, axis=-1),
                bboxes_targets.split(4, axis=-1))
            loss_box_i = loss_box_i.mean() * self.loss_weights['box']
            #print(i, ' box ', loss_box_i)
            loss_box += loss_box_i

            # obj loss
            iou = iou.detach().clip(0)
            score_iou = paddle.cast(iou.detach().squeeze(-1).clip(0), target_obj.dtype)
            # target_obj[img_inds, priors_inds, grid_y_inds,
            #            grid_x_inds] = iou.type(target_obj.dtype)
            with paddle.no_grad():
                x = paddle.gather_nd(target_obj, mask)
                target_obj = paddle.scatter_nd_add(
                    target_obj, mask, score_iou - x)
            loss_obj_i = self.loss_obj(objectnesses[i], target_obj) * self.obj_level_weights[i] * self.loss_weights['obj']
            #print(i, 'obj ', loss_obj_i)
            loss_obj += loss_obj_i

            # cls loss
            if self.num_classes > 1:
                cls_scores_ = cls_scores[i].reshape([batch_size, self.num_anchor, -1, h, w]).transpose(
                    (0, 1, 3, 4, 2))
                pred_cls_scores = cls_scores_.gather_nd(mask) # [72, 4]

                target_class = paddle.full_like(pred_cls_scores, 0.)
                target_class[range(batch_targets_scaled.shape[0]), class_inds] = 1.
                loss_cls_i = self.loss_cls(pred_cls_scores, target_class) * self.loss_weights['cls']
                #print(i, 'cls ', loss_cls_i)
                loss_cls += loss_cls_i
            else:
                loss_cls += cls_scores[i].sum() * 0

        batch_size = cls_scores[0].shape[0]
        num_gpus = gt_meta.get('num_gpus', 8)
        yolo_losses = dict(
            loss_cls=loss_cls * batch_size * num_gpus,
            loss_obj=loss_obj * batch_size * num_gpus,
            loss_bbox=loss_box * batch_size * num_gpus)
        yolo_losses['loss'] = (loss_cls + loss_obj + loss_box) * batch_size * num_gpus
        return yolo_losses

    def _decode_bbox_to_xywh(self, bbox_pred, priors_base_sizes):
        bbox_pred = F.sigmoid(bbox_pred)
        pred_xy = bbox_pred[:, :2] * 2 - 0.5
        pred_wh = (bbox_pred[:, 2:] * 2)**2 * priors_base_sizes
        decoded_bbox_pred = paddle.concat((pred_xy, pred_wh), -1)
        return decoded_bbox_pred


def filter_scores_and_topk(scores, score_thr, topk, results=None):
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = paddle.nonzero(valid_mask)
    num_topk = min(topk, valid_idxs.size)
    # .sort is actually faster than .topk (at least on GPUs)
    idxs = scores.argsort(descending=True)
    scores = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(axis=1)

    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, paddle.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(f'Only supports dict or list or Tensor, '
                                      f'but get {type(results)}.')
    return scores, labels, keep_idxs, filtered_results
