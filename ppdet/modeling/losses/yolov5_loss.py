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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from ..bbox_utils import bbox_iou

__all__ = ['YOLOv5Loss', 'YOLOv5InsLoss']


@register
class YOLOv5Loss(nn.Layer):
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 downsample_ratios=[8, 16, 32],
                 balance=[4.0, 1.0, 0.4],
                 box_weight=0.05,
                 obj_weight=1.0,
                 cls_weght=0.5,
                 bias=0.5,
                 anchor_t=4.0,
                 label_smooth_eps=0.):
        super(YOLOv5Loss, self).__init__()
        self.num_classes = num_classes
        self.balance = balance
        self.na = 3  # not len(anchors)
        self.gr = 1.0

        self.BCEcls = nn.BCEWithLogitsLoss(reduction="mean")
        self.BCEobj = nn.BCEWithLogitsLoss(reduction="mean")

        self.loss_weights = {
            'box': box_weight,
            'obj': obj_weight,
            'cls': cls_weght,
        }

        eps = label_smooth_eps if label_smooth_eps > 0 else 0.
        self.cls_pos_label = 1.0 - 0.5 * eps
        self.cls_neg_label = 0.5 * eps

        self.downsample_ratios = downsample_ratios
        self.bias = bias  # named 'g' in torch yolov5
        self.off = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
            ],
            dtype=np.float32) * bias  # offsets
        self.anchor_t = anchor_t
        self.to_static = False

    def build_targets(self, outputs, targets, anchors):
        if 0:
            # collate_batch True
            # targets['gt_class'] [bs, max_gt_nums, 1]
            # targets['gt_bbox'] [bs, max_gt_nums, 4]
            # targets['pad_gt_mask'] [bs, max_gt_nums, 1]
            gt_nums = targets['pad_gt_mask'].sum(1).squeeze(-1).numpy()
            nt = int(sum(gt_nums))
            anchors = anchors.numpy()
            na = anchors.shape[1]  # not len(anchors)
            tcls, tbox, indices, anch = [], [], [], []

            gain = np.ones(7, dtype=np.float32)  # normalized to gridspace gain
            ai = np.tile(
                np.arange(
                    na, dtype=np.float32).reshape(na, 1), [1, nt])

            batch_size = outputs[0].shape[0]
            gt_labels = []
            for idx in range(batch_size):
                gt_num = int(gt_nums[idx])
                if gt_num == 0:
                    continue
                gt_bbox = targets['gt_bbox'][idx][:gt_num].numpy()
                gt_class = targets['gt_class'][idx][:gt_num].numpy() * 1.0
                img_idx = np.repeat(np.array([[idx]]), gt_num, axis=0)
                gt_labels.append(
                    np.concatenate((img_idx, gt_class, gt_bbox), -1))
        else:
            gt_nums = [len(bbox) for bbox in targets['gt_bbox']]
            nt = int(sum(gt_nums))
            anchors = anchors.numpy()
            na = anchors.shape[1]  # not len(anchors)
            tcls, tbox, indices, anch = [], [], [], []

            gain = np.ones(7, dtype=np.float32)  # normalized to gridspace gain
            ai = np.tile(
                np.arange(
                    na, dtype=np.float32).reshape(na, 1), [1, nt])

            batch_size = outputs[0].shape[0]
            gt_labels = []
            for idx in range(batch_size):
                gt_num = gt_nums[idx]
                if gt_num == 0:
                    continue
                gt_bbox = targets['gt_bbox'][idx][:gt_num].numpy()
                gt_class = targets['gt_class'][idx][:gt_num].numpy() * 1.0
                img_idx = np.repeat(np.array([[idx]]), gt_num, axis=0)
                gt_labels.append(
                    np.concatenate((img_idx, gt_class, gt_bbox), -1))

        if (len(gt_labels)):
            gt_labels = np.concatenate(gt_labels)
        else:
            gt_labels = np.zeros([0, 6])

        targets_labels = np.concatenate((np.tile(
            np.expand_dims(gt_labels, 0), [na, 1, 1]), ai[:, :, None]), 2)
        g = self.bias  # 0.5

        for i in range(len(anchors)):
            anchor = np.array(anchors[i]) / self.downsample_ratios[i]
            gain[2:6] = np.array(
                outputs[i].shape, dtype=np.float32)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets_labels to
            t = targets_labels * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchor[:, None]
                j = np.maximum(r, 1 / r).max(2) < self.anchor_t
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = np.stack((np.ones_like(j), j, k, l, m))
                t = np.tile(t, [5, 1, 1])[j]
                offsets = (np.zeros_like(gxy)[None] + self.off[:, None])[j]
            else:
                t = targets_labels[0]
                offsets = 0

            # Define
            b, c = t[:, :2].astype(np.int64).T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).astype(np.int64)
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].astype(np.int64)  # anchor indices
            gj, gi = gj.clip(0, gain[3] - 1), gi.clip(0, gain[2] - 1)
            indices.append(
                (paddle.to_tensor(b), paddle.to_tensor(a),
                 paddle.to_tensor(gj, 'int64'), paddle.to_tensor(gi, 'int64')))
            tbox.append(
                paddle.to_tensor(
                    np.concatenate((gxy - gij, gwh), 1), dtype=paddle.float32))
            anch.append(paddle.to_tensor(anchor[a]))
            tcls.append(paddle.to_tensor(c))
        return tcls, tbox, indices, anch

    def yolov5_loss(self, pi, t_cls, t_box, t_indices, t_anchor, balance):
        loss = dict()
        b, a, gj, gi = t_indices  # image, anchor, gridy, gridx
        n = b.shape[0]  # number of targets
        tobj = paddle.zeros_like(pi[:, :, :, :, 4])
        loss_box = paddle.to_tensor([0.])
        loss_cls = paddle.to_tensor([0.])
        if n:
            mask = paddle.stack([b, a, gj, gi], 1)
            ps = pi.gather_nd(mask)
            # Regression
            pxy = F.sigmoid(ps[:, :2]) * 2 - 0.5
            pwh = (F.sigmoid(ps[:, 2:4]) * 2)**2 * t_anchor
            pbox = paddle.concat((pxy, pwh), 1)
            iou = bbox_iou(pbox.T, t_box.T, x1y1x2y2=False, ciou=True)
            loss_box = (1.0 - iou).mean()

            # Objectness
            score_iou = paddle.cast(iou.detach().clip(0), tobj.dtype)
            # with paddle.no_grad():
            #     x = paddle.gather_nd(tobj, mask)
            #     tobj = paddle.scatter_nd_add(
            #         tobj, mask, (1.0 - self.gr) + self.gr * score_iou - x)
            with paddle.no_grad():
                tobj[b, a, gj, gi] = (1.0 - self.gr
                                      ) + self.gr * score_iou  # iou ratio

            # Classification
            if self.num_classes > 1:  # cls loss (only if multiple classes)
                # t = paddle.full_like(ps[:, 5:], self.cls_neg_label)
                # t[range(n), t_cls] = self.cls_pos_label
                # loss_cls = self.BCEcls(ps[:, 5:], t)

                t = paddle.full_like(ps[:, 5:], self.cls_neg_label)
                if not self.to_static:
                    t = paddle.put_along_axis(
                        t,
                        t_cls.unsqueeze(-1),
                        values=self.cls_pos_label,
                        axis=1)
                else:
                    for i in range(n):
                        t[i, t_cls[i]] = self.cls_pos_label

                loss_cls = self.BCEcls(ps[:, 5:], t)

        obji = self.BCEobj(pi[:, :, :, :, 4], tobj)  # [bs, 3, h, w]

        loss_obj = obji * balance

        loss['loss_box'] = loss_box * self.loss_weights['box']
        loss['loss_obj'] = loss_obj * self.loss_weights['obj']
        loss['loss_cls'] = loss_cls * self.loss_weights['cls']
        return loss

    def forward(self, inputs, targets, anchors):
        yolo_losses = dict()
        if not self.to_static:
            tcls, tbox, indices, anch = self.build_targets(inputs, targets,
                                                           anchors)
        else:
            tcls, tbox, indices, anch = self.build_targets_paddle(
                inputs, targets, anchors)

        for i, (p_det, balance) in enumerate(zip(inputs, self.balance)):
            t_cls = tcls[i]
            t_box = tbox[i]
            t_anchor = anch[i]
            t_indices = indices[i]

            bs, ch, h, w = p_det.shape
            pi = p_det.reshape(
                (bs, self.na, int(ch / self.na), h, w)).transpose(
                    (0, 1, 3, 4, 2))

            yolo_loss = self.yolov5_loss(pi, t_cls, t_box, t_indices, t_anchor,
                                         balance)

            for k, v in yolo_loss.items():
                if k in yolo_losses:
                    yolo_losses[k] += v
                else:
                    yolo_losses[k] = v

        batch_size = inputs[0].shape[0]
        num_gpus = targets.get('num_gpus', 8)
        loss = 0
        for k, v in yolo_losses.items():
            yolo_losses[k] = v * batch_size * num_gpus
            loss += yolo_losses[k]
        yolo_losses['loss'] = loss
        return yolo_losses

    def build_targets_paddle(self, outputs, targets, anchors):
        # targets['gt_class'] [bs, max_gt_nums, 1]
        # targets['gt_bbox'] [bs, max_gt_nums, 4]
        # targets['pad_gt_mask'] [bs, max_gt_nums, 1]
        gt_nums = [len(bbox) for bbox in targets['gt_bbox']]
        nt = int(sum(gt_nums))
        anchors = anchors
        na = anchors.shape[1]  # not len(anchors)
        tcls, tbox, indices, anch = [], [], [], []

        gain = paddle.ones(
            [7], dtype=paddle.float32)  # normalized to gridspace gain
        ai = paddle.tile(
            paddle.arange(
                na, dtype=paddle.float32).reshape([na, 1]), [1, nt])

        batch_size = outputs[0].shape[0]
        gt_labels = []
        for i, (
                gt_num, gt_bboxs, gt_classes
        ) in enumerate(zip(gt_nums, targets['gt_bbox'], targets['gt_class'])):
            if gt_num == 0:
                continue
            gt_bbox = gt_bboxs[:gt_num].astype('float32')
            gt_class = (gt_classes[:gt_num] * 1.0).astype('float32')
            img_idx = paddle.repeat_interleave(
                paddle.to_tensor([i]), gt_num,
                axis=0)[None, :].astype('float32').T

            gt_labels.append(
                paddle.concat(
                    (img_idx, gt_class, gt_bbox), axis=-1))

        if (len(gt_labels)):
            gt_labels = paddle.concat(gt_labels)
        else:
            gt_labels = paddle.zeros([0, 6], dtype=paddle.float32)

        targets_labels = paddle.concat((paddle.tile(
            paddle.unsqueeze(gt_labels, 0), [na, 1, 1]), ai[:, :, None]), 2)
        g = self.bias  # 0.5

        for i in range(len(anchors)):
            anchor = anchors[i] / self.downsample_ratios[i]
            gain[2:6] = paddle.to_tensor(
                outputs[i].shape,
                dtype=paddle.float32)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets_labels to
            t = targets_labels * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchor[:, None]
                j = paddle.maximum(r, 1 / r).max(2) < self.anchor_t
                t = paddle.flatten(t, 0, 1)
                j = paddle.flatten(j.astype(paddle.int32), 0,
                                   1).astype(paddle.bool)
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T.astype(paddle.int64)
                l, m = ((gxi % 1 < g) & (gxi > 1)).T.astype(paddle.int64)
                j = paddle.flatten(
                    paddle.stack((paddle.ones_like(j), j, k, l, m)), 0,
                    1).astype(paddle.bool)
                t = paddle.flatten(paddle.tile(t, [5, 1, 1]), 0, 1)
                t = t[j]
                offsets = paddle.zeros_like(gxy)[None, :] + paddle.to_tensor(
                    self.off)[:, None]
                offsets = paddle.flatten(offsets, 0, 1)[j]
            else:
                t = targets_labels[0]
                offsets = 0

            # Define
            b, c = t[:, :2].astype(paddle.int64).T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).astype(paddle.int64)
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].astype(paddle.int64)  # anchor indices
            gj, gi = gj.clip(0, gain[3] - 1), gi.clip(0, gain[2] - 1)
            indices.append(
                (b, a, gj.astype(paddle.int64), gi.astype(paddle.int64)))
            tbox.append(
                paddle.concat((gxy - gij.astype(gxy.dtype), gwh), 1).astype(paddle.float32))
            anch.append(anchor[a])
            tcls.append(c)
        return tcls, tbox, indices, anch


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, paddle.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


@register
class YOLOv5InsLoss(nn.Layer):
    __shared__ = ['num_classes']

    def __init__(
            self,
            num_classes=80,
            downsample_ratios=[8, 16, 32],
            balance=[4.0, 1.0, 0.4],
            box_weight=0.05,
            obj_weight=1.0,
            cls_weght=0.5,
            bias=0.5,
            anchor_t=4.0,
            overlap=True,  #
            label_smooth_eps=0.):
        super(YOLOv5InsLoss, self).__init__()
        self.num_classes = num_classes
        self.overlap = overlap
        self.balance = balance
        self.na = 3  # not len(anchors)
        self.gr = 1.0

        self.BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=paddle.to_tensor([1.0]), reduction="mean")
        self.BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=paddle.to_tensor([1.0]), reduction="mean")

        self.loss_weights = {
            'box': box_weight,
            'obj': obj_weight,
            'cls': cls_weght,
        }

        eps = label_smooth_eps if label_smooth_eps > 0 else 0.
        self.cls_pos_label = 1.0 - 0.5 * eps
        self.cls_neg_label = 0.5 * eps

        self.downsample_ratios = downsample_ratios
        self.bias = bias  # named 'g' in torch yolov5
        self.off = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
            ],
            dtype=np.float32) * bias  # offsets
        self.anchor_t = anchor_t
        self.to_static = False

    def build_targets(self, outputs, targets, anchors):
        if 0:
            # collate_batch True
            # targets['gt_class'] [bs, max_gt_nums, 1]
            # targets['gt_bbox'] [bs, max_gt_nums, 4]
            # targets['pad_gt_mask'] [bs, max_gt_nums, 1]
            gt_nums = targets['pad_gt_mask'].sum(1).squeeze(-1).numpy()
            nt = int(sum(gt_nums))
            anchors = anchors.numpy()
            na = anchors.shape[1]  # not len(anchors)
            tcls, tbox, indices, anch = [], [], [], []
            tidxs, xywhn = [], []

            gain = np.ones(8, dtype=np.float32)  # normalized to gridspace gain
            ai = np.tile(
                np.arange(
                    na, dtype=np.float32).reshape(na, 1), [1, nt])

            batch_size = outputs[0].shape[0]
            gt_labels = []
            for idx in range(batch_size):
                gt_num = int(gt_nums[idx])
                if gt_num == 0:
                    continue
                gt_bbox = targets['gt_bbox'][idx][:gt_num].numpy()
                gt_class = targets['gt_class'][idx][:gt_num].numpy() * 1.0
                img_idx = np.repeat(np.array([[idx]]), gt_num, axis=0)
                gt_labels.append(
                    np.concatenate((img_idx, gt_class, gt_bbox), -1))
        else:
            gt_nums = [len(bbox) for bbox in targets['gt_bbox']]
            nt = int(sum(gt_nums))
            anchors = anchors.numpy()
            na = anchors.shape[1]  # not len(anchors)
            tcls, tbox, indices, anch = [], [], [], []
            tidxs, xywhn = [], []

            gain = np.ones(8, dtype=np.float32)  # normalized to gridspace gain
            ai = np.tile(
                np.arange(
                    na, dtype=np.float32).reshape(na, 1), [1, nt])

            batch_size = outputs[0].shape[0]
            gt_labels = []
            for idx in range(batch_size):
                gt_num = gt_nums[idx]
                if gt_num == 0: continue
                gt_bbox = targets['gt_bbox'][idx][:gt_num]
                gt_class = targets['gt_class'][idx][:gt_num] * 1.0
                img_idx = np.repeat(np.array([[idx]]), gt_num, axis=0)
                gt_labels.append(
                    np.concatenate((img_idx, gt_class, gt_bbox), -1))

        if (len(gt_labels)):
            gt_labels = np.concatenate(gt_labels)
        else:
            gt_labels = np.zeros([0, 6])

        if self.overlap:
            batch = outputs[0].shape[0]
            ti = []
            for i in range(batch):
                num = (gt_labels[:, 0] == i
                       ).sum()  # find number of targets of each image
                if num == 0: continue
                ti.append(
                    np.tile(np.arange(num).reshape(1, num), [na, 1]) +
                    1)  # (na, num)
            ti = np.concatenate(ti, 1)  # (na, num_gts)
        else:
            ti = np.tile(np.arange(nt).reshape(1, nt), [na, 1])

        targets_labels = np.concatenate((np.tile(
            np.expand_dims(gt_labels, 0), [na, 1, 1]), ai[:, :, None],
                                         ti[..., None]), 2)
        g = self.bias  # 0.5

        for i in range(len(anchors)):
            anchor = np.array(anchors[i]) / self.downsample_ratios[i]
            gain[2:6] = np.array(
                outputs[i].shape, dtype=np.float32)[[3, 2, 3, 2]]  # xyxy gain
            # gain = [ 1.,  1., 80., 80., 80., 80.,  1.,  1.]

            # Match targets_labels to
            t = targets_labels * gain  # shape (3,nt,8) * (8) = (3, nt, 8)
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchor[:, None]  # (3, nt, 2)
                j = np.maximum(r, 1 / r).max(2) < self.anchor_t
                t = t[j]  # filter # (m, 8)

                # Offsets
                gxy = t[:, 2:4]  # grid xy  # (m, 2)
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = np.stack((np.ones_like(j), j, k, l, m))
                t = np.tile(t, [5, 1, 1])[j]
                offsets = (np.zeros_like(gxy)[None] + self.off[:, None])[j]
            else:
                t = targets_labels[0]
                offsets = 0

            # Define
            b, c = t[:, :2].astype(np.int64).T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).astype(np.int64)
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].astype(np.int64)  # anchor indices
            gj, gi = gj.clip(0, gain[3] - 1), gi.clip(0, gain[2] - 1)
            indices.append(
                (paddle.to_tensor(b), paddle.to_tensor(a),
                 paddle.to_tensor(gj, 'int64'), paddle.to_tensor(gi, 'int64')))
            tbox.append(
                paddle.to_tensor(
                    np.concatenate((gxy - gij, gwh), 1), dtype=paddle.float32))
            anch.append(paddle.to_tensor(anchor[a]))
            tcls.append(paddle.to_tensor(c))

            tidx = t[:, 7].astype(np.int64)  # tidx
            tidxs.append(paddle.to_tensor(tidx))
            xywhn.append(
                paddle.to_tensor(
                    t[:, 2:6],
                    dtype=paddle.float32) / gain[2:6])  # xywh normalized

        return tcls, tbox, indices, anch, tidxs, xywhn

    def yolov5ins_loss(self, pi, proto, t_cls, t_box, t_indices, t_anchor,
                       t_tidxs, t_xywhn, masks, balance):
        bs, nm, mask_h, mask_w = proto.shape

        loss = dict()
        b, a, gj, gi = t_indices  # image, anchor, gridy, gridx
        n = b.shape[0]  # number of targets
        tobj = paddle.zeros_like(pi[:, :, :, :, 4])
        loss_box = paddle.to_tensor([0.])
        loss_cls = paddle.to_tensor([0.])
        loss_seg = paddle.to_tensor([0.])
        if n:
            mask = paddle.stack([b, a, gj, gi], 1)
            ps = pi.gather_nd(mask)
            ps, pmask = ps.split([5 + self.num_classes, nm], -1)  ###

            # Regression
            pxy = F.sigmoid(ps[:, :2]) * 2 - 0.5
            pwh = (F.sigmoid(ps[:, 2:4]) * 2)**2 * t_anchor
            pbox = paddle.concat((pxy, pwh), 1)
            iou = bbox_iou(pbox.T, t_box.T, x1y1x2y2=False, ciou=True)
            loss_box = (1.0 - iou).mean()

            # Objectness
            score_iou = paddle.cast(iou.detach().clip(0), tobj.dtype)
            # with paddle.no_grad():
            #     x = paddle.gather_nd(tobj, mask)
            #     tobj = paddle.scatter_nd_add(
            #         tobj, mask, (1.0 - self.gr) + self.gr * score_iou - x)
            with paddle.no_grad():
                tobj[b, a, gj, gi] = (1.0 - self.gr
                                      ) + self.gr * score_iou  # iou ratio

            # Classification
            if self.num_classes > 1:  # cls loss (only if multiple classes)
                # t = paddle.full_like(ps[:, 5:], self.cls_neg_label)
                # t[range(n), t_cls] = self.cls_pos_label
                # loss_cls = self.BCEcls(ps[:, 5:], t)

                t = paddle.full_like(ps[:, 5:], self.cls_neg_label)
                if not self.to_static:
                    t = paddle.put_along_axis(
                        t,
                        t_cls.unsqueeze(-1),
                        values=self.cls_pos_label,
                        axis=1)
                else:
                    for i in range(n):
                        t[i, t_cls[i]] = self.cls_pos_label

                loss_cls = self.BCEcls(ps[:, 5:], t)

            # Mask regression
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(
                    masks[None], (mask_h, mask_w), mode='nearest')[0]
            marea = t_xywhn[:, 2:].prod(
                1)  # mask width, height normalized, shape: [n]
            mxyxy = xywh2xyxy(t_xywhn * paddle.to_tensor(
                [mask_w, mask_h, mask_w, mask_h]))  # [n, 4]

            for bi in b.unique():
                j = b == bi  # matching index
                if j.sum() == 0:
                    continue
                if self.overlap:
                    mask_gti = paddle.where(
                        masks[bi][None] == t_tidxs[j].reshape([-1, 1, 1]), 1.0,
                        0.0)
                else:
                    mask_gti = masks[t_tidxs][j]
                mask_gti = mask_gti.cast('float32')
                loss_seg += self.single_mask_loss(mask_gti, pmask[j], proto[bi],
                                                  mxyxy[j], marea[j], nm)
                #                            [n, 160, 160] [n, 32] [32, 160, 160]  [n, 4]  [n] 32

        obji = self.BCEobj(pi[:, :, :, :, 4], tobj)  # [bs, 3, h, w]
        loss_obj = obji * balance

        loss['loss_box'] = loss_box * self.loss_weights['box']
        loss['loss_obj'] = loss_obj * self.loss_weights['obj']
        loss['loss_cls'] = loss_cls * self.loss_weights['cls']
        loss['loss_seg'] = loss_seg * self.loss_weights['box'] / bs
        return loss

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area, nm):
        # Mask loss for one image
        pred_mask = (pred @proto.reshape([nm, -1])).reshape(
            [-1, *proto.shape[1:]])  # (n,32) @ (32,80,80) -> (n,80,80)
        loss = F.binary_cross_entropy_with_logits(
            pred_mask, gt_mask, reduction='none')
        return (crop_mask(loss, xyxy).mean(axis=(1, 2)) / area).mean()

    def forward(self, inputs, targets, anchors):
        proto = inputs[-1]

        yolo_losses = dict()
        if not self.to_static:
            tcls, tbox, indices, anch, tidxs, xywhn = self.build_targets(
                inputs[:-1], targets, anchors)
        else:
            tcls, tbox, indices, anch, tidxs, xywhn = self.build_targets_paddle(
                inputs[:-1], targets, anchors)

        masks = paddle.concat(targets['gt_segm'], 0).cast('float32')
        for i, (p_det_ins,
                balance) in enumerate(zip(inputs[:-1], self.balance)):
            t_cls = tcls[i]
            t_box = tbox[i]
            t_anchor = anch[i]
            t_indices = indices[i]
            t_tidxs = tidxs[i]
            t_xywhn = xywhn[i]

            bs, ch, h, w = p_det_ins.shape
            pi = p_det_ins.reshape(
                (bs, self.na, int(ch / self.na), h, w)).transpose(
                    (0, 1, 3, 4, 2))

            yolo_loss = self.yolov5ins_loss(pi, proto, t_cls, t_box, t_indices,
                                            t_anchor, t_tidxs, t_xywhn, masks,
                                            balance)

            for k, v in yolo_loss.items():
                if k in yolo_losses:
                    yolo_losses[k] += v
                else:
                    yolo_losses[k] = v

        batch_size = inputs[0].shape[0]
        num_gpus = targets.get('num_gpus', 8)
        loss = 0
        for k, v in yolo_losses.items():
            yolo_losses[k] = v * batch_size * num_gpus
            loss += yolo_losses[k]
        yolo_losses['loss'] = loss
        return yolo_losses

    def build_targets_paddle(self, outputs, targets, anchors):
        # targets['gt_class'] [bs, max_gt_nums, 1]
        # targets['gt_bbox'] [bs, max_gt_nums, 4]
        # targets['pad_gt_mask'] [bs, max_gt_nums, 1]
        gt_nums = [len(bbox) for bbox in targets['gt_bbox']]
        nt = int(sum(gt_nums))
        anchors = anchors
        na = anchors.shape[1]  # not len(anchors)
        tcls, tbox, indices, anch = [], [], [], []

        gain = paddle.ones(
            [7], dtype=paddle.float32)  # normalized to gridspace gain
        ai = paddle.tile(
            paddle.arange(
                na, dtype=paddle.float32).reshape([na, 1]), [1, nt])

        batch_size = outputs[0].shape[0]
        gt_labels = []
        for i, (
                gt_num, gt_bboxs, gt_classes
        ) in enumerate(zip(gt_nums, targets['gt_bbox'], targets['gt_class'])):
            if gt_num == 0:
                continue
            gt_bbox = gt_bboxs[:gt_num].astype('float32')
            gt_class = (gt_classes[:gt_num] * 1.0).astype('float32')
            img_idx = paddle.repeat_interleave(
                paddle.to_tensor([i]), gt_num,
                axis=0)[None, :].astype('float32').T

            gt_labels.append(
                paddle.concat(
                    (img_idx, gt_class, gt_bbox), axis=-1))

        if (len(gt_labels)):
            gt_labels = paddle.concat(gt_labels)
        else:
            gt_labels = paddle.zeros([0, 6], dtype=paddle.float32)

        targets_labels = paddle.concat((paddle.tile(
            paddle.unsqueeze(gt_labels, 0), [na, 1, 1]), ai[:, :, None]), 2)
        g = self.bias  # 0.5

        for i in range(len(anchors)):
            anchor = anchors[i] / self.downsample_ratios[i]
            gain[2:6] = paddle.to_tensor(
                outputs[i].shape,
                dtype=paddle.float32)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets_labels to
            t = targets_labels * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchor[:, None]
                j = paddle.maximum(r, 1 / r).max(2) < self.anchor_t
                t = paddle.flatten(t, 0, 1)
                j = paddle.flatten(j.astype(paddle.int32), 0,
                                   1).astype(paddle.bool)
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T.astype(paddle.int64)
                l, m = ((gxi % 1 < g) & (gxi > 1)).T.astype(paddle.int64)
                j = paddle.flatten(
                    paddle.stack((paddle.ones_like(j), j, k, l, m)), 0,
                    1).astype(paddle.bool)
                t = paddle.flatten(paddle.tile(t, [5, 1, 1]), 0, 1)
                t = t[j]
                offsets = paddle.zeros_like(gxy)[None, :] + paddle.to_tensor(
                    self.off)[:, None]
                offsets = paddle.flatten(offsets, 0, 1)[j]
            else:
                t = targets_labels[0]
                offsets = 0

            # Define
            b, c = t[:, :2].astype(paddle.int64).T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).astype(paddle.int64)
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].astype(paddle.int64)  # anchor indices
            gj, gi = gj.clip(0, gain[3] - 1), gi.clip(0, gain[2] - 1)
            indices.append(
                (b, a, gj.astype(paddle.int64), gi.astype(paddle.int64)))
            tbox.append(
                paddle.concat((gxy - gij, gwh), 1).astype(paddle.float32))
            anch.append(anchor[a])
            tcls.append(c)
        return tcls, tbox, indices, anch


def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box

    Args:
      masks (paddle.Tensor): [n, h, w] tensor of masks
      boxes (paddle.Tensor): [n, 4] tensor of bbox coordinates in relative point form

    Returns:
      (paddle.Tensor): The masks are being cropped to the bounding box.
    """
    _, h, w = masks.shape  # [n, 160, 160]
    x1, y1, x2, y2 = paddle.chunk(boxes[:, :, None], 4, axis=1)
    # x1 shape(n,1,1) # [n, 1, 1]
    r = paddle.arange(w, dtype=x1.dtype)[None, None, :]
    # rows shape(1,w,1) # [1, 1, w]
    c = paddle.arange(h, dtype=y1.dtype)[None, :, None]
    # cols shape(h,1,1) # [1, h, 1]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
