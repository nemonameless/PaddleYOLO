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
from IPython import embed

__all__ = ['YOLOv7Loss']


@register
class YOLOv7Loss(nn.Layer):
    """
    this code is based on https://github.com/WongKinYiu/yolov7
    """
    __shared__ = ['num_classes', 'use_aux']

    def __init__(self,
                 num_classes=80,
                 downsample_ratios=[8, 16, 32],
                 balance=[4.0, 1.0, 0.4],
                 box_weight=0.05,
                 cls_weght=0.3,
                 obj_weight=0.7,
                 bias=0.5,
                 anchor_t=4.0,
                 label_smooth_eps=0.,
                 use_aux=False):
        super(YOLOv7Loss, self).__init__()
        self.num_classes = num_classes
        self.balance = balance
        self.use_aux = use_aux
        if self.use_aux:
            self.balance = balance * 2
        self.na = 3  # len(anchors[0]) not len(anchors)
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
        if self.use_aux:
            self.downsample_ratios = downsample_ratios * 2
        self.bias = bias  # named 'g' in torch yolov5/yolov7
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

    def forward(self, head_outs, gt_targets, anchors):
        self.nl = len(anchors)

        # 1.split head_outs feature from [b,c,h,w] to [b,na,c//na,h,w]
        inputs = []
        for i in range(self.nl):
            pi = head_outs[i]
            bs, _, h, w = pi.shape
            pi = pi.reshape((bs, self.na, -1, h, w)).transpose((0, 1, 3, 4, 2))
            inputs.append(pi)
        if self.use_aux:
            for i in range(self.nl):
                pi = head_outs[i + self.nl]
                bs, _, h, w = pi.shape
                pi = pi.reshape((bs, self.na, -1, h, w)).transpose(
                    (0, 1, 3, 4, 2))
                inputs.append(pi)

        # 2.generate targets_labels [nt, 6] from gt_targets(dict)
        anchors = anchors.numpy()
        if 1:
            # collate_batch True
            # gt_targets['gt_class'] [bs, max_gt_nums, 1]
            # gt_targets['gt_bbox'] [bs, max_gt_nums, 4]
            # gt_targets['pad_gt_mask'] [bs, max_gt_nums, 1]
            gt_nums = gt_targets['pad_gt_mask'].sum(1).squeeze(-1).numpy()
            batch_size = head_outs[0].shape[0]
            targets_labels = []  # [nt, 6]
            for idx in range(batch_size):
                gt_num = int(gt_nums[idx])
                if gt_num == 0:
                    continue
                gt_bbox = gt_targets['gt_bbox'][idx][:gt_num].reshape(
                    [-1, 4]).numpy()
                gt_class = gt_targets['gt_class'][idx][:gt_num].reshape(
                    [-1, 1]).numpy() * 1.0
                img_idx = np.repeat(np.array([[idx]]), gt_num, axis=0)
                targets_labels.append(
                    np.concatenate((img_idx, gt_class, gt_bbox), -1))
        else:
            gt_nums = [len(bbox) for bbox in gt_targets['gt_bbox']]
            batch_size = head_outs[0].shape[0]
            targets_labels = []  # [nt, 6]
            for idx in range(batch_size):
                gt_num = int(gt_nums[idx])
                if gt_num == 0:
                    continue
                gt_bbox = gt_targets['gt_bbox'][idx][:gt_num].reshape(
                    [-1, 4])
                gt_class = gt_targets['gt_class'][idx][:gt_num].reshape(
                    [-1, 1]) * 1.0
                img_idx = np.repeat(np.array([[idx]]), gt_num, axis=0)
                targets_labels.append(
                    np.concatenate((img_idx, gt_class, gt_bbox), -1))

        if (len(targets_labels)):
            targets_labels = np.concatenate(targets_labels)
        else:
            targets_labels = np.zeros([0, 6])

        # 3.build targets
        batch_images = gt_targets['image']  # just get shape
        if not self.use_aux:
            bs, as_, gjs, gis, targets, anchors = self.build_targets(
                inputs, targets_labels, anchors, batch_images)
            pre_gen_gains = [
                paddle.to_tensor(pp.shape, 'float32')[[3, 2, 3, 2]]
                for pp in inputs
            ]
        else:
            bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets2(
                inputs[:self.nl], targets_labels, anchors, batch_images)
            bs, as_, gjs, gis, targets, anchors = self.build_targets(
                inputs[:self.nl], targets_labels, anchors, batch_images)
            pre_gen_gains_aux = [
                paddle.to_tensor(pp.shape, 'float32')[[3, 2, 3, 2]]
                for pp in inputs[:self.nl]
            ]
            pre_gen_gains = [
                paddle.to_tensor(pp.shape, 'float32')[[3, 2, 3, 2]]
                for pp in inputs[:self.nl]
            ]

        # Losses
        lcls, lbox = paddle.zeros([1]), paddle.zeros([1])
        lobj = paddle.zeros([1])  # single class will always be tensor([0.])
        for i in range(self.nl):
            pi = inputs[i]
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]
            tobj = paddle.zeros_like(pi[..., 0])
            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # numpy index
                if len(ps.shape) == 1:  # Note: when only one sample
                    ps = ps.unsqueeze(0)

                # Regression
                tensor_grid = paddle.to_tensor(np.stack([gi, gj], 1), 'float32')
                tensor_anch = paddle.to_tensor(anchors[i], 'float32')
                tensor_box = paddle.to_tensor(targets[i][:, 2:6], 'float32')
                pxy = F.sigmoid(ps[:, :2]) * 2. - 0.5
                pwh = (F.sigmoid(ps[:, 2:4]) * 2)**2 * tensor_anch
                pbox = paddle.concat([pxy, pwh], 1)  # predicted box
                selected_tbox = tensor_box * pre_gen_gains[i]
                selected_tbox[:, :2] -= tensor_grid
                iou = bbox_iou(
                    pbox.T,
                    selected_tbox.T,
                    x1y1x2y2=False,
                    ciou=True,
                    eps=1e-7)
                lbox += (1.0 - iou).mean()

                # Objectness
                score_iou = paddle.cast(iou.detach().clip(0), tobj.dtype)
                with paddle.no_grad():
                    # numpy index
                    tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou

                # Classification
                selected_tcls = targets[i][:, 1].astype(np.int64)
                if self.num_classes > 1:  # cls loss (only if multiple classes)
                    t = paddle.full_like(ps[:, 5:], self.cls_neg_label)
                    t[range(n), selected_tcls] = self.cls_pos_label
                    lcls += self.BCEcls(ps[:, 5:], t)

            if self.use_aux:
                pi_aux = inputs[i + self.nl]
                b_aux, a_aux, gj_aux, gi_aux = bs_aux[i], as_aux_[i], gjs_aux[
                    i], gis_aux[i]
                tobj_aux = paddle.zeros_like(pi_aux[..., 0])

                n_aux = b_aux.shape[0]  # number of targets
                if n_aux:
                    ps_aux = pi_aux[b_aux, a_aux, gj_aux, gi_aux]  # numpy index
                    if len(ps_aux.shape) == 1:  # Note: when only one sample
                        ps_aux = ps_aux.unsqueeze(0)

                    # Regression
                    tensor_grid_aux = paddle.to_tensor(
                        np.stack([gi_aux, gj_aux], 1), 'float32')
                    tensor_anch_aux = paddle.to_tensor(anchors_aux[i],
                                                       'float32')
                    tensor_box_aux = paddle.to_tensor(targets_aux[i][:, 2:6],
                                                      'float32')
                    pxy_aux = F.sigmoid(ps_aux[:, :2]) * 2. - 0.5
                    pwh_aux = (F.sigmoid(ps_aux[:, 2:4]) *
                               2)**2 * tensor_anch_aux
                    pbox_aux = paddle.concat((pxy_aux, pwh_aux), 1)
                    selected_tbox_aux = tensor_box_aux * pre_gen_gains_aux[i]
                    selected_tbox_aux[:, :2] -= tensor_grid_aux
                    iou_aux = bbox_iou(
                        pbox_aux.T,
                        selected_tbox_aux.T,
                        x1y1x2y2=False,
                        ciou=True)
                    lbox += 0.25 * (1.0 - iou_aux).mean()

                    # Objectness
                    score_iou_aux = paddle.cast(iou_aux.detach().clip(0),
                                                tobj_aux.dtype)
                    with paddle.no_grad():
                        tobj_aux[b_aux, a_aux, gj_aux, gi_aux] = (
                            1.0 - self.gr) + self.gr * score_iou_aux

                    # Classification
                    selected_tcls_aux = targets_aux[i][:, 1].astype(np.int64)
                    if self.num_classes > 1:  # cls loss (only if multiple classes)
                        t_aux = paddle.full_like(ps_aux[:, 5:],
                                                 self.cls_neg_label)
                        t_aux[range(n_aux),
                              selected_tcls_aux] = self.cls_pos_label
                        lcls += 0.25 * self.BCEcls(ps_aux[:, 5:], t_aux)

            obji = self.BCEobj(pi[:, :, :, :, 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.use_aux:
                obji_aux = self.BCEobj(pi_aux[:, :, :, :, 4], tobj_aux)
                lobj += 0.25 * obji_aux * self.balance[i]  # obj_aux loss

        batch_size = head_outs[0].shape[0]
        num_gpus = gt_targets.get('num_gpus', 8)
        yolo_losses = dict()
        yolo_losses['loss_box'] = lbox * self.loss_weights['box']* batch_size * num_gpus
        yolo_losses['loss_cls'] = lcls * self.loss_weights['cls']* batch_size * num_gpus
        yolo_losses['loss_obj'] = lobj * self.loss_weights['obj']* batch_size * num_gpus
        loss_all = yolo_losses['loss_box'] + yolo_losses[
            'loss_obj'] + yolo_losses['loss_cls']
        yolo_losses['loss'] = loss_all
        return yolo_losses

    def build_targets(self, p, targets, anchors, batch_images):
        indices, anch = self.find_3_positive(p, targets, anchors)
        # numpy indices,anch for fast assign

        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]

        nl = len(p)
        for batch_idx in range(p[0].shape[0]):
            b_idx = targets[:, 0] == batch_idx
            if b_idx.sum() == 0:
                continue
            this_target = targets[b_idx]
            txywh = this_target[:, 2:6] * batch_images[batch_idx].shape[1]
            # this_target[:, 2:6] * 640
            txyxy = xywh2xyxy(paddle.to_tensor(txywh, 'float32'))  # tensor op

            pxyxys, p_cls, p_obj = [], [], []
            from_which_layer = []
            all_b, all_a, all_gj, all_gi = [], [], [], []
            all_anch = []

            empty_feats_num = 0
            for i, pi in enumerate(p):
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                if idx.sum() == 0:
                    empty_feats_num += 1
                    continue
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(np.ones([len(b)]) * i)

                fg_pred = pi[b, a, gj, gi]  # numpy index
                if len(fg_pred.shape) == 1:  # Note: when only one sample
                    fg_pred = fg_pred.unsqueeze(0)
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])

                tensor_grid = paddle.to_tensor(np.stack([gi, gj], 1), 'float32')
                pxy = (F.sigmoid(fg_pred[:, :2]) * 2. - 0.5 + tensor_grid
                       ) * self.downsample_ratios[i]
                tensor_anch = paddle.to_tensor(anch[i][idx], 'float32')
                pwh = (F.sigmoid(fg_pred[:, 2:4]) *
                       2)**2 * tensor_anch * self.downsample_ratios[i]
                pxywh = paddle.concat([pxy, pwh], -1)
                pxyxy = xywh2xyxy(pxywh)  # tensor op
                pxyxys.append(pxyxy)

            if empty_feats_num == len(p) or len(pxyxys) == 0:  # Note: empty
                continue
            pxyxys = paddle.concat(pxyxys, 0)

            p_obj = paddle.concat(p_obj, 0)
            p_cls = paddle.concat(p_cls, 0)

            from_which_layer = np.concatenate(from_which_layer, 0)
            all_b = np.concatenate(all_b, 0)
            all_a = np.concatenate(all_a, 0)
            all_gj = np.concatenate(all_gj, 0)
            all_gi = np.concatenate(all_gi, 0)
            all_anch = np.concatenate(all_anch, 0)

            pairwise_ious = box_iou(txyxy, pxyxys)  # tensor op
            # [N, 4] [M, 4] to get [N, M] ious

            pairwise_iou_loss = -paddle.log(pairwise_ious + 1e-8)

            min_topk = 10
            topk_ious, _ = paddle.topk(pairwise_ious,
                                       min(min_topk, pairwise_ious.shape[1]), 1)
            dynamic_ks = paddle.clip(topk_ious.sum(1).cast('int'), min=1)

            gt_cls_per_image = (paddle.tile(
                F.one_hot(
                    paddle.to_tensor(this_target[:, 1], 'int64'),
                    self.num_classes).unsqueeze(1), [1, pxyxys.shape[0], 1]))

            num_gt = this_target.shape[0]
            cls_preds_ = (
                F.sigmoid(paddle.tile(p_cls.unsqueeze(0), [num_gt, 1, 1])) *
                F.sigmoid(paddle.tile(p_obj.unsqueeze(0), [num_gt, 1, 1])))

            y = cls_preds_.sqrt_()
            pairwise_cls_loss = F.binary_cross_entropy_with_logits(
                paddle.log(y / (1 - y) + 1e-8),
                gt_cls_per_image,
                reduction="none").sum(-1)
            del cls_preds_

            cost = (pairwise_cls_loss + 3.0 * pairwise_iou_loss)

            matching_matrix = np.zeros(cost.shape)
            for gt_idx in range(num_gt):
                _, pos_idx = paddle.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
                matching_matrix[gt_idx, pos_idx.numpy()] = 1.0
            del topk_ious, dynamic_ks, pos_idx

            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                cost_argmin = np.argmin(cost.numpy()[:, anchor_matching_gt > 1],
                                        0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]

            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(
                    this_target[layer_idx])  # this_ not all_
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = np.concatenate(matching_bs[i], 0)
                matching_as[i] = np.concatenate(matching_as[i], 0)
                matching_gjs[i] = np.concatenate(matching_gjs[i], 0)
                matching_gis[i] = np.concatenate(matching_gis[i], 0)
                matching_targets[i] = np.concatenate(matching_targets[i], 0)
                matching_anchs[i] = np.concatenate(matching_anchs[i], 0)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs

    def find_3_positive(self, outputs, targets, all_anchors):
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = np.ones(7, dtype=np.float32)  # normalized to gridspace gain
        ai = np.tile(np.arange(na, dtype=np.float32).reshape(na, 1), [1, nt])
        targets_labels = np.concatenate((np.tile(
            np.expand_dims(targets, 0), [na, 1, 1]), ai[:, :, None]), 2)
        g = self.bias  # 0.5

        for i in range(len(all_anchors)):
            anchors = np.array(all_anchors[i]) / self.downsample_ratios[i]
            gain[2:6] = np.array(
                outputs[i].shape, dtype=np.float32)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets_labels to anchors
            t = targets_labels * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = np.maximum(r, 1. / r).max(2) < self.anchor_t
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = np.stack([np.ones_like(j), j, k, l, m])
                t = np.tile(t, [5, 1, 1])[j]
                offsets = (np.zeros_like(gxy)[None] + self.off[:, None])[j]
            else:
                t = targets_labels[0]
                offsets = 0

            # Define
            b, c = t[:, :2].astype(np.int64).T
            gxy = t[:, 2:4]  # grid xy
            gij = (gxy - offsets).astype(np.int64)
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].astype(np.int64)  # anchor indices
            gj, gi = gj.clip(0, gain[3] - 1).astype(np.int64), gi.clip(
                0, gain[2] - 1).astype(np.int64)
            indices.append((b, a, gj, gi))
            anch.append(anchors[a])  # anchors
        # return numpy rather than tensor
        return indices, anch

    def build_targets2(self, p, targets, anchors, batch_images):
        indices, anch = self.find_5_positive(p, targets, anchors)
        # numpy indices,anch for fast assign

        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]

        nl = len(p)
        for batch_idx in range(p[0].shape[0]):
            b_idx = targets[:, 0] == batch_idx
            if b_idx.sum() == 0:
                continue
            this_target = targets[b_idx]
            txywh = this_target[:, 2:6] * batch_images[batch_idx].shape[1]
            # this_target[:, 2:6] * 1280
            txyxy = xywh2xyxy(paddle.to_tensor(txywh, 'float32'))  # tensor op

            pxyxys, p_cls, p_obj = [], [], []
            from_which_layer = []
            all_b, all_a, all_gj, all_gi = [], [], [], []
            all_anch = []

            empty_feats_num = 0
            for i, pi in enumerate(p):
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                if idx.sum() == 0:
                    empty_feats_num += 1
                    continue
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(np.ones([len(b)]) * i)

                fg_pred = pi[b, a, gj, gi]  # numpy index
                if len(fg_pred.shape) == 1:  # Note: when only one sample
                    fg_pred = fg_pred.unsqueeze(0)
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])

                tensor_grid = paddle.to_tensor(np.stack([gi, gj], 1), 'float32')
                pxy = (F.sigmoid(fg_pred[:, :2]) * 2. - 0.5 + tensor_grid
                       ) * self.downsample_ratios[i]
                tensor_anch = paddle.to_tensor(anch[i][idx], 'float32')
                pwh = (F.sigmoid(fg_pred[:, 2:4]) *
                       2)**2 * tensor_anch * self.downsample_ratios[i]
                pxywh = paddle.concat([pxy, pwh], -1)
                pxyxy = xywh2xyxy(pxywh)  # tensor op
                pxyxys.append(pxyxy)

            if empty_feats_num == len(p) or len(pxyxys) == 0:  # Note: empty
                continue
            pxyxys = paddle.concat(pxyxys, 0)

            p_obj = paddle.concat(p_obj, 0)
            p_cls = paddle.concat(p_cls, 0)

            from_which_layer = np.concatenate(from_which_layer, 0)
            all_b = np.concatenate(all_b, 0)
            all_a = np.concatenate(all_a, 0)
            all_gj = np.concatenate(all_gj, 0)
            all_gi = np.concatenate(all_gi, 0)
            all_anch = np.concatenate(all_anch, 0)

            pairwise_ious = box_iou(txyxy, pxyxys)  # tensor op
            # [N, 4] [M, 4] to get [N, M] ious

            pairwise_iou_loss = -paddle.log(pairwise_ious + 1e-8)

            min_topk = 20  # diff, 10 in build_targets()
            topk_ious, _ = paddle.topk(pairwise_ious,
                                       min(min_topk, pairwise_ious.shape[1]), 1)
            dynamic_ks = paddle.clip(topk_ious.sum(1).cast('int'), min=1)

            gt_cls_per_image = (paddle.tile(
                F.one_hot(
                    paddle.to_tensor(this_target[:, 1], 'int64'),
                    self.num_classes).unsqueeze(1), [1, pxyxys.shape[0], 1]))

            num_gt = this_target.shape[0]
            cls_preds_ = (
                F.sigmoid(paddle.tile(p_cls.unsqueeze(0), [num_gt, 1, 1])) *
                F.sigmoid(paddle.tile(p_obj.unsqueeze(0), [num_gt, 1, 1])))

            y = cls_preds_.sqrt_()
            pairwise_cls_loss = F.binary_cross_entropy_with_logits(
                paddle.log(y / (1 - y) + 1e-8),
                gt_cls_per_image,
                reduction="none").sum(-1)
            del cls_preds_

            cost = (pairwise_cls_loss + 3.0 * pairwise_iou_loss)

            matching_matrix = np.zeros(cost.shape)
            for gt_idx in range(num_gt):
                _, pos_idx = paddle.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
                matching_matrix[gt_idx, pos_idx.numpy()] = 1.0
            del topk_ious, dynamic_ks, pos_idx

            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                cost_argmin = np.argmin(cost.numpy()[:, anchor_matching_gt > 1],
                                        0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]

            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(
                    this_target[layer_idx])  # this_ not all_
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = np.concatenate(matching_bs[i], 0)
                matching_as[i] = np.concatenate(matching_as[i], 0)
                matching_gjs[i] = np.concatenate(matching_gjs[i], 0)
                matching_gis[i] = np.concatenate(matching_gis[i], 0)
                matching_targets[i] = np.concatenate(matching_targets[i], 0)
                matching_anchs[i] = np.concatenate(matching_anchs[i], 0)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs

    def find_5_positive(self, outputs, targets, all_anchors):
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = np.ones(7, dtype=np.float32)  # normalized to gridspace gain
        ai = np.tile(np.arange(na, dtype=np.float32).reshape(na, 1), [1, nt])
        targets_labels = np.concatenate((np.tile(
            np.expand_dims(targets, 0), [na, 1, 1]), ai[:, :, None]), 2)
        g = 1.0  # Note: diff, not self.bias(0.5) in find_3_positive()

        for i in range(len(all_anchors)):
            anchors = np.array(all_anchors[i]) / self.downsample_ratios[i]
            gain[2:6] = np.array(
                outputs[i].shape, dtype=np.float32)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets_labels to anchors
            t = targets_labels * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = np.maximum(r, 1. / r).max(2) < self.anchor_t
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = np.stack([np.ones_like(j), j, k, l, m])
                t = np.tile(t, [5, 1, 1])[j]
                offsets = (np.zeros_like(gxy)[None] + self.off[:, None])[j]
            else:
                t = targets_labels[0]
                offsets = 0

            # Define
            b, c = t[:, :2].astype(np.int64).T
            gxy = t[:, 2:4]  # grid xy
            gij = (gxy - offsets).astype(np.int64)
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].astype(np.int64)  # anchor indices
            gj, gi = gj.clip(0, gain[3] - 1).astype(np.int64), gi.clip(
                0, gain[2] - 1).astype(np.int64)
            indices.append((b, a, gj, gi))
            anch.append(anchors[a])  # anchors
        # return numpy rather than tensor
        return indices, anch


def xywh2xyxy(x):
    """
    [x, y, w, h] to [x1, y1, x2, y2], paddle Tensor op
    """
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def box_iou(box1, box2):
    """
    [N, 4] [M, 4] to get [N, M] ious, boxes in [x1, y1, x2, y2] format. paddle Tensor op
     """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (paddle.minimum(box1[:, None, 2:], box2[:, 2:]) - paddle.maximum(
        box1[:, None, :2], box2[:, :2])).clip(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)
