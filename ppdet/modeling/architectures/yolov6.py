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

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['YOLOv6']


@register
class YOLOv6(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['with_mask']

    def __init__(self,
                 backbone='EfficientRep',
                 neck='RepBiFPAN',
                 yolo_head='EffiDeHead',
                 post_process='BBoxPostProcess',
                 with_mask=False,
                 for_mot=False):
        """
        YOLOv6(https://arxiv.org/abs/2209.02976, https://arxiv.org/abs/2301.05586)

        Args:
            backbone (nn.Layer): backbone instance
            neck (nn.Layer): neck instance
            yolo_head (nn.Layer): head instance
            for_mot (bool): whether return other features for multi-object tracking
                models, default False in pure object detection models.
        """
        super(YOLOv6, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.yolo_head = yolo_head
        self.post_process = post_process
        self.for_mot = for_mot
        self.with_mask = with_mask

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        # fpn
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        # head
        kwargs = {'input_shape': neck.out_shape}
        yolo_head = create(cfg['yolo_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "yolo_head": yolo_head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        neck_feats = self.neck(body_feats, self.for_mot)

        if self.training:
            yolo_losses = self.yolo_head(neck_feats, self.inputs)
            return yolo_losses
        else:
            yolo_head_outs = self.yolo_head(neck_feats)

            if not self.with_mask:
                post_outs = self.yolo_head.post_process(
                    yolo_head_outs,
                    im_shape=self.inputs['im_shape'],
                    scale_factor=self.inputs['scale_factor'],
                    infer_shape=self.inputs['image'].shape[2:])
            else:
                gt_labels = self.inputs.get('gt_class', None)
                gt_bboxes = self.inputs.get('gt_bbox', None)
                gt_masks = self.inputs.get('gt_segm', None)
                post_outs = self.yolo_head.post_process(
                    yolo_head_outs,
                    im_shape=self.inputs['im_shape'],
                    scale_factor=self.inputs['scale_factor'],
                    infer_shape=self.inputs['image'].shape[2:],
                    gt_labels=gt_labels,
                    gt_bboxes=gt_bboxes,
                    gt_masks=gt_masks)

            if not isinstance(post_outs, (tuple, list)):
                # if set exclude_post_process, concat([pred_bboxes, pred_scores]) not scaled to origin
                # export onnx as torch yolo models
                return post_outs
            else:
                if not self.with_mask:
                    # if set exclude_nms, [pred_bboxes, pred_scores] scaled to origin
                    bbox, bbox_num = post_outs  # default for end-to-end eval/infer
                    output = {'bbox': bbox, 'bbox_num': bbox_num}
                else:
                    bbox, bbox_num, mask = post_outs  # default for end-to-end eval/infer
                    output = {'bbox': bbox, 'bbox_num': bbox_num, 'mask': mask}
                    # Note: YOLOv6 Ins models don't support exclude_post_process or exclude_nms
                return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
