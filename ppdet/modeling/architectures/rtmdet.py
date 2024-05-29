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

__all__ = ['RTMDet']


@register
class RTMDet(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['with_mask']

    def __init__(self,
                 backbone='CSPNeXt',
                 neck='CSPNeXtPAFPN',
                 head='RTMDetHead',
                 post_process='BBoxPostProcess',
                 with_mask=False,
                 for_mot=False):
        """
        RTMDet see https://arxiv.org/abs/

        Args:
            backbone (nn.Layer): backbone instance
            neck (nn.Layer): neck instance
            head (nn.Layer): head instance
            for_mot (bool): whether return other features for multi-object tracking
                models, default False in pure object detection models.
        """
        super(RTMDet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
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
        head = create(cfg['head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "head": head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        neck_feats = self.neck(body_feats, self.for_mot)

        if self.training:
            yolo_losses = self.head(neck_feats, self.inputs)
            return yolo_losses
        else:
            yolo_head_outs = self.head(neck_feats)
            post_outs = self.head.post_process(yolo_head_outs,
                                               self.inputs['im_shape'],
                                               self.inputs['scale_factor'])

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
                    # Note: RTMDet Ins models don't support exclude_post_process or exclude_nms
                return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
