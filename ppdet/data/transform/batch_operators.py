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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import typing

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import cv2
import math
import numpy as np
from .operators import register_op, BaseOperator, Resize
from .op_helper import jaccard_overlap

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = [
    'PadBatch',
    'BatchRandomResize',
    'Gt2YoloTarget',
    'PadGT',
    'PadMaskBatch',
]


@register_op
class PadBatch(BaseOperator):
    """
    Pad a batch of samples so they can be divisible by a stride.
    The layout of each image should be 'CHW'.
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0):
        super(PadBatch, self).__init__()
        self.pad_to_stride = pad_to_stride

    def __call__(self, samples, context=None):
        """
        Args:
            samples (list): a batch of sample, each is dict.
        """
        coarsest_stride = self.pad_to_stride

        # multi scale input is nested list
        if isinstance(samples,
                      typing.Sequence) and len(samples) > 0 and isinstance(
                          samples[0], typing.Sequence):
            inner_samples = samples[0]
        else:
            inner_samples = samples

        max_shape = np.array(
            [data['image'].shape for data in inner_samples]).max(axis=0)
        if coarsest_stride > 0:
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        for data in inner_samples:
            im = data['image']
            im_c, im_h, im_w = im.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im
            data['image'] = padding_im
            if 'semantic' in data and data['semantic'] is not None:
                semantic = data['semantic']
                padding_sem = np.zeros(
                    (1, max_shape[1], max_shape[2]), dtype=np.float32)
                padding_sem[:, :im_h, :im_w] = semantic
                data['semantic'] = padding_sem
            if 'gt_segm' in data and data['gt_segm'] is not None:
                gt_segm = data['gt_segm']
                padding_segm = np.zeros(
                    (gt_segm.shape[0], max_shape[1], max_shape[2]),
                    dtype=np.uint8)
                padding_segm[:, :im_h, :im_w] = gt_segm
                data['gt_segm'] = padding_segm

        return samples


@register_op
class BatchRandomResize(BaseOperator):
    """
    Resize image to target size randomly. random target_size and interpolation method
    Args:
        target_size (int, list, tuple): image target size, if random size is True, must be list or tuple
        keep_ratio (bool): whether keep_raio or not, default true
        interp (int): the interpolation method
        random_size (bool): whether random select target size of image
        random_interp (bool): whether random select interpolation method
    """

    def __init__(self,
                 target_size,
                 keep_ratio,
                 interp=cv2.INTER_NEAREST,
                 random_size=True,
                 random_interp=False):
        super(BatchRandomResize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        self.interp = interp
        assert isinstance(target_size, (
            int, Sequence)), "target_size must be int, list or tuple"
        if random_size and not isinstance(target_size, list):
            raise TypeError(
                "Type of target_size is invalid when random_size is True. Must be List, now is {}".
                format(type(target_size)))
        self.target_size = target_size
        self.random_size = random_size
        self.random_interp = random_interp

    def __call__(self, samples, context=None):
        if self.random_size:
            index = np.random.choice(len(self.target_size))
            target_size = self.target_size[index]
        else:
            target_size = self.target_size

        if self.random_interp:
            interp = np.random.choice(self.interps)
        else:
            interp = self.interp

        resizer = Resize(target_size, keep_ratio=self.keep_ratio, interp=interp)
        return resizer(samples, context=context)


@register_op
class Gt2YoloTarget(BaseOperator):
    __shared__ = ['num_classes']
    """
    Generate YOLOv3 targets by groud truth data, this operator is only used in
    fine grained YOLOv3 loss mode
    """

    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 num_classes=80,
                 iou_thresh=1.):
        super(Gt2YoloTarget, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh

    def __call__(self, samples, context=None):
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."

        h, w = samples[0]['image'].shape[1:3]
        an_hw = np.array(self.anchors) / np.array([[w, h]])
        for sample in samples:
            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']
            if 'gt_score' not in sample:
                sample['gt_score'] = np.ones(
                    (gt_bbox.shape[0], 1), dtype=np.float32)
            gt_score = sample['gt_score']
            for i, (
                    mask, downsample_ratio
            ) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
                grid_h = int(h / downsample_ratio)
                grid_w = int(w / downsample_ratio)
                target = np.zeros(
                    (len(mask), 6 + self.num_classes, grid_h, grid_w),
                    dtype=np.float32)
                for b in range(gt_bbox.shape[0]):
                    gx, gy, gw, gh = gt_bbox[b, :]
                    cls = gt_class[b]
                    score = gt_score[b]
                    if gw <= 0. or gh <= 0. or score <= 0.:
                        continue

                    # find best match anchor index
                    best_iou = 0.
                    best_idx = -1
                    for an_idx in range(an_hw.shape[0]):
                        iou = jaccard_overlap(
                            [0., 0., gw, gh],
                            [0., 0., an_hw[an_idx, 0], an_hw[an_idx, 1]])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = an_idx

                    gi = int(gx * grid_w)
                    gj = int(gy * grid_h)

                    # gtbox should be regresed in this layes if best match 
                    # anchor index in anchor mask of this layer
                    if best_idx in mask:
                        best_n = mask.index(best_idx)

                        # x, y, w, h, scale
                        target[best_n, 0, gj, gi] = gx * grid_w - gi
                        target[best_n, 1, gj, gi] = gy * grid_h - gj
                        target[best_n, 2, gj, gi] = np.log(
                            gw * w / self.anchors[best_idx][0])
                        target[best_n, 3, gj, gi] = np.log(
                            gh * h / self.anchors[best_idx][1])
                        target[best_n, 4, gj, gi] = 2.0 - gw * gh

                        # objectness record gt_score
                        target[best_n, 5, gj, gi] = score

                        # classification
                        target[best_n, 6 + cls, gj, gi] = 1.

                    # For non-matched anchors, calculate the target if the iou 
                    # between anchor and gt is larger than iou_thresh
                    if self.iou_thresh < 1:
                        for idx, mask_i in enumerate(mask):
                            if mask_i == best_idx: continue
                            iou = jaccard_overlap(
                                [0., 0., gw, gh],
                                [0., 0., an_hw[mask_i, 0], an_hw[mask_i, 1]])
                            if iou > self.iou_thresh and target[idx, 5, gj,
                                                                gi] == 0.:
                                # x, y, w, h, scale
                                target[idx, 0, gj, gi] = gx * grid_w - gi
                                target[idx, 1, gj, gi] = gy * grid_h - gj
                                target[idx, 2, gj, gi] = np.log(
                                    gw * w / self.anchors[mask_i][0])
                                target[idx, 3, gj, gi] = np.log(
                                    gh * h / self.anchors[mask_i][1])
                                target[idx, 4, gj, gi] = 2.0 - gw * gh

                                # objectness record gt_score
                                target[idx, 5, gj, gi] = score

                                # classification
                                target[idx, 6 + cls, gj, gi] = 1.
                sample['target{}'.format(i)] = target

            # remove useless gt_class and gt_score after target calculated
            sample.pop('gt_class')
            sample.pop('gt_score')

        return samples


@register_op
class PadGT(BaseOperator):
    """
    Pad 0 to `gt_class`, `gt_bbox`, `gt_score`...
    The num_max_boxes is the largest for batch.
    Args:
        return_gt_mask (bool): If true, return `pad_gt_mask`,
                                1 means bbox, 0 means no bbox.
    """

    def __init__(self, return_gt_mask=True, pad_img=False, minimum_gtnum=0):
        super(PadGT, self).__init__()
        self.return_gt_mask = return_gt_mask
        self.pad_img = pad_img
        self.minimum_gtnum = minimum_gtnum

    def _impad(self,
               img: np.ndarray,
               *,
               shape=None,
               padding=None,
               pad_val=0,
               padding_mode='constant') -> np.ndarray:
        """Pad the given image to a certain shape or pad on all sides with
        specified padding mode and padding value.

        Args:
            img (ndarray): Image to be padded.
            shape (tuple[int]): Expected padding shape (h, w). Default: None.
            padding (int or tuple[int]): Padding on each border. If a single int is
                provided this is used to pad all borders. If tuple of length 2 is
                provided this is the padding on left/right and top/bottom
                respectively. If a tuple of length 4 is provided this is the
                padding for the left, top, right and bottom borders respectively.
                Default: None. Note that `shape` and `padding` can not be both
                set.
            pad_val (Number | Sequence[Number]): Values to be filled in padding
                areas when padding_mode is 'constant'. Default: 0.
            padding_mode (str): Type of padding. Should be: constant, edge,
                reflect or symmetric. Default: constant.
                - constant: pads with a constant value, this value is specified
                with pad_val.
                - edge: pads with the last value at the edge of the image.
                - reflect: pads with reflection of image without repeating the last
                value on the edge. For example, padding [1, 2, 3, 4] with 2
                elements on both sides in reflect mode will result in
                [3, 2, 1, 2, 3, 4, 3, 2].
                - symmetric: pads with reflection of image repeating the last value
                on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
                both sides in symmetric mode will result in
                [2, 1, 1, 2, 3, 4, 4, 3]

        Returns:
            ndarray: The padded image.
        """

        assert (shape is not None) ^ (padding is not None)
        if shape is not None:
            width = max(shape[1] - img.shape[1], 0)
            height = max(shape[0] - img.shape[0], 0)
            padding = (0, 0, int(width), int(height))

        # check pad_val
        import numbers
        if isinstance(pad_val, tuple):
            assert len(pad_val) == img.shape[-1]
        elif not isinstance(pad_val, numbers.Number):
            raise TypeError('pad_val must be a int or a tuple. '
                            f'But received {type(pad_val)}')

        # check padding
        if isinstance(padding, tuple) and len(padding) in [2, 4]:
            if len(padding) == 2:
                padding = (padding[0], padding[1], padding[0], padding[1])
        elif isinstance(padding, numbers.Number):
            padding = (padding, padding, padding, padding)
        else:
            raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                             f'But received {padding}')

        # check padding mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        border_type = {
            'constant': cv2.BORDER_CONSTANT,
            'edge': cv2.BORDER_REPLICATE,
            'reflect': cv2.BORDER_REFLECT_101,
            'symmetric': cv2.BORDER_REFLECT
        }
        img = cv2.copyMakeBorder(
            img,
            padding[1],
            padding[3],
            padding[0],
            padding[2],
            border_type[padding_mode],
            value=pad_val)

        return img

    def checkmaxshape(self, samples):
        maxh, maxw = 0, 0
        for sample in samples:
            h, w = sample['im_shape']
            if h > maxh:
                maxh = h
            if w > maxw:
                maxw = w
        return (maxh, maxw)

    def __call__(self, samples, context=None):
        num_max_boxes = max([len(s['gt_bbox']) for s in samples])
        num_max_boxes = max(self.minimum_gtnum, num_max_boxes)
        if self.pad_img:
            maxshape = self.checkmaxshape(samples)
        for sample in samples:
            if self.pad_img:
                img = sample['image']
                padimg = self._impad(img, shape=maxshape)
                sample['image'] = padimg
            if self.return_gt_mask:
                sample['pad_gt_mask'] = np.zeros(
                    (num_max_boxes, 1), dtype=np.float32)
            if num_max_boxes == 0:
                continue

            num_gt = len(sample['gt_bbox'])
            pad_gt_class = np.zeros((num_max_boxes, 1), dtype=np.int32)
            pad_gt_bbox = np.zeros((num_max_boxes, 4), dtype=np.float32)
            if num_gt > 0:
                pad_gt_class[:num_gt] = sample['gt_class']
                pad_gt_bbox[:num_gt] = sample['gt_bbox']
            sample['gt_class'] = pad_gt_class
            sample['gt_bbox'] = pad_gt_bbox
            # pad_gt_mask
            if 'pad_gt_mask' in sample:
                sample['pad_gt_mask'][:num_gt] = 1
            # gt_score
            if 'gt_score' in sample:
                pad_gt_score = np.zeros((num_max_boxes, 1), dtype=np.float32)
                if num_gt > 0:
                    pad_gt_score[:num_gt] = sample['gt_score']
                sample['gt_score'] = pad_gt_score
            if 'is_crowd' in sample:
                pad_is_crowd = np.zeros((num_max_boxes, 1), dtype=np.int32)
                if num_gt > 0:
                    pad_is_crowd[:num_gt] = sample['is_crowd']
                sample['is_crowd'] = pad_is_crowd
            if 'difficult' in sample:
                pad_diff = np.zeros((num_max_boxes, 1), dtype=np.int32)
                if num_gt > 0:
                    pad_diff[:num_gt] = sample['difficult']
                sample['difficult'] = pad_diff
            if 'gt_joints' in sample:
                num_joints = sample['gt_joints'].shape[1]
                pad_gt_joints = np.zeros(
                    (num_max_boxes, num_joints, 3), dtype=np.float32)
                if num_gt > 0:
                    pad_gt_joints[:num_gt] = sample['gt_joints']
                sample['gt_joints'] = pad_gt_joints
            if 'gt_areas' in sample:
                pad_gt_areas = np.zeros((num_max_boxes, 1), dtype=np.float32)
                if num_gt > 0:
                    pad_gt_areas[:num_gt, 0] = sample['gt_areas']
                sample['gt_areas'] = pad_gt_areas
            if 'gt_segm' in sample:
                mask_h, mask_w = sample['gt_segm'].shape[-2:]
                pad_gt_segm = np.zeros(
                    (num_max_boxes, mask_h, mask_w), dtype=np.float32)
                if num_gt > 0:
                    pad_gt_segm[:num_gt] = sample['gt_segm']
                sample['gt_segm'] = pad_gt_segm
        return samples


@register_op
class PadMaskBatch(BaseOperator):
    """
    Pad a batch of samples so that they can be divisible by a stride.
    The layout of each image should be 'CHW'.
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
        return_pad_mask (bool): If `return_pad_mask = True`, return
            `pad_mask` for transformer.
    """

    def __init__(self, pad_to_stride=0, return_pad_mask=True):
        super(PadMaskBatch, self).__init__()
        self.pad_to_stride = pad_to_stride
        self.return_pad_mask = return_pad_mask

    def __call__(self, samples, context=None):
        """
        Args:
            samples (list): a batch of sample, each is dict.
        """
        coarsest_stride = self.pad_to_stride

        max_shape = np.array([data['image'].shape for data in samples]).max(
            axis=0)
        if coarsest_stride > 0:
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        for data in samples:
            im = data['image']
            im_c, im_h, im_w = im.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im.astype(np.float32)
            data['image'] = padding_im
            if 'semantic' in data and data['semantic'] is not None:
                semantic = data['semantic']
                padding_sem = np.zeros(
                    (1, max_shape[1], max_shape[2]), dtype=np.float32)
                padding_sem[:, :im_h, :im_w] = semantic
                data['semantic'] = padding_sem
            if 'gt_segm' in data and data['gt_segm'] is not None:
                gt_segm = data['gt_segm']
                padding_segm = np.zeros(
                    (gt_segm.shape[0], max_shape[1], max_shape[2]),
                    dtype=np.uint8)
                padding_segm[:, :im_h, :im_w] = gt_segm
                data['gt_segm'] = padding_segm
            if self.return_pad_mask:
                padding_mask = np.zeros(
                    (max_shape[1], max_shape[2]), dtype=np.float32)
                padding_mask[:im_h, :im_w] = 1.
                data['pad_mask'] = padding_mask

        return samples
