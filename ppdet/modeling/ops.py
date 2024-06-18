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

import paddle
import paddle.nn.functional as F
import paddle.nn as nn
from paddle import ParamAttr
from paddle.regularizer import L2Decay
try:
    import paddle._legacy_C_ops as C_ops
except:
    import paddle._C_ops as C_ops

from paddle import in_dynamic_mode
from paddle.common_ops_import import Variable, LayerHelper, check_variable_and_dtype, check_type, check_dtype

__all__ = [
    'multiclass_nms', 'matrix_nms', 'batch_norm', 'mish', 'silu', 'swish',
    'identity'
]


def identity(x):
    return x


def mish(x):
    return F.mish(x) if hasattr(F, mish) else x * F.tanh(F.softplus(x))


def silu(x):
    return F.silu(x)


def swish(x):
    return x * F.sigmoid(x)


TRT_ACT_SPEC = {'swish': swish, 'silu': swish}

ACT_SPEC = {'mish': mish, 'silu': silu}


def get_act_fn(act=None, trt=False):
    assert act is None or isinstance(act, (
        str, dict)), 'name of activation should be str, dict or None'
    if not act:
        return identity

    if isinstance(act, dict):
        name = act['name']
        act.pop('name')
        kwargs = act
    else:
        name = act
        kwargs = dict()

    if trt and name in TRT_ACT_SPEC:
        fn = TRT_ACT_SPEC[name]
    elif name in ACT_SPEC:
        fn = ACT_SPEC[name]
    else:
        fn = getattr(F, name)

    return lambda x: fn(x, **kwargs)


def batch_norm(ch,
               norm_type='bn',
               norm_decay=0.,
               freeze_norm=False,
               initializer=None,
               data_format='NCHW'):

    norm_lr = 0. if freeze_norm else 1.
    weight_attr = ParamAttr(
        initializer=initializer,
        learning_rate=norm_lr,
        regularizer=L2Decay(norm_decay),
        trainable=False if freeze_norm else True)
    bias_attr = ParamAttr(
        learning_rate=norm_lr,
        regularizer=L2Decay(norm_decay),
        trainable=False if freeze_norm else True)

    if norm_type in ['sync_bn', 'bn']:
        norm_layer = nn.BatchNorm2D(
            ch,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format)

    norm_params = norm_layer.parameters()
    if freeze_norm:
        for param in norm_params:
            param.stop_gradient = True

    return norm_layer


@paddle.jit.not_to_static
def multiclass_nms(bboxes,
                   scores,
                   score_threshold,
                   nms_top_k,
                   keep_top_k,
                   nms_threshold=0.3,
                   normalized=True,
                   nms_eta=1.,
                   background_label=-1,
                   return_index=False,
                   return_rois_num=True,
                   rois_num=None,
                   name=None):
    """
    This operator is to do multi-class non maximum suppression (NMS) on
    boxes and scores.
    In the NMS step, this operator greedily selects a subset of detection bounding
    boxes that have high scores larger than score_threshold, if providing this
    threshold, then selects the largest nms_top_k confidences scores if nms_top_k
    is larger than -1. Then this operator pruns away boxes that have high IOU
    (intersection over union) overlap with already selected boxes by adaptive
    threshold NMS based on parameters of nms_threshold and nms_eta.
    Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
    per image if keep_top_k is larger than -1.
    Args:
        bboxes (Tensor): Two types of bboxes are supported:
                           1. (Tensor) A 3-D Tensor with shape
                           [N, M, 4 or 8 16 24 32] represents the
                           predicted locations of M bounding bboxes,
                           N is the batch size. Each bounding box has four
                           coordinate values and the layout is
                           [xmin, ymin, xmax, ymax], when box size equals to 4.
                           2. (LoDTensor) A 3-D Tensor with shape [M, C, 4]
                           M is the number of bounding boxes, C is the
                           class number
        scores (Tensor): Two types of scores are supported:
                           1. (Tensor) A 3-D Tensor with shape [N, C, M]
                           represents the predicted confidence predictions.
                           N is the batch size, C is the class number, M is
                           number of bounding boxes. For each category there
                           are total M scores which corresponding M bounding
                           boxes. Please note, M is equal to the 2nd dimension
                           of BBoxes.
                           2. (LoDTensor) A 2-D LoDTensor with shape [M, C].
                           M is the number of bbox, C is the class number.
                           In this case, input BBoxes should be the second
                           case with shape [M, C, 4].
        background_label (int): The index of background label, the background
                                label will be ignored. If set to -1, then all
                                categories will be considered. Default: 0
        score_threshold (float): Threshold to filter out bounding boxes with
                                 low confidence score. If not provided,
                                 consider all boxes.
        nms_top_k (int): Maximum number of detections to be kept according to
                         the confidences after the filtering detections based
                         on score_threshold.
        nms_threshold (float): The threshold to be used in NMS. Default: 0.3
        nms_eta (float): The threshold to be used in NMS. Default: 1.0
        keep_top_k (int): Number of total bboxes to be kept per image after NMS
                          step. -1 means keeping all bboxes after NMS step.
        normalized (bool): Whether detections are normalized. Default: True
        return_index(bool): Whether return selected index. Default: False
        rois_num(Tensor): 1-D Tensor contains the number of RoIs in each image. 
            The shape is [B] and data type is int32. B is the number of images.
            If it is not None then return a list of 1-D Tensor. Each element 
            is the output RoIs' number of each image on the corresponding level
            and the shape is [B]. None by default.
        name(str): Name of the multiclass nms op. Default: None.
    Returns:
        A tuple with two Variables: (Out, Index) if return_index is True,
        otherwise, a tuple with one Variable(Out) is returned.
        Out: A 2-D LoDTensor with shape [No, 6] represents the detections.
        Each row has 6 values: [label, confidence, xmin, ymin, xmax, ymax]
        or A 2-D LoDTensor with shape [No, 10] represents the detections.
        Each row has 10 values: [label, confidence, x1, y1, x2, y2, x3, y3,
        x4, y4]. No is the total number of detections.
        If all images have not detected results, all elements in LoD will be
        0, and output tensor is empty (None).
        Index: Only return when return_index is True. A 2-D LoDTensor with
        shape [No, 1] represents the selected index which type is Integer.
        The index is the absolute value cross batches. No is the same number
        as Out. If the index is used to gather other attribute such as age,
        one needs to reshape the input(N, M, 1) to (N * M, 1) as first, where
        N is the batch size and M is the number of boxes.
    Examples:
        .. code-block:: python

            import paddle
            from ppdet.modeling import ops
            boxes = paddle.static.data(name='bboxes', shape=[81, 4],
                                      dtype='float32', lod_level=1)
            scores = paddle.static.data(name='scores', shape=[81],
                                      dtype='float32', lod_level=1)
            out, index = ops.multiclass_nms(bboxes=boxes,
                                            scores=scores,
                                            background_label=0,
                                            score_threshold=0.5,
                                            nms_top_k=400,
                                            nms_threshold=0.3,
                                            keep_top_k=200,
                                            normalized=False,
                                            return_index=True)
    """
    helper = LayerHelper('multiclass_nms3', **locals())

    if in_dynamic_mode():
        attrs = ('background_label', background_label, 'score_threshold',
                 score_threshold, 'nms_top_k', nms_top_k, 'nms_threshold',
                 nms_threshold, 'keep_top_k', keep_top_k, 'nms_eta', nms_eta,
                 'normalized', normalized)
        output, index, nms_rois_num = C_ops.multiclass_nms3(bboxes, scores,
                                                            rois_num, *attrs)
        if not return_index:
            index = None
        return output, nms_rois_num, index

    else:
        output = helper.create_variable_for_type_inference(dtype=bboxes.dtype)
        index = helper.create_variable_for_type_inference(dtype='int32')

        inputs = {'BBoxes': bboxes, 'Scores': scores}
        outputs = {'Out': output, 'Index': index}

        if rois_num is not None:
            inputs['RoisNum'] = rois_num

        if return_rois_num:
            nms_rois_num = helper.create_variable_for_type_inference(
                dtype='int32')
            outputs['NmsRoisNum'] = nms_rois_num

        helper.append_op(
            type="multiclass_nms3",
            inputs=inputs,
            attrs={
                'background_label': background_label,
                'score_threshold': score_threshold,
                'nms_top_k': nms_top_k,
                'nms_threshold': nms_threshold,
                'keep_top_k': keep_top_k,
                'nms_eta': nms_eta,
                'normalized': normalized
            },
            outputs=outputs)
        output.stop_gradient = True
        index.stop_gradient = True
        if not return_index:
            index = None
        if not return_rois_num:
            nms_rois_num = None

        return output, nms_rois_num, index


@paddle.jit.not_to_static
def matrix_nms(bboxes,
               scores,
               score_threshold,
               post_threshold,
               nms_top_k,
               keep_top_k,
               use_gaussian=False,
               gaussian_sigma=2.,
               background_label=0,
               normalized=True,
               return_index=False,
               return_rois_num=True,
               name=None):
    """
    **Matrix NMS**
    This operator does matrix non maximum suppression (NMS).
    First selects a subset of candidate bounding boxes that have higher scores
    than score_threshold (if provided), then the top k candidate is selected if
    nms_top_k is larger than -1. Score of the remaining candidate are then
    decayed according to the Matrix NMS scheme.
    Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
    per image if keep_top_k is larger than -1.
    Args:
        bboxes (Tensor): A 3-D Tensor with shape [N, M, 4] represents the
                           predicted locations of M bounding bboxes,
                           N is the batch size. Each bounding box has four
                           coordinate values and the layout is
                           [xmin, ymin, xmax, ymax], when box size equals to 4.
                           The data type is float32 or float64.
        scores (Tensor): A 3-D Tensor with shape [N, C, M]
                           represents the predicted confidence predictions.
                           N is the batch size, C is the class number, M is
                           number of bounding boxes. For each category there
                           are total M scores which corresponding M bounding
                           boxes. Please note, M is equal to the 2nd dimension
                           of BBoxes. The data type is float32 or float64.
        score_threshold (float): Threshold to filter out bounding boxes with
                                 low confidence score.
        post_threshold (float): Threshold to filter out bounding boxes with
                                low confidence score AFTER decaying.
        nms_top_k (int): Maximum number of detections to be kept according to
                         the confidences after the filtering detections based
                         on score_threshold.
        keep_top_k (int): Number of total bboxes to be kept per image after NMS
                          step. -1 means keeping all bboxes after NMS step.
        use_gaussian (bool): Use Gaussian as the decay function. Default: False
        gaussian_sigma (float): Sigma for Gaussian decay function. Default: 2.0
        background_label (int): The index of background label, the background
                                label will be ignored. If set to -1, then all
                                categories will be considered. Default: 0
        normalized (bool): Whether detections are normalized. Default: True
        return_index(bool): Whether return selected index. Default: False
        return_rois_num(bool): whether return rois_num. Default: True
        name(str): Name of the matrix nms op. Default: None.
    Returns:
        A tuple with three Tensor: (Out, Index, RoisNum) if return_index is True,
        otherwise, a tuple with two Tensor (Out, RoisNum) is returned.
        Out (Tensor): A 2-D Tensor with shape [No, 6] containing the
             detection results.
             Each row has 6 values: [label, confidence, xmin, ymin, xmax, ymax]
             (After version 1.3, when no boxes detected, the lod is changed
             from {0} to {1})
        Index (Tensor): A 2-D Tensor with shape [No, 1] containing the
            selected indices, which are absolute values cross batches.
        rois_num (Tensor): A 1-D Tensor with shape [N] containing 
            the number of detected boxes in each image.
    Examples:
        .. code-block:: python
            import paddle
            from ppdet.modeling import ops
            boxes = paddle.static.data(name='bboxes', shape=[None,81, 4],
                                      dtype='float32', lod_level=1)
            scores = paddle.static.data(name='scores', shape=[None,81],
                                      dtype='float32', lod_level=1)
            out = ops.matrix_nms(bboxes=boxes, scores=scores, background_label=0,
                                 score_threshold=0.5, post_threshold=0.1,
                                 nms_top_k=400, keep_top_k=200, normalized=False)
    """
    check_variable_and_dtype(bboxes, 'BBoxes', ['float32', 'float64'],
                             'matrix_nms')
    check_variable_and_dtype(scores, 'Scores', ['float32', 'float64'],
                             'matrix_nms')
    check_type(score_threshold, 'score_threshold', float, 'matrix_nms')
    check_type(post_threshold, 'post_threshold', float, 'matrix_nms')
    check_type(nms_top_k, 'nums_top_k', int, 'matrix_nms')
    check_type(keep_top_k, 'keep_top_k', int, 'matrix_nms')
    check_type(normalized, 'normalized', bool, 'matrix_nms')
    check_type(use_gaussian, 'use_gaussian', bool, 'matrix_nms')
    check_type(gaussian_sigma, 'gaussian_sigma', float, 'matrix_nms')
    check_type(background_label, 'background_label', int, 'matrix_nms')

    if in_dynamic_mode():
        attrs = ('background_label', background_label, 'score_threshold',
                 score_threshold, 'post_threshold', post_threshold, 'nms_top_k',
                 nms_top_k, 'gaussian_sigma', gaussian_sigma, 'use_gaussian',
                 use_gaussian, 'keep_top_k', keep_top_k, 'normalized',
                 normalized)
        out, index, rois_num = C_ops.matrix_nms(bboxes, scores, *attrs)
        if not return_index:
            index = None
        if not return_rois_num:
            rois_num = None
        return out, rois_num, index
    else:
        helper = LayerHelper('matrix_nms', **locals())
        output = helper.create_variable_for_type_inference(dtype=bboxes.dtype)
        index = helper.create_variable_for_type_inference(dtype='int32')
        outputs = {'Out': output, 'Index': index}
        if return_rois_num:
            rois_num = helper.create_variable_for_type_inference(dtype='int32')
            outputs['RoisNum'] = rois_num

        helper.append_op(
            type="matrix_nms",
            inputs={'BBoxes': bboxes,
                    'Scores': scores},
            attrs={
                'background_label': background_label,
                'score_threshold': score_threshold,
                'post_threshold': post_threshold,
                'nms_top_k': nms_top_k,
                'gaussian_sigma': gaussian_sigma,
                'use_gaussian': use_gaussian,
                'keep_top_k': keep_top_k,
                'normalized': normalized
            },
            outputs=outputs)
        output.stop_gradient = True

        if not return_index:
            index = None
        if not return_rois_num:
            rois_num = None
        return output, rois_num, index


def sigmoid_cross_entropy_with_logits(input,
                                      label,
                                      ignore_index=-100,
                                      normalize=False):
    output = F.binary_cross_entropy_with_logits(input, label, reduction='none')
    mask_tensor = paddle.cast(label != ignore_index, 'float32')
    output = paddle.multiply(output, mask_tensor)
    if normalize:
        sum_valid_mask = paddle.sum(mask_tensor)
        output = output / sum_valid_mask
    return output


def smooth_l1(input, label, inside_weight=None, outside_weight=None,
              sigma=None):
    input_new = paddle.multiply(input, inside_weight)
    label_new = paddle.multiply(label, inside_weight)
    delta = 1 / (sigma * sigma)
    out = F.smooth_l1_loss(input_new, label_new, reduction='none', delta=delta)
    out = paddle.multiply(out, outside_weight)
    out = out / delta
    out = paddle.reshape(out, shape=[out.shape[0], -1])
    out = paddle.sum(out, axis=1)
    return out


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.shape[0:4]
    assert num_channels % groups == 0, 'num_channels should be divisible by groups'
    channels_per_group = num_channels // groups
    x = paddle.reshape(
        x=x, shape=[batch_size, groups, channels_per_group, height, width])
    x = paddle.transpose(x=x, perm=[0, 2, 1, 3, 4])
    x = paddle.reshape(x=x, shape=[batch_size, num_channels, height, width])
    return x


def get_static_shape(tensor):
    shape = tensor.shape
    return shape
