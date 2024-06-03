import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist

from paddle.autograd import PyLayer

class _AllReduce(PyLayer):
    @staticmethod
    def forward(ctx, input):
        input_clone = input.clone().detach()
        dist.all_reduce(input_clone, sync_op=True)
        return input_clone

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output)
        return grad_output


def differentiable_all_reduce(input):
    """
    Differentiable counterpart of `dist.all_reduce`.
    """
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() == 1
    ):
        return input
    return _AllReduce.apply(input)


class NaiveSyncBatchNorm(nn.layer.norm.BatchNorm2D):
    """
    NPU SyncBatchNorm.
    This is a slower but correct alternative to `nn.SyncBatchNorm`.

    Note:
        There isn't a single definition of Sync BatchNorm.

        When ``stats_mode==""``, this module computes overall statistics by using
        statistics of each worker with equal weight.  The result is true statistics
        of all samples (as if they are all on one worker) only when all workers
        have the same (N, H, W). This mode does not support inputs with zero batch size.

        When ``stats_mode=="N"``, this module computes overall statistics by weighting
        the statistics of each worker by their ``N``. The result is true statistics
        of all samples (as if they are all on one worker) only when all workers
        have the same (H, W). It is slower than ``stats_mode==""``.

        Even though the result of this module may not be the true statistics of all samples,
        it may still be reasonable because it might be preferrable to assign equal weights
        to all workers, regardless of their (H, W) dimension, instead of putting larger weight
        on larger images. From preliminary experiments, little difference is found between such
        a simplified implementation and an accurate computation of overall mean & variance.
    """

    def __init__(self, *args, stats_mode="", **kwargs):
        super().__init__(*args, **kwargs)
        assert stats_mode in ["", "N"]
        self._stats_mode = stats_mode

    def forward(self, input):
        if dist.get_world_size() == 1 or not self.training:
            return super().forward(input)

        B, C = input.shape[0], input.shape[1]

        input_dim = len(input.shape)
        
        if input_dim == 3:
            mean = paddle.mean(input, axis=[0, 2])
            meansqr = paddle.mean(input * input, axis=[0, 2])
        elif input_dim == 2:
            mean = paddle.mean(input, axis=[0])
            meansqr = paddle.mean(input * input, axis=[0])
        elif input_dim == 4:
            mean = paddle.mean(input, axis=[0, 2, 3])
            meansqr = paddle.mean(input * input, axis=[0, 2, 3])
        else:
            raise ValueError()

        if self._stats_mode == "":
            assert B > 0, 'SyncBatchNorm(stats_mode="") does not support zero batch size.'
            vec = paddle.concat([mean, meansqr], axis=0) * (1.0 / dist.get_world_size())
            vec = differentiable_all_reduce(vec)
            mean, meansqr = paddle.split(vec, [C, C])
            momentum = (1. - self._momentum)
        else:
            if B == 0:
                vec = paddle.zeros([2 * C + 1], dtype=mean.dtype)
                vec = vec + input.sum()  # make sure there is gradient w.r.t input
            else:
                vec = paddle.concat(
                    [
                        mean,
                        meansqr,
                        paddle.ones([1], dtype=mean.dtype),
                    ],
                    dim=0,
                )
            vec = differentiable_all_reduce(vec * B)

            total_batch = vec[-1].detach()
            momentum = total_batch.clip(max=1) * (1. - self._momentum)  # no update if total_batch is 0
            mean, meansqr, _ = paddle.split(vec / total_batch.clip(min=1), [C, C, -1])  # avoid div-by-zero

        var = meansqr - mean * mean     
        invstd = paddle.rsqrt(var + self._epsilon)
        scale = self.weight * invstd
        bias = self.bias - mean * scale

        if input_dim == 3:
            scale = scale.reshape([1, -1, 1])
            bias = bias.reshape([1, -1, 1])
        elif input_dim == 2:
            scale = scale.reshape([1, -1])
            bias = bias.reshape([1, -1])
        elif input_dim == 4:
            scale = scale.reshape([1, -1, 1, 1])
            bias = bias.reshape([1, -1, 1, 1])
        else:
            raise ValueError()

        with paddle.no_grad():
            paddle.assign(self._mean + (momentum * (mean.detach() - self._mean)), self._mean)
            paddle.assign(self._variance + (momentum * (var.detach() - self._variance)), self._variance)

        ret = input * scale + bias

        return ret

    @classmethod
    def convert_sync_batchnorm(cls, layer):
        """
        Helper function to convert :class: `paddle.nn.BatchNorm*d` layers in the model to :class: `paddle.nn.SyncBatchNorm` layers.

        Parameters:
            layer(paddle.nn.Layer): model containing one or more `BatchNorm*d` layers.

        Returns:
            The original model with converted SyncBatchNorm layers. If BatchNorm*d layer in the model, use SyncBatchNorm layer instead.

        Examples:
            .. code-block:: python

                import paddle
                import paddle.nn as nn

                model = nn.Sequential(nn.Conv2D(3, 5, 3), nn.BatchNorm2D(5))
                sync_model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        """
        layer_output = layer
        if isinstance(layer, nn.layer.norm._BatchNormBase):
            if (
                layer._weight_attr is not None
                and not isinstance(layer._weight_attr, bool)
                and layer._weight_attr.name is not None
            ):
                layer._weight_attr.name = layer._weight_attr.name + '_sync'
            if (
                layer._bias_attr is not None
                and not isinstance(layer._bias_attr, bool)
                and layer._bias_attr.name is not None
            ):
                layer._bias_attr.name = layer._bias_attr.name + '_sync'

            layer_output = NaiveSyncBatchNorm(
                layer._num_features,
                layer._momentum,
                layer._epsilon,
                layer._weight_attr,
                layer._bias_attr,
                layer._data_format,
                layer._name,
            )

            if (
                layer._weight_attr is not False
                and layer._bias_attr is not False
            ):
                with paddle.no_grad():
                    layer_output.weight = layer.weight
                    layer_output.bias = layer.bias
            layer_output._mean = layer._mean
            layer_output._variance = layer._variance

        for name, sublayer in layer.named_children():
            layer_output.add_sublayer(
                name, cls.convert_sync_batchnorm(sublayer)
            )
        del layer
        return layer_output
