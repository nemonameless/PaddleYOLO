# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register, serializable
from .csp_darknet import BaseConv, get_activation, SPPFLayer
from .yolov8_csp_darknet import C2fLayer
from ..shape_spec import ShapeSpec

__all__ = ['YOLOv10CSPDarkNet']


class SCDownLayer(nn.Layer):
    """Spatial-channel decoupled downsampling layer, named SCDown in YOLOv10"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 act='silu'):
        super().__init__()
        self.conv1 = BaseConv(in_channels,
                              out_channels,
                              ksize=1,
                              stride=1,
                              groups=1,
                              bias=False,
                              act=act)
        self.conv2 = BaseConv(out_channels,
                              out_channels,
                              ksize=kernel_size,
                              stride=stride,
                              groups=out_channels,
                              bias=False,
                              act=nn.Identity())

    def forward(self, x):
        return self.conv2(self.conv1(x))


class RepVGGBlock(nn.Layer):
    """RepVGGBlock supports changing the kernel_size and groups"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 groups,
                 act='silu'):
        super().__init__()
        assert isinstance(kernel_size, (tuple, list))
        assert len(kernel_size) == 2
        assert max(kernel_size) % 2 == 1
        assert min(kernel_size) % 2 == 1
        assert in_channels % groups == 0
        assert out_channels % groups == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups

        self.conv1 = BaseConv(in_channels,
                              out_channels,
                              ksize=max(kernel_size),
                              stride=1,
                              groups=groups,
                              bias=False,
                              act=nn.Identity())
        self.conv2 = BaseConv(in_channels,
                              out_channels,
                              ksize=min(kernel_size),
                              stride=1,
                              groups=groups,
                              bias=False,
                              act=nn.Identity())
        self.act = get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            return self.act(self.conv(x))
        else:
            return self.act(self.conv1(x) + self.conv2(x))

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2D(in_channels=self.in_channels,
                                  out_channels=self.out_channels,
                                  kernel_size=max(self.kernel_size),
                                  stride=1,
                                  padding=(max(self.kernel_size) - 1) // 2,
                                  groups=self.groups)
        kernel, bias = self._get_equivalent_kernel_bias()
        self.conv.weight.set_value(kernel)
        self.conv.bias.set_value(bias)
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def _get_equivalent_kernel_bias(self):
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)
        padding = (max(self.kernel_size) - min(self.kernel_size)) // 2
        kernel2 = F.pad(kernel2, [padding, padding, padding, padding])
        return kernel1 + kernel2, bias1 + bias2

    @staticmethod
    def _fuse_bn_tensor(branch):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.bn._mean
        running_var = branch.bn._variance
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape([-1, 1, 1, 1])
        return kernel * t, beta - running_mean * gamma / std


class CIBLayer(nn.Layer):
    """The compact inverted block layer, named CIB in YOLOv10"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 add_shortcut=True,
                 use_large_kernel=False,
                 expansion=0.5,
                 act='silu'):
        super().__init__()
        if add_shortcut:
            assert in_channels == out_channels
        self.add_shortcut = add_shortcut

        mid_channels = int(out_channels * expansion)
        if use_large_kernel:
            layer = RepVGGBlock(2 * mid_channels,
                                2 * mid_channels,
                                kernel_size=[7, 3],
                                groups=2 * mid_channels,
                                act=act)
        else:
            layer = BaseConv(2 * mid_channels,
                             2 * mid_channels,
                             ksize=3,
                             stride=1,
                             groups=2 * mid_channels,
                             bias=False,
                             act=act)
        self.block = nn.Sequential(*[
            BaseConv(in_channels,
                     in_channels,
                     ksize=3,
                     stride=1,
                     groups=in_channels,
                     bias=False,
                     act=act),
            BaseConv(in_channels,
                     2 * mid_channels,
                     ksize=1,
                     stride=1,
                     groups=1,
                     bias=False,
                     act=act), layer,
            BaseConv(2 * mid_channels,
                     out_channels,
                     ksize=1,
                     stride=1,
                     groups=1,
                     bias=False,
                     act=act),
            BaseConv(out_channels,
                     out_channels,
                     ksize=3,
                     stride=1,
                     groups=out_channels,
                     bias=False,
                     act=act)
        ])

    def forward(self, x):
        if self.add_shortcut:
            return self.block(x) + x
        else:
            return self.block(x)


class C2fCIBLayer(nn.Layer):
    """Using CIB in C2f"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=1,
                 add_shortcut=False,
                 use_large_kernel=False,
                 expansion=0.5,
                 act="silu"):
        super().__init__()
        mid_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels,
                              2 * mid_channels,
                              ksize=1,
                              stride=1,
                              groups=1,
                              bias=False,
                              act=act)
        self.conv2 = BaseConv((2 + num_blocks) * mid_channels,
                              out_channels,
                              ksize=1,
                              stride=1,
                              groups=1,
                              bias=False,
                              act=act)
        self.bottlenecks = nn.LayerList([
            CIBLayer(mid_channels,
                     mid_channels,
                     add_shortcut=add_shortcut,
                     use_large_kernel=use_large_kernel,
                     expansion=1.0,
                     act=act) for _ in range(num_blocks)
        ])

    def forward(self, x):
        out = list(self.conv1(x).chunk(2, axis=1))
        out.extend(m(out[-1]) for m in self.bottlenecks)
        return self.conv2(paddle.concat(out, axis=1))


class AttnLayer(nn.Layer):
    """Attention layer using in YOLOv10"""

    def __init__(self, embed_dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5

        proj_dim = embed_dim + 2 * self.num_heads * self.key_dim
        self.qkv_proj = BaseConv(embed_dim,
                                 proj_dim,
                                 ksize=1,
                                 stride=1,
                                 groups=1,
                                 bias=False,
                                 act=nn.Identity())
        self.out_proj = BaseConv(embed_dim,
                                 embed_dim,
                                 ksize=1,
                                 stride=1,
                                 groups=1,
                                 bias=False,
                                 act=nn.Identity())
        self.pe = BaseConv(embed_dim,
                           embed_dim,
                           ksize=3,
                           stride=1,
                           groups=embed_dim,
                           bias=False,
                           act=nn.Identity())

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_proj(x)
        qkv = paddle.reshape(qkv, [B, self.num_heads, -1, H * W])
        q, k, v = paddle.split(qkv, [self.key_dim, self.key_dim, self.head_dim],
                               axis=2)
        prod = paddle.matmul(q * self.scale, k, transpose_x=True)
        attn = F.softmax(prod, axis=-1)
        out = paddle.matmul(v, attn, transpose_y=True)
        out = paddle.reshape(out, [B, C, H, W])
        out = out + self.pe(paddle.reshape(v, [B, C, H, W]))
        out = self.out_proj(out)
        return out


class PSALayer(nn.Layer):
    """Partial self-attention layer named PSA in YOLOv10"""

    def __init__(self, embed_dim, expansion=0.5, act='silu'):
        super().__init__()
        hidden_dim = int(embed_dim * expansion)
        self.conv1 = BaseConv(embed_dim,
                              2 * hidden_dim,
                              ksize=1,
                              stride=1,
                              groups=1,
                              bias=False,
                              act=act)
        self.conv2 = BaseConv(2 * hidden_dim,
                              embed_dim,
                              ksize=1,
                              stride=1,
                              groups=1,
                              bias=False,
                              act=act)
        self.attn = AttnLayer(hidden_dim,
                              num_heads=hidden_dim // 64,
                              attn_ratio=0.5)
        self.ffn = nn.Sequential(*[
            BaseConv(hidden_dim,
                     2 * hidden_dim,
                     ksize=1,
                     stride=1,
                     groups=1,
                     bias=False,
                     act=act),
            BaseConv(2 * hidden_dim,
                     hidden_dim,
                     ksize=1,
                     stride=1,
                     groups=1,
                     bias=False,
                     act=nn.Identity())
        ])

    def forward(self, x):
        out = self.conv1(x)
        out1, out2 = paddle.chunk(out, 2, axis=1)
        out2 = out2 + self.attn(out2)
        out2 = out2 + self.ffn(out2)
        out = paddle.concat([out1, out2], axis=1)
        return self.conv2(out)


@register
@serializable
class YOLOv10CSPDarkNet(nn.Layer):
    """
    YOLOv10 CSPDarkNet backbone.
    """
    __shared__ = ['depth_mult', 'width_mult', 'act']

    # in_channels, out_channels, num_blocks, use_scdown, use_c2fcib, add_shortcut, use_sppf, use_psa
    arch_settings = {
        'P5': [[64, 128, 3, False, False, True, False, False],
               [128, 256, 6, False, False, True, False, False],
               [256, 512, 6, True, False, True, False, False],
               [512, 1024, 3, True, True, True, True, True]],
        'X': [[64, 128, 3, False, False, True, False, False],
              [128, 256, 6, False, False, True, False, False],
              [256, 512, 6, True, True, True, False, False],
              [512, 1024, 3, True, True, True, True, True]],
    }

    def __init__(self,
                 arch='P5',
                 depth_mult=1.0,
                 width_mult=1.0,
                 last_stage_ch=1024,
                 use_c2fcib=True,
                 use_large_kernel=False,
                 act='silu',
                 return_idx=[2, 3, 4]):
        super().__init__()
        self.return_idx = return_idx
        arch_setting = self.arch_settings[arch]
        if last_stage_ch != 1024:
            assert last_stage_ch > 0
            arch_setting[-1][1] = last_stage_ch
        _use_c2fcib = use_c2fcib
        _out_channels = []

        input_channels = 3
        base_channels = int(arch_setting[0][0] * width_mult)
        self.stem = BaseConv(input_channels,
                             base_channels,
                             ksize=3,
                             stride=2,
                             groups=1,
                             bias=False,
                             act=act)
        _out_channels.append(base_channels)

        layers_num = 1
        self.csp_dark_blocks = []

        for i, (in_channels, out_channels, num_blocks, use_scdown, use_c2fcib,
                add_shortcut, use_sppf, use_psa) in enumerate(arch_setting):
            in_channels = int(in_channels * width_mult)
            out_channels = int(out_channels * width_mult)
            _out_channels.append(out_channels)
            num_blocks = max(round(num_blocks * depth_mult), 1)
            stage = []

            if use_scdown:
                scdown_layer = self.add_sublayer(
                    f'layers{layers_num}.stage{i + 1}.scdown_layer',
                    SCDownLayer(in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                act=act))
                stage.append(scdown_layer)
                layers_num += 1
            else:
                conv_layer = self.add_sublayer(
                    f'layers{layers_num}.stage{i + 1}.conv_layer',
                    BaseConv(in_channels,
                             out_channels,
                             ksize=3,
                             stride=2,
                             groups=1,
                             bias=False,
                             act=act))
                stage.append(conv_layer)
                layers_num += 1

            if _use_c2fcib and use_c2fcib:
                c2fcib_layer = self.add_sublayer(
                    f'layers{layers_num}.stage{i + 1}.c2fcib_layer',
                    C2fCIBLayer(out_channels,
                                out_channels,
                                num_blocks=num_blocks,
                                add_shortcut=add_shortcut,
                                use_large_kernel=use_large_kernel,
                                expansion=0.5,
                                act=act))
                stage.append(c2fcib_layer)
                layers_num += 1
            else:
                c2f_layer = self.add_sublayer(
                    f'layers{layers_num}.stage{i + 1}.c2f_layer',
                    C2fLayer(out_channels,
                             out_channels,
                             num_blocks=num_blocks,
                             shortcut=add_shortcut,
                             expansion=0.5,
                             depthwise=False,
                             bias=False,
                             act=act))
                stage.append(c2f_layer)
                layers_num += 1

            if use_sppf:
                sppf_layer = self.add_sublayer(
                    f'layers{layers_num}.stage{i + 1}.sppf_layer',
                    SPPFLayer(out_channels,
                              out_channels,
                              ksize=5,
                              bias=False,
                              act=act))
                stage.append(sppf_layer)
                layers_num += 1

            if use_psa:
                psa_layer = self.add_sublayer(
                    f'layers{layers_num}.stage{i + 1}.psa_layer',
                    PSALayer(out_channels, expansion=0.5, act=act))
                stage.append(psa_layer)
                layers_num += 1

            self.csp_dark_blocks.append(nn.Sequential(*stage))

        self._out_channels = [_out_channels[i] for i in self.return_idx]
        self.strides = [[2, 4, 8, 16, 32, 64][i] for i in self.return_idx]

    def forward(self, inputs):
        x = inputs['image']
        outputs = []
        out = self.stem(x)
        if 0 in self.return_idx:
            outputs.append(out)
        for i, layer in enumerate(self.csp_dark_blocks):
            out = layer(out)
            if i + 1 in self.return_idx:
                outputs.append(out)
        return outputs

    @property
    def out_shape(self):
        return [
            ShapeSpec(channels=c, stride=s)
            for c, s in zip(self._out_channels, self.strides)
        ]
