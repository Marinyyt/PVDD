import math
import numpy as np

import torch
import torch.nn as nn

from .layers import *


# resnet block for generator
class ResnetBlock(nn.Module):
    def __init__(self, in_dim, padding_type='zero', norm_layer=None,
                 activation=nn.ReLU(True),
                 use_dropout=False,
                 conv_unit=None,
                 divide_ratio=False,
                 div2conv=False,
                 scale=None,
                 size=None,
                 activation_first=False,
                 se=False,
                 spatial_inject=False,
                 *args, **kwargs
                 ):
        super(ResnetBlock, self).__init__()
        self.divide_ratio = divide_ratio
        self.div2conv = div2conv
        self.scale_factor = scale
        self.size = size
        self.use_se = se
        self.spatial_inject = spatial_inject
        self.conv_unit = nn.Conv2d if conv_unit is None else conv_unit
        self.conv_block = self.build_conv_block(in_dim, padding_type, norm_layer,
                                                activation, use_dropout, activation_first)
        self.upsample_layer = self._build_upsample_layer()
        if div2conv:
            self.div = Div2Conv(in_channels=in_dim, out_channels=in_dim, kernel_size=1, stride=1, padding=0,
                                groups=in_dim, bias=False)

    def _build_upsample_layer(self):
        if self.scale_factor is not None:
            return nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        else:
            return None

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout, activation_first):
        conv_block = []
        p = 0
        if not activation_first:
            if padding_type == 'reflect':
                conv_block += [nn.ReflectionPad2d(1)]
            elif padding_type == 'replicate':
                conv_block += [nn.ReplicationPad2d(1)]
            elif padding_type == 'zero':
                p = 1
            else:
                raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            if self.spatial_inject:
                conv_block += [SFTLayer(1, dim // 2, dim // 2, activation=activation)]
            conv_block += [self.conv_unit(dim, dim, kernel_size=3, padding=p)]
            if norm_layer is not None:
                conv_block += [norm_layer(dim)]
            if activation is not None:
                conv_block += [activation]

            if use_dropout:
                conv_block += [nn.Dropout(0.5)]

            p = 0
            if padding_type == 'reflect':
                conv_block += [nn.ReflectionPad2d(1)]
            elif padding_type == 'replicate':
                conv_block += [nn.ReplicationPad2d(1)]
            elif padding_type == 'zero':
                p = 1
            else:
                raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            if self.spatial_inject:
                conv_block += [SFTLayer(1, dim // 2, dim // 2, activation=activation)]
            conv_block += [self.conv_unit(dim, dim, kernel_size=3, padding=p)]
            if norm_layer is not None:
                conv_block += [norm_layer(dim)]
        else:
            # norm+activation
            if norm_layer is not None:
                conv_block += [norm_layer(dim)]
            if activation is not None:
                conv_block += [activation]
            # conv
            conv_block += [self.conv_unit(dim, dim, kernel_size=3, padding=1)]
            if use_dropout:
                conv_block += [nn.Dropout(0.5)]
            # norm+activation
            if norm_layer is not None:
                conv_block += [norm_layer(dim)]
            if activation is not None:
                conv_block += [activation]
            p = 0
            if padding_type == 'reflect':
                conv_block += [nn.ReflectionPad2d(1)]
            elif padding_type == 'replicate':
                conv_block += [nn.ReplicationPad2d(1)]
            elif padding_type == 'zero':
                p = 1
            else:
                raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            # conv
            conv_block += [self.conv_unit(dim, dim, kernel_size=3, padding=p)]

        if self.use_se:
            conv_block += [self._get_SEblock()(dim, dim)]
        return nn.Sequential(*conv_block)
    
    def _get_SEblock(self):
        return SE

    def _unfold(self, x, x_spatial=None):
        for n, m in enumerate(self.conv_block.children()):
            if self.spatial_inject and (n == 0 or n == 3):
                x = m(x, x_spatial)
            else:
                x = m(x)
        return x

    def forward(self, x, x_spatial=None):
        if self.upsample_layer is not None:
            x = self.upsample_layer(x)
        if self.spatial_inject:
            out = x + self._unfold(x, x_spatial)
        else:
            out = x + self.conv_block(x)
        if not self.divide_ratio:
            return out
        else:
            if self.div2conv:
                out = self.div(out)
            else:
                out = out / np.sqrt(2)
            return out



class ResnetBlockSEHard(ResnetBlock):

    def __init__(self, dim, padding_type, norm_layer, 
                activation=nn.ReLU(True), 
                use_dropout=False, 
                conv_unit=None, 
                divide_ratio=False, 
                div2conv=False, 
                scale=None, 
                size=None, 
                activation_first=False, 
                se=False, 
                spatial_inject=False):
        super(ResnetBlockSEHard, self).__init__(dim, padding_type, norm_layer, 
                activation, use_dropout, conv_unit, divide_ratio, div2conv, 
                scale, size, activation_first, se, spatial_inject)

    def _get_SEblock(self):
        return SEHard



class HINResBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 downsample=False,
                 relu_slope=0.2,
                 use_csff=False,
                 use_HIN=True,
                 scale=None,
                 *args, **kwargs):
        super(HINResBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_dim, in_dim, 1, 1, 0)
        self.use_csff = use_csff
        self.scale_factor = scale

        self.conv_1 = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
            self.csff_dec = nn.Conv2d(in_dim, in_dim, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(in_dim//2, affine=True)
        self.use_HIN = use_HIN
        self.upsample_layer = self._build_upsample_layer()

    def _build_upsample_layer(self):
        if self.scale_factor is not None:
            return nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        else:
            return None

    def forward(self, x, enc=None, dec=None, *args, **kwargs):
        if self.upsample_layer is not None:
            x = self.upsample_layer(x)
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            out = out + self.csff_enc(enc) + self.csff_dec(dec)
        return out


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, groups=inp, bias=False),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )


class GhostBottleneck(nn.Module):
    """
    GhostNet: More Features from Cheap Operations, https://arxiv.org/abs/1911.11907
    """
    def __init__(self,
                 in_dim,
                 hidden_dim=None,
                 out_dim=None,
                 kernel_size=3,
                 stride=1,
                 expand_ratio=1.5,
                 se=False,
                 div2conv=False,
                 *args, **kwargs):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]
        self.div2conv = div2conv
        hidden_dim = hidden_dim if hidden_dim is not None else math.ceil(in_dim * expand_ratio)
        modules = []
        # pw
        modules.append(GhostModule(in_dim, hidden_dim, kernel_size=1, relu=True))
        # dw
        modules.append(depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride == 2 else nn.Sequential())
        # Squeeze-and-Excite
        if se:
            modules.append(SE(hidden_dim, hidden_dim))
        # pw-linear
        modules.append(GhostModule(hidden_dim, out_dim, kernel_size=1, relu=False))
        self.conv = nn.Sequential(*modules)

        if stride == 1 and in_dim == out_dim:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(in_dim, in_dim, 3, stride, relu=False),
                nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False),
            )

        if div2conv:
            self.div = Div2Conv(in_channels=out_dim, out_channels=out_dim, kernel_size=1, stride=1,
                                padding=0, groups=out_dim, bias=False)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        if self.div2conv:
            out = self.div(out)
        return out


class GhostModule(nn.Module):
    def __init__(self,
                 in_dim,
                 oup,
                 kernel_size=1,
                 ratio=2,
                 dw_size=3,
                 stride=1,
                 relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_dim, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out