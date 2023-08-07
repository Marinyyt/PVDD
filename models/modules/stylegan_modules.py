import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.pixelshuffle_module import PixelShuffleAlign

try:
    from models.modules.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
except Exception as ex:
    print(ex)
    from models.modules.op.op_native import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

__all__ = ['EqualConv2d', 'EqualLinear', 'ModulatedDWConv2d', 'FromRGB', 'ToRGB', 'StyledConv', 'ResBlock',
           'HalfInResBlock', 'SFTLayer']


class ModulatedDWConv2d(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=True,
            upsample=False,
            downsample=False,
    ):
        super().__init__()
        self.upsample=upsample
        self.downsample=downsample
        # create conv
        self.weight_dw = nn.Parameter(
            torch.randn(channels_in, 1, kernel_size, kernel_size)
        )
        self.weight_permute = nn.Parameter(
            torch.randn(channels_out, channels_in, 1, 1)
        )
        self.demodulate = demodulate
        # create modulation network
        if demodulate:
            self.modulation = EqualLinear(style_dim, channels_in, bias_init=1)
        # create demodulation parameters
        if self.demodulate:
            self.register_buffer("style_inv", torch.randn(1, 1, channels_in, 1, 1))
        # some service staff
        self.scale = 1.0 / math.sqrt(channels_in * kernel_size ** 2)
        self.padding = kernel_size // 2

    def forward(self, x, style=None, *args, **kwargs):
        modulation = self.get_modulation(style)
        x = modulation * x

        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        elif self.downsample:
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)

        x = F.conv2d(x, self.weight_dw, padding=self.padding, groups=x.size(1))
        x = F.conv2d(x, self.weight_permute)

        if self.demodulate:
            demodulation = self.get_demodulation(style)
            x = demodulation * x
        return x

    def get_modulation(self, style=None):
        if style is not None:
            style = self.modulation(style).view(style.size(0), -1, 1, 1)
            modulation = self.scale * style
        else:
            modulation = self.scale
        return modulation

    def get_demodulation(self, style):
        w = (self.weight_dw.transpose(0, 1) * self.weight_permute).unsqueeze(0)
        norm = torch.rsqrt((self.scale * self.style_inv * w).pow(2).sum([2, 3, 4]) + 1e-8)
        demodulation = norm
        return demodulation.view(*demodulation.size(), 1, 1)


class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0., lr_mul=1., activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def _deploy_forward(self, x):
        weight = self.weight.clone()
        weight.data = weight.data * self.scale
        if self.activation:
            out = F.linear(x, weight, self.bias * self.lr_mul)
            out = F.leaky_relu(out, negative_slope=0.2)
        else:
            out = F.linear(x, weight, bias=self.bias * self.lr_mul)
        return out

    def forward(self, input):
        if not self.training:
            return self._deploy_forward(input)
        if self.activation:
            out = F.linear(input, self.weight * self.scale, self.bias * self.lr_mul)
            out = F.leaky_relu(out, negative_slope=0.2)
        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def _deploy_forward(self, x):
        weight = self.weight.clone()
        weight.data = weight.data * self.scale
        return F.conv2d(
            x,
            weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

    def forward(self, input):
        # fuse weights for deploy
        if not self.training:
            return self._deploy_forward(input)

        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualizedSE(nn.Module):
    def __init__(self, Cin, Cout):
        super(EqualizedSE, self).__init__()
        num_hidden = max(Cout // 16, 4)
        self.se = nn.Sequential(
            EqualLinear(Cin, num_hidden),
            nn.ReLU(inplace=True),
            EqualLinear(num_hidden, Cout),
            nn.Sigmoid()
        )

    def forward(self, x):
        se = F.adaptive_avg_pool2d(x, 1)
        se = se.view(se.size(0), -1)
        se = self.se(se)
        se = se.view(se.size(0), -1, 1, 1)
        return x * se


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim=None,
            upsample=False,
            downsample=False,
            demodulate=True,
            learned_demodulate=False,
            blur_kernel=[1, 2, 1],
            style_dem=False,
            without_act=False
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.upsample = upsample
        self.downsample = downsample
        self.learned_demodulate = learned_demodulate
        self.style_dem = style_dem
        self.without_act = without_act

        ratio = 1

        if upsample:
            factor = 2
            self.p = (len(blur_kernel) - factor) - (kernel_size - 1)
            self.blur = Blur(channels=out_channel)
            self.pixel_shuffle = PixelShuffleAlign(upsacel_factor=2)
            ratio = 4

        if downsample:
            factor = 2
            self.p = (len(blur_kernel) - factor) + (kernel_size - 1)
            self.blur = Blur(channels=in_channel)
        self.ratio = ratio

        self.out_channel = out_channel

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(out_channel * ratio, in_channel, kernel_size, kernel_size))

        if style_dim is not None:
            self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
            self.demodulate = demodulate
            if self.learned_demodulate:
                self.demodulation = EqualLinear(in_channel if not style_dem else style_dim, out_channel * ratio,
                                                bias_init=1)
            else:
                self.register_parameter("style_inv", nn.Parameter(torch.randn(1, 1, in_channel, 1, 1)))
        else:
            self.demodulate = False

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style_in=None, style_dem=None):
        b, c, h, w = input.shape

        if self.training:
            # in training mode, we will apply original modulation and demodulation algorithm
            if style_in is not None:
                style = self.modulation(style_in)
                input = input * style.view(-1, c, 1, 1)
                weight = self.scale * self.weight
            else:
                weight = self.scale * self.weight
        else:
            weight = self.weight.clone()
            if style_in is not None:
                style = self.modulation(style_in)
                weight.data = self.scale * self.weight.data  # nart not support directly operate weight params
                input = input * style.view(-1, c, 1, 1)
            else:
                weight.data = self.scale * self.weight.data

        if self.demodulate:
            if not self.learned_demodulate:
                demod = torch.rsqrt((weight.unsqueeze(0) * self.style_inv).pow(2).sum([2, 3, 4]) + 1e-8)
            else:
                if not self.without_act:
                    demod = F.sigmoid(self.demodulation(style if not self.style_dem else style_dem))
                else:
                    demod = self.demodulation(style if not self.style_dem else style_dem)

        if self.upsample:
            out = F.conv2d(input, weight, padding=1, stride=1)
            if self.demodulate:
                out = out * demod.view(*demod.size(), 1, 1)
            out = self.pixel_shuffle(out)
            out = self.blur(out)
        elif self.downsample:
            input = self.blur(input)
            out = F.conv2d(input, weight, padding=self.padding, stride=2)
            if self.demodulate:
                out = out * demod.view(-1, self.out_channel, 1, 1)
        else:
            out = F.conv2d(input, weight, padding=self.padding)
            if self.demodulate:
                out = out * demod.view(-1, self.out_channel, 1, 1)

        return out


class ModulatedConv2dSE(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim=None,
            upsample=False,
            downsample=False,
            demodulate=True,
            learned_demodulate=False,
            blur_kernel=[1, 2, 1],
            without_act=False
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.upsample = upsample
        self.downsample = downsample
        self.learned_demodulate = learned_demodulate

        ratio = 1

        if upsample:
            factor = 2
            self.p = (len(blur_kernel) - factor) - (kernel_size - 1)

            self.blur = Blur(channels=out_channel)
            self.pixel_shuffle = PixelShuffleAlign(upsacel_factor=2)
            ratio = 4

        if downsample:
            factor = 2
            self.p = (len(blur_kernel) - factor) + (kernel_size - 1)

            self.blur = Blur(channels=in_channel)

        self.out_channel = out_channel

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(out_channel * ratio, in_channel, kernel_size, kernel_size))

        if style_dim is not None:
            self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
            self.demodulate = demodulate
            if self.learned_demodulate:
                self.demodulation = EqualizedSE(out_channel, out_channel)
        else:
            self.demodulate = False

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style_in=None, *args):
        batch, in_channel, height, width = input.shape

        if self.training:
            if style_in is not None:
                style = self.modulation(style_in)
                weight = self.scale * self.weight
                input = input * style.view(-1, in_channel, 1, 1)
            else:
                weight = self.scale * self.weight
        else:
            weight = self.weight.clone()
            if style_in is not None:
                style = self.modulation(style_in)
                weight.data = self.scale * self.weight.data  # nart not support directly operate weight params
                input = input * style.view(-1, in_channel, 1, 1)
            else:
                weight.data = self.scale * self.weight.data

        if self.upsample:
            out = F.conv2d(input, weight, padding=1, stride=1)
            out = self.pixel_shuffle(out)
            if self.demodulate:
                out = self.demodulation(out)
            out = self.blur(out)
        elif self.downsample:
            input = self.blur(input)
            out = F.conv2d(input, weight, padding=self.padding, stride=2)
            if self.demodulate:
                out = self.demodulation(out)
        else:
            out = F.conv2d(input, weight, padding=self.padding)
            out = out

        return out


class StyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim=None,
            upsample=False,
            downsample=False,
            demodulate=True,
            learned_demodulate=False,
            blur_kernel=[1, 3, 3, 1],
            inject_noise=True,
            se_demodulate=False,
            dw_demodulate=False,
            style_dem=False,
            without_act=False
    ):
        super().__init__()

        if se_demodulate:
            self.conv = ModulatedConv2dSE(
                in_channel,
                out_channel,
                kernel_size,
                style_dim,
                upsample=upsample,
                downsample=downsample,
                blur_kernel=blur_kernel,
                demodulate=demodulate,
                learned_demodulate=learned_demodulate,
            )
        elif dw_demodulate:
            self.conv = ModulatedDWConv2d(
                in_channel,
                out_channel,
                style_dim=style_dim,
                kernel_size=kernel_size,
                demodulate=demodulate,
                upsample=upsample,
                downsample=downsample
            )
        else:
            self.conv = ModulatedConv2d(
                in_channel,
                out_channel,
                kernel_size,
                style_dim,
                upsample=upsample,
                downsample=downsample,
                blur_kernel=blur_kernel,
                demodulate=demodulate,
                learned_demodulate=learned_demodulate,
                style_dem=style_dem,
                without_act=without_act
            )

        self.downsample = downsample
        self.inject_noise = inject_noise

        self.noise = NoiseInjection()
        self.activate = nn.LeakyReLU(0.2)

    def forward(self, input, style=None, noise=None, style_dem=None):
        out = self.conv(input, style, style_dem)
        if self.inject_noise:
            out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out


class FromRGB(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if downsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, out_channel, 1)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = self.conv(input)
        # out = out + self.bias

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, out_channel=3, style_dim=None, upsample=True, learned_demodulate=False,
                 blur_kernel=[1, 3, 3, 1], style_dem=False, without_act=False):
        super().__init__()

        if upsample:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv = ModulatedConv2d(
            in_channel, out_channel, 1, style_dim,
            learned_demodulate=learned_demodulate,
            style_dem=style_dem,
            without_act=without_act
        )
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input, style=None, skip=None, style_dem=None):
        out = self.conv(input, style, style_dem=style_dem)
        # out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    """Implements the blur layer used in StyleGAN."""

    def __init__(self,
                 channels,
                 kernel=(1, 2, 1),
                 normalize=True,
                 flip=False):
        super().__init__()
        kernel = np.array(kernel, dtype=np.float32).reshape(1, 3)
        kernel = kernel.T.dot(kernel)
        if normalize:
            kernel /= np.sum(kernel)
        if flip:
            kernel = kernel[::-1, ::-1]
        kernel = kernel.reshape(3, 3, 1, 1)
        kernel = np.tile(kernel, [1, 1, channels, 1])
        kernel = np.transpose(kernel, [2, 3, 0, 1])
        self.register_buffer('kernel', torch.from_numpy(kernel))
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding=1, groups=self.channels)


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class SFTLayer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, activation=nn.LeakyReLU(negative_slope=0.2, inplace=True)):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = EqualConv2d(input_nc, ngf, 3, 1, 1)
        self.SFT_scale_conv1 = EqualConv2d(ngf, output_nc, 1)
        self.SFT_shift_conv0 = EqualConv2d(input_nc, ngf, 3, 1, 1)
        self.SFT_shift_conv1 = EqualConv2d(ngf, output_nc, 1)
        self.activation = activation

    def forward(self, x, x_spatial, **kwargs):
        # split sft
        x_identity = x[:, :x.size(1) // 2, ...]
        x = x[:, x.size(1) // 2:, ...]
        scale = self.SFT_scale_conv1(self.activation(self.SFT_scale_conv0(x_spatial)))
        shift = self.SFT_shift_conv1(self.activation(self.SFT_shift_conv0(x_spatial)))
        # x * (scale + 1) + shift
        x_modulated = x * scale + shift
        return torch.cat([x_identity, x_modulated], dim=1)


class HalfInResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HalfInResBlock, self).__init__()

        self.identity_conv = EqualConv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)

        self.conv_1 = EqualConv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)  # down sample layer
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_2 = EqualConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.norm = nn.InstanceNorm2d(out_channel // 2, affine=True)

        self.register_buffer('scale_factor', torch.ones(out_channel, 1, 1, 1) / np.sqrt(2))
        self.groups = out_channel

    def forward(self, x):
        h_out = self.conv_1(x)
        h_out_1, h_out_2 = torch.chunk(h_out, 2, dim=1)
        h_out = torch.cat([self.norm(h_out_1), h_out_2], dim=1)
        h_out = self.leaky_relu1(h_out)
        h_out = self.leaky_relu2(self.conv_2(h_out))
        out = h_out + self.identity_conv(x)
        return F.conv2d(out, self.scale_factor, groups=self.groups)


class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            bias=True,
            activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            layers.append(Blur(in_channel))
            stride = 2
            self.padding = kernel_size // 2

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(nn.LeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out