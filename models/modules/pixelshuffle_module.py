from copy import deepcopy

import torch.nn as nn

try:
    import torch.fx
    torch.fx.wrap('len')
except:
    pass

from .layers import BlurLayer


class PixelShuffleAlign(nn.Module):
    def __init__(self, upsacel_factor: int = 1, mode: str = 'caffe', *args, **kwargs):
        """
        :param upsacel_factor: upsample scale
        :param mode: caffe, pytorch
        """
        super(PixelShuffleAlign, self).__init__()
        self.upscale_factor = upsacel_factor
        self.mode = mode

    def forward(self, x):
        # assert len(x.size()) == 4, "Received input tensor shape is {}".format(
        #     x.size())
        # if len(x.size()) != 4:
        #     raise ValueError("input tensor shape {} is not supported.".format(x.size()))
        N, C, H, W = x.size()
        c = C // (self.upscale_factor ** 2)
        h, w = H * self.upscale_factor, W * self.upscale_factor

        if self.mode == 'caffe':
            # (N, C, H, W) => (N, r, r, c, H, W)
            x = x.reshape(-1, self.upscale_factor,
                          self.upscale_factor, c, H, W)
            x = x.permute(0, 3, 4, 1, 5, 2)
        elif self.mode == 'pytorch':
            # (N, C, H, W) => (N, r, r, c, H, W)
            x = x.reshape(-1, c, self.upscale_factor,
                          self.upscale_factor, H, W)
            x = x.permute(0, 1, 4, 2, 5, 3)
        else:
            raise NotImplementedError(
                "{} mode is not implemented".format(self.mode))

        x = x.reshape(-1, c, h, w)
        return x


def conv_block(in_nc,
               out_nc,
               kernel_size,
               stride=1,
               dilation=1,
               groups=1,
               bias=True,
               padding_type='zero',
               norm_layer=None,
               activation=nn.ReLU(True),
               use_dropout=False,
               conv_unit=None,
               blur_layer=False
               ):
    conv_block = []
    p = 0
    if padding_type == 'reflect':
        conv_block += [nn.ReflectionPad2d(1)]
    elif padding_type == 'replicate':
        conv_block += [nn.ReplicationPad2d(1)]
    elif padding_type == 'zero':
        p = 1
    else:
        raise NotImplementedError('padding [%s] is not implemented' % padding_type)

    conv_block += [conv_unit(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=p, bias=bias)]
    if blur_layer:
        conv_block += [BlurLayer(out_nc)]
    if norm_layer is not None:
        conv_block += [norm_layer(out_nc)]

    conv_block += [deepcopy(activation)]
    if use_dropout:
        conv_block += [nn.Dropout(0.5)]
    return nn.Sequential(*conv_block)


def pixelshuffle_block(in_nc, out_nc,
                       upscale_factor=2,
                       kernel_size=3,
                       stride=1,
                       bias=True,
                       padding_type='zero',
                       norm_layer=nn.InstanceNorm2d,
                       activation=nn.ReLU(True),
                       conv_unit=None,
                       mode_block='pytorch',
                       blur_layer=False,
                       ):
    """
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    """
    conv_unit = nn.Conv2d if conv_unit is None else conv_unit
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride,
                      bias=bias,
                      padding_type=padding_type,
                      norm_layer=norm_layer,
                      activation=activation,
                      conv_unit=conv_unit,
                      blur_layer=blur_layer
                      )

    pixel_shuffle = PixelShuffleAlign(upscale_factor, mode=mode_block)

    norm = norm_layer(out_nc) if norm_layer is not None else None
    a = activation
    model_ = [conv, pixel_shuffle, norm, a] if norm_layer is not None else [conv, pixel_shuffle]
    return nn.Sequential(*model_)


class SubPixelDown(nn.Module):
    def __init__(self, downscale_factor: int = 2, mode: str = 'caffe', *args, **kwargs):
        """
        :param downsacel_factor: downsample scale
        :param mode: caffe, pytorch
        """
        super(SubPixelDown, self).__init__()
        self.dsf = downscale_factor
        self.mode = mode

    def forward(self, x):
        assert len(x.size()) == 4, "Received input tensor shape is {}".format(
            x.size())
        N, C, H, W = x.size()
        c = C * (self.dsf ** 2)
        h, w = H // self.dsf, W // self.dsf

        x = x.reshape(-1, C, h, self.dsf, w, self.dsf)
        if self.mode == 'caffe':
            x = x.permute(0, 3, 5, 1, 2, 4)
        elif self.mode == 'pytorch':
            x = x.permute(0, 1, 3, 5, 2, 4)
        else:
            raise NotImplementedError(
                "{} mode is not implemented".format(self.mode))

        x = x.reshape(-1, c, h, w)
        return x
