from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .pixelshuffle_module import PixelShuffleAlign



class AvgPool2dConv(nn.Module):
    """Refactor avgpooling to conv, note that `count_include_pad` should be True in `torch.nn.AvgPool2d` 

    Args:
        input_nc: number of input channel
        kernel_size: kernel_size of AvgPooling
        stride: stride of AvgPooling
        padding: padding of AvgPooling
    
    """

    def __init__(self, input_nc, kernel_size, stride, padding=0):
        super(AvgPool2dConv, self).__init__()
        self.input_nc = input_nc
        self.stride = stride
        self.padding = padding
        weight = torch.ones(input_nc, 1, kernel_size, kernel_size) / float(kernel_size * kernel_size)
        self.register_buffer("weight", weight)

    def forward(self, x):
        return F.conv2d(x, self.weight, bias=None, stride=self.stride, padding=self.padding, groups=self.input_nc)




def refactor_pixelshuffleup(weight, bias=None, group=1, upscale_factor=2, caffe2pytorch=True):
    """Rearrange conv weight to support different implementation of pixelshuffle.
       Note that this function is tested only when conv layer is immediately before pixelshuffle layer.
       
       When group > 1, then it is impossible to achieve our goal through only rearranging weight.
       In such case, a intermediate conv layer should be inserted between original conv and pixelshuffle layer.
       Thus, we will not change the weight of original conv, instead, we create a new conv layer and return its weight.
    
    Args:
        weight: weight of conv layer before pixelshuffle
        bias: bias of conv layer before pixelshuffle
        group: group of conv layer before pixelshuffle
        caffe2pytorch: True: transfer from caffe to pytorch implementation
                       False: from pytorch to caffe implementation.
    
    Returns:
        weight, bias: when group == 1, return transferred weight and bias of conv
        weight, conv_layer: when group > 1, return a new intermediate conv layer and its weight
    """

    assert group >= 0

    C, IC, H, W = weight.shape
    c = C // (upscale_factor *upscale_factor)

    if group > 1:
        IC, H, W = C, 1, 1
        conv_layer = nn.Conv2d(C, C, 1, bias=False)
        weight = torch.zeros(C, C, 1, 1)
        for i in range(C):
            weight[i, i, :, :] = 1.
        bias = None

    if caffe2pytorch:
        weight = weight.reshape(upscale_factor, upscale_factor, c, IC, H, W)
        weight = weight.permute(2, 0, 1, 3, 4, 5)
        weight = weight.reshape(C, IC, H, W)
        if bias is not None:
            bias = bias.reshape(upscale_factor, upscale_factor, c)
            bias = bias.permute(2, 0, 1)
            bias = bias.reshape(C,)
    else:
        weight = weight.reshape(c, upscale_factor, upscale_factor, IC, H, W)
        weight = weight.permute(1, 2, 0, 3, 4, 5)
        weight = weight.reshape(C, IC, H, W)
        if bias is not None:
            bias = bias.reshape(c, upscale_factor, upscale_factor)
            bias = bias.permute(1, 2, 0)
            bias = bias.reshape(C,)

    if group > 1:
        state_dict = conv_layer.state_dict()
        state_dict['weight'] = weight
        conv_layer.load_state_dict(state_dict)
        return weight, conv_layer
    else:
        return weight, bias


class Interp2ConvPixelshuffle(nn.Module):
    """Refactor interp layer to conv+pixelshuffle, which can speed up in DSP/NPU.
       Note that there are some differences on the boundary between interp and conv+pixelshuffle.
       By now, we only support `scale_factor=2`, `mode=bilinear` and `align_corners=False` in nn.Upsample.

    Args:
        input_nc: number of input channel
        pixelshuffle_mode: pixelshuffle mode, caffe or pytorch

    """

    # kernel is designed based on caffe implemention of pixelshuffle when `group == 1`
    # and it equals to pytorch implementation when `group > 1`

    def __init__(self, input_nc, pixelshuffle_mode="caffe"):
        super(Interp2ConvPixelshuffle, self).__init__()
        assert input_nc == 1
        self.input_nc = input_nc

        kernel = [[[1/16.0, 3.0/16.0, 0],
                    [3.0/16.0, 9/16.0, 0],
                    [0,    0,    0]],
                    [[0., 3.0/16.0, 1/16.0],
                    [0., 9./16.0, 3.0/16.0],
                    [0,    0,    0]],
                    [[0,    0,    0],
                    [3.0/16.0, 9.0/16.0, 0],
                    [1/16.0, 3.0/16.0, 0]],
                    [[0,    0,    0],
                    [0, 9.0/16.0, 3.0/16.0],
                    [0, 3.0/16.0, 1/16.0]],]

        kernel = np.array(kernel, np.float32)
        kernel = kernel.reshape(4, 1, 3, 3)

        weight = torch.zeros(input_nc * 4, 1, 3, 3)

        for i in range(self.input_nc):
            weight[i*4:i*4+4, 0:1, :, :] = torch.from_numpy(kernel)
        
        if input_nc == 1:
            if pixelshuffle_mode == "pytorch":
                weight, _ = refactor_pixelshuffleup(weight)
            
        self.register_buffer("weight", weight)
        self.pixelshuffle = PixelShuffleAlign(upsacel_factor=2, mode=pixelshuffle_mode)

    def forward(self, x):
        x = F.conv2d(x, self.weight, bias=None, stride=1, padding=1, groups=self.input_nc)
        x = self.pixelshuffle(x)
        return x
        

        
class Concat2Conv(nn.Module):
    """
    Refactor concat to conv. This is needed when concat(a, a) is used in tflite

    Args:
        input_nc: number of input channel
    """

    def __init__(self, input_nc):
        super(Concat2Conv, self).__init__()
        self.input_nc = input_nc
        weight = torch.zeros(input_nc * 2, input_nc, 1, 1)
        oc, _, _, _ = weight.shape
        for o in range(oc // 2):
            weight[o, o, :, :] = torch.ones(1, 1, 1, 1)
        for o in range(oc // 2, oc):
            weight[o, o - oc // 2, :, :] = torch.ones(1, 1, 1, 1)
        self.register_buffer("weight", weight)

    def forward(self, x):
        return F.conv2d(x, self.weight, bias=None, stride=1, padding=0)
