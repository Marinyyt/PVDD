import torch.nn as nn
import torch
from torch.nn.functional import unfold
from torch.nn.functional import fold
import torch.nn.functional as nn_func

import numpy as np
import math

from models.networks.auxiliary_functions import *

from .base import BaseNet
from utils.registry import NETWORK_REGISTRY

class Logger(object):
    def __init__(self, fname="logfile.log"):
        self.terminal = sys.stdout
        self.log = open(fname, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class SepConvFM2D(nn.Module):
    def __init__(self, l_ind):
        super(SepConvFM2D, self).__init__()

        if 0 == l_ind:
            self.f_in = 150
        else:
            self.f_in = max(math.ceil(150 / (2 ** l_ind)), 150)

        self.f_out = max(math.ceil(self.f_in / 2), 150)
        self.n_in = math.ceil(15 / (2 ** l_ind))
        self.n_out = math.ceil(self.n_in / 2)
        self.vh_groups = (self.f_in // 3) * self.n_in
        self.f_groups = self.n_in
        self.n_groups = self.f_out
        self.f_in_g = self.f_in * self.n_in
        self.f_out_g = self.f_out * self.n_in
        self.n_in_g = self.n_in * self.n_groups
        self.n_out_g = self.n_out * self.n_groups
        self.conv_vh = nn.Conv2d(in_channels=self.f_in_g, out_channels=self.f_in_g,
                                 kernel_size=(7, 7), bias=False, groups=self.vh_groups)
        self.conv_f = nn.Conv2d(in_channels=self.f_in_g, out_channels=self.f_out_g,
                                kernel_size=(1, 1), bias=False, groups=self.f_groups)
        self.conv_n = nn.Conv2d(in_channels=self.n_in_g, out_channels=self.n_out_g,
                                kernel_size=(1, 1), bias=False, groups=self.n_groups)

    def forward(self, x):
        b, n, f, v, h = x.shape  # batches, neighbors, features, horizontal, vertical

        x = reflection_pad_2d(x.reshape(b, n * f, v, h), (3, 3))
        x = self.conv_vh(x)
        x = self.conv_f(x).reshape(b, n, self.f_out, v, h)
        x = self.conv_n(x.transpose(1, 2).reshape(b, self.f_out * n, v, h)).\
            reshape(b, self.f_out, self.n_out, v, h).transpose(1, 2)

        return x


class SepConvOut2D(nn.Module):
    def __init__(self, l_ind):
        super(SepConvOut2D, self).__init__()

        self.f_in = max(math.ceil(150 / (2 ** l_ind)), 150)
        self.vh_groups = self.f_in // 3
        self.conv_vh = nn.Conv2d(in_channels=self.f_in, out_channels=self.f_in,
                                 kernel_size=(7, 7), bias=False, groups=self.vh_groups)
        self.conv_f = nn.Conv2d(in_channels=self.f_in, out_channels=3,
                                kernel_size=(1, 1), bias=False)

    def forward(self, x):
        x = reflection_pad_2d(x.squeeze(1), (3, 3))
        x = self.conv_vh(x)
        x = self.conv_f(x)

        return x


class SepConvReF2D(nn.Module):
    def __init__(self):
        super(SepConvReF2D, self).__init__()

        self.sep_conv = SepConvFM2D(0)
        self.b = nn.Parameter(torch.zeros((1, self.sep_conv.n_out, self.sep_conv.f_out, 1, 1), dtype=torch.float32))
        self.re = nn.ReLU()

    def forward(self, x):
        x = self.sep_conv(x)
        x = x + self.b
        x = self.re(x)

        return x

    def extra_repr(self):
        return 'b.shape=' + str(tuple(self.b.shape))


class SepConvBnReM2D(nn.Module):
    def __init__(self, l_ind):
        super(SepConvBnReM2D, self).__init__()

        self.sep_conv = SepConvFM2D(l_ind)
        self.bn = nn.BatchNorm2d(num_features=self.sep_conv.f_out * self.sep_conv.n_out)
        self.re = nn.ReLU()

    def forward(self, x):
        x = self.sep_conv(x)
        b, n, f, v, h = x.shape  # batches, neighbors, features, horizontal, vertical
        x = self.bn(x.reshape(b, n * f, v, h)).reshape(b, n, f, v, h)
        x = self.re(x)

        return x


class SepConvOutB2D(nn.Module):
    def __init__(self, l_ind):
        super(SepConvOutB2D, self).__init__()

        self.sep_conv = SepConvOut2D(l_ind)
        self.b = nn.Parameter(torch.zeros((1, 3, 1, 1), dtype=torch.float32))

    def forward(self, x):
        x = self.sep_conv(x)
        x = x + self.b

        return x

    def extra_repr(self):
        return 'b.shape=' + str(tuple(self.b.shape))


class SepConvNet2D(nn.Module):
    def __init__(self):
        super(SepConvNet2D, self).__init__()

        self.sep_conv_block0 = SepConvReF2D()
        for i in range(1, 4):
            self.add_module('sep_conv_block{}'.format(i), SepConvBnReM2D(i))
        self.add_module('sep_conv_block{}'.format(4), SepConvOutB2D(4))

    def forward(self, x_f):
        for name, layer in self.named_children():
            x_f = layer(x_f)

        return x_f


class ResNn(nn.Module):
    def __init__(self):
        super(ResNn, self).__init__()

        self.sep_conv_net = SepConvNet2D()


    def forward(self, x_f, x, x_valid):

        x_f = self.sep_conv_net(x_f).squeeze(2)

        return x_valid - x_f

@NETWORK_REGISTRY.register()
class VidCnn(BaseNet):
    def __init__(self, input_nc = 3):
        super(VidCnn, self).__init__()

        self.res_nn = ResNn()

    def find_nn(self, seq_pad):
        seq_n = seq_pad[:, :, 3:-3, 37:-37, 37:-37]
        b, c, f, h, w = seq_pad.shape
        min_d = torch.full((b, 1, f - 6, h - 88, w - 88, 14), float('inf'),
                           dtype=seq_pad.dtype, device=seq_pad.device)
        min_i = torch.full(min_d.shape, -(seq_n.numel() + 1),
                           dtype=torch.long, device=seq_pad.device)
        i_arange_patch_pad = torch.arange(b * f * (h - 14) * (w - 14), dtype=torch.long,
                                          device=seq_pad.device).view(b, 1, f, (h - 14), (w - 14))
        i_arange_patch_pad = i_arange_patch_pad[..., 3:-3, 37:-37, 37:-37]
        i_arange_patch = torch.arange(np.array(min_d.shape[0:-1]).prod(), dtype=torch.long,
                                      device=seq_pad.device).view(min_d.shape[0:-1])

        for t_s in range(7):
            t_e = t_s - 6 if t_s != 6 else None
            for v_s in range(75):
                v_e = v_s - 74 if v_s != 74 else None
                for h_s in range(75):
                    if h_s == 37 and v_s == 37 and t_s == 3:
                        continue
                    h_e = h_s - 74 if h_s != 74 else None

                    seq_d = ((seq_pad[..., t_s:t_e, v_s:v_e, h_s:h_e] - seq_n) ** 2).mean(dim=1, keepdim=True)
                    seq_d = torch.cumsum(seq_d, dim=-1)
                    tmp = seq_d[..., 0:-15]
                    seq_d = seq_d[..., 14:]
                    seq_d[..., 1:] = seq_d[..., 1:] - tmp

                    seq_d = torch.cumsum(seq_d, dim=-2)
                    tmp = seq_d[..., 0:-15, :]
                    seq_d = seq_d[..., 14:, :]
                    seq_d[..., 1:, :] = seq_d[..., 1:, :] - tmp

                    neigh_d_max, neigh_i_rel = min_d.max(-1)
                    neigh_i_abs = neigh_i_rel + i_arange_patch * 14
                    tmp_i = i_arange_patch_pad + ((t_s - 3) * (h - 14) * (w - 14) + (v_s - 37) * (w - 14) + h_s - 37)

                    i_change = seq_d < neigh_d_max
                    min_d.flatten()[neigh_i_abs[i_change]] = seq_d.flatten()[i_arange_patch[i_change]]
                    min_i.flatten()[neigh_i_abs[i_change]] = tmp_i[i_change]

        return min_d, min_i

    def create_layers(self, seq_pad, min_i):

        b, c, f, h, w = seq_pad.shape
        in_layers = torch.full((b, 15, 147, f - 6, h - 74, w - 74),
                               float('nan'), dtype=seq_pad.dtype, device=seq_pad.device)

        self_i = torch.arange(b * f * (h - 6) * (w - 6), dtype=torch.long,
                              device=seq_pad.device).view(b, 1, f, h - 6, w - 6)
        self_i = self_i[..., 3:-3, 37:-37, 37:-37].unsqueeze(-1)
        min_i = torch.cat((self_i, min_i), dim=-1)

        f_ind = 0
        min_i = min_i.permute(0, 2, 5, 1, 3, 4)
        min_i = min_i.reshape(b * (f - 6) * 15, h - 80, w - 80)

        for map_v_s in range(7):
            for map_h_s in range(7):
                min_i_tmp = min_i[..., map_v_s:(h - 80 - (h - 74 - map_v_s) % 7):7, \
                                       map_h_s:(w - 80 - (w - 74 - map_h_s) % 7):7].flatten()

                layers_pad_tmp = unfold(seq_pad.transpose(1, 2).
                                        reshape(b * f, 3, h, w), (7, 7))
                layers_pad_tmp = layers_pad_tmp.transpose(0, 1).reshape(147, -1)
                layers_pad_tmp = layers_pad_tmp[:, min_i_tmp]
                layers_pad_tmp = layers_pad_tmp.view(147, b * (f - 6) * 15,
                                                    ((h - 74 - map_v_s) // 7) * ((w - 74 - map_h_s) // 7))
                layers_pad_tmp = layers_pad_tmp.transpose(0, 1)
                layers_pad_tmp = fold(
                    input=layers_pad_tmp,
                    output_size=(h - 74 - map_v_s - (h - 74 - map_v_s) % 7, 
                                 w - 74 - map_h_s - (w - 74 - map_h_s) % 7),
                    kernel_size=(7, 7), stride=7)
                layers_pad_tmp = layers_pad_tmp.view(b, f - 6, 15, 3, \
                    h - 74 - map_v_s - (h - 74 - map_v_s) % 7, 
                    w - 74 - map_h_s - (w - 74 - map_h_s) % 7)
                layers_pad_tmp = layers_pad_tmp.permute(0, 2, 3, 1, 4, 5)

                in_layers[:, :, f_ind:f_ind + 3, :, \
                    map_v_s:(h - 74 - (h - 74 - map_v_s) % 7), 
                    map_h_s:(w - 74 - (w - 74 - map_h_s) % 7)] = layers_pad_tmp
                f_ind = f_ind + 3
        in_layers = in_layers[..., 6:-6, 6:-6]

        return in_layers

    def forward(self, seq_in, gpu_usage):
        b, f, c, h, w = seq_in.size()
        seq_in = seq_in.permute(0, 2, 1, 3, 4)
        reflect_pad = (0, 10, 10)
        const_pad = (37, 37, 37, 37, 0, 0)
        seq_in = reflection_pad_3d(seq_in, reflect_pad)
        seq_in = nn_func.pad(seq_in, const_pad, mode = 'constant', value = -1)
        # print(seq_in.size())
        if gpu_usage == 1 and torch.cuda.is_available():
            min_i = self.find_sorted_nn(seq_in.cuda())
        else:
            min_i = self.find_sorted_nn(seq_in)

        seq_in = seq_in[..., 4:-4, 4:-4]
        in_layers = self.create_layers(seq_in, min_i)
        seq_valid_full = seq_in[..., 3 : -3, 43:-43, 43:-43]
        seq_valid = seq_valid_full[..., 0, :, :]
        in_layers = in_layers.squeeze(-3)
        seq_valid_full = seq_valid_full.squeeze(-3)
        in_weights = (in_layers - in_layers[:, 0:1, ...]) ** 2
        # print(in_weights.size())
        b, n, f, v, h = in_weights.shape
        in_weights = in_weights.view(b, n, f // 3, 3, v, h).mean(2)
        in_layers = torch.cat((in_layers, in_weights), 2)

        seq_out = self.res_nn(in_layers, seq_valid_full, seq_valid)
        # print(seq_out.size())
        return seq_out

    def find_sorted_nn(self, seq_in):
        seq_pad_nn = seq_in
        min_d, min_i = self.find_nn(seq_pad_nn)
        min_d, sort_i = torch.sort(min_d, -1)
        min_i = min_i.gather(-1, sort_i)

        return min_i


class ImCnn(nn.Module):
    def __init__(self):
        super(ImCnn, self).__init__()

        self.res_nn = ResNn()

    def find_nn(self, seq_pad):
        seq_n = seq_pad[..., 37:-37, 37:-37]
        b, c, f, h, w = seq_pad.shape
        min_d = torch.full((b, 1, f, h - 80, w - 80, 14), float('inf'),
                           dtype=seq_pad.dtype, device=seq_pad.device)
        min_i = torch.full(min_d.shape, -(seq_n.numel() + 1),
                           dtype=torch.long, device=seq_pad.device)
        i_arange_patch_pad = torch.arange(b * f * (h - 6) * (w - 6), dtype=torch.long,
                                          device=seq_pad.device).view(b, 1, f, h - 6, w - 6)
        i_arange_patch_pad = i_arange_patch_pad[..., 37:-37, 37:-37]
        i_arange_patch = torch.arange(np.array(min_d.shape[0:-1]).prod(), dtype=torch.long,
                                      device=seq_pad.device).view(min_d.shape[0:-1])

        for v_s in range(75):
            v_e = v_s - 74 if v_s != 74 else None
            for h_s in range(75):
                if h_s == 37 and v_s == 37:
                    continue
                h_e = h_s - 74 if h_s != 74 else None

                seq_d = ((seq_pad[..., v_s:v_e, h_s:h_e] - seq_n) ** 2).mean(dim=1, keepdim=True)

                seq_d = torch.cumsum(seq_d, dim=-1)
                tmp = seq_d[..., 0:-7]
                seq_d = seq_d[..., 6:]
                seq_d[..., 1:] = seq_d[..., 1:] - tmp

                seq_d = torch.cumsum(seq_d, dim=-2)
                tmp = seq_d[..., 0:-7, :]
                seq_d = seq_d[..., 6:, :]
                seq_d[..., 1:, :] = seq_d[..., 1:, :] - tmp

                neigh_d_max, neigh_i_rel = min_d.max(-1)
                neigh_i_abs = neigh_i_rel + i_arange_patch * 14
                tmp_i = i_arange_patch_pad + ((v_s - 37) * (w - 6) + h_s - 37)

                i_change = seq_d < neigh_d_max
                min_d.flatten()[neigh_i_abs[i_change]] = seq_d.flatten()[i_arange_patch[i_change]]
                min_i.flatten()[neigh_i_abs[i_change]] = tmp_i[i_change]

        return min_d, min_i

    def create_layers(self, seq_pad, min_i):

        b, c, f, h, w = seq_pad.shape

        in_layers = torch.full((b, 15, 147, f, h - 74, w - 74),
                               float('nan'), dtype=seq_pad.dtype, device=seq_pad.device)

        self_i = torch.arange(b * f * (h - 6) * (w - 6), dtype=torch.long,
                              device=seq_pad.device).view(b, 1, f, h - 6, w - 6)
        self_i = self_i[..., 37:-37, 37:-37].unsqueeze(-1)
        min_i = torch.cat((self_i, min_i), dim=-1)

        f_ind = 0
        min_i = min_i.permute(0, 2, 5, 1, 3, 4)
        min_i = min_i.reshape(b * f * 15, h - 80, w - 80)

        for map_v_s in range(7):
            for map_h_s in range(7):
                min_i_tmp = min_i[..., map_v_s:(h - 80 - (h - 74 - map_v_s) % 7):7, \
                                       map_h_s:(w - 80 - (w - 74 - map_h_s) % 7):7].flatten()

                layers_pad_tmp = unfold(seq_pad.transpose(1, 2).
                                        reshape(b * f, 3, h, w), (7, 7))
                layers_pad_tmp = layers_pad_tmp.transpose(0, 1).reshape(147, -1)
                layers_pad_tmp = layers_pad_tmp[:, min_i_tmp]
                layers_pad_tmp = layers_pad_tmp.view(147, b * f * 15,
                                                    ((h - 74 - map_v_s) // 7) * ((w - 74 - map_h_s) // 7))
                layers_pad_tmp = layers_pad_tmp.transpose(0, 1)
                layers_pad_tmp = fold(
                    input=layers_pad_tmp,
                    output_size=(h - 74 - map_v_s - (h - 74 - map_v_s) % 7, 
                                 w - 74 - map_h_s - (w - 74 - map_h_s) % 7),
                    kernel_size=(7, 7), stride=7)
                layers_pad_tmp = layers_pad_tmp.view(b, f, 15, 3, \
                    h - 74 - map_v_s - (h - 74 - map_v_s) % 7, 
                    w - 74 - map_h_s - (w - 74 - map_h_s) % 7)
                layers_pad_tmp = layers_pad_tmp.permute(0, 2, 3, 1, 4, 5)

                in_layers[:, :, f_ind:f_ind + 3, :, \
                    map_v_s:(h - 74 - (h - 74 - map_v_s) % 7), 
                    map_h_s:(w - 74 - (w - 74 - map_h_s) % 7)] = layers_pad_tmp
                f_ind = f_ind + 3
        in_layers = in_layers[..., 6:-6, 6:-6]

        return in_layers

    def forward(self, seq_in, gpu_usage):
        if gpu_usage == 1 and torch.cuda.is_available():
            min_i = self.find_sorted_nn(seq_in.cuda()).cpu()
        else:
            min_i = self.find_sorted_nn(seq_in)

        in_layers = self.create_layers(seq_in, min_i)
        in_layers = in_layers.squeeze(-3)

        seq_valid_full = seq_in[..., 43:-43, 43:-43]
        seq_valid = seq_valid_full[..., 0, :, :]
        seq_valid_full = seq_valid_full.squeeze(-3)
        
        in_weights = (in_layers - in_layers[:, 0:1, ...]) ** 2
        b, n, f, v, h = in_weights.shape
        in_weights = in_weights.view(b, n, f // 3, 3, v, h).mean(2)
        in_layers = torch.cat((in_layers, in_weights), 2)

        seq_out = self.res_nn(in_layers, seq_valid_full, seq_valid)

        return seq_out

    def find_sorted_nn(self, seq_in):
        seq_pad_nn = seq_in
        min_d, min_i = self.find_nn(seq_pad_nn)
        min_d, sort_i = torch.sort(min_d, -1)
        min_i = min_i.gather(-1, sort_i)

        return min_i


class ConvRe3DF(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvRe3DF, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3, 3), bias=True,
                              padding=(0, 1, 1), padding_mode='zeros')
        self.re = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.re(x)

        return x


class ConvBnRe3DM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBnRe3DM, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3, 3), bias=False,
                              padding=(0, 1, 1), padding_mode='zeros')
        self.re = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.re(x)

        return x


class ConvRe2DF(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvRe2DF, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), bias=False,
                              padding=(1, 1), padding_mode='zeros')
        self.re = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.re(x)

        return x


class ConvBnRe2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBnRe2D, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), bias=False,
                              padding=(1, 1), padding_mode='zeros')
        self.re = nn.LeakyReLU()

        self.a = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x):
        x = self.conv(x)
        x = self.re(x)

        return x


class Conv2DL(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv2DL, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), bias=True,
                              padding=(1, 1), padding_mode='zeros')

    def forward(self, x):
        x = self.conv(x)

        return x


class Conv3DNet(nn.Module):
    def __init__(self):
        super(Conv3DNet, self).__init__()

        self.conv_3d_block0 = ConvRe3DF(in_ch=6, out_ch=48)
        
        for i in range(1, 3):
            out_ch_tmp = 48 if i < 2 else 96
            self.add_module('conv_3d_block{}'.format(i), ConvBnRe3DM(in_ch=48, out_ch=out_ch_tmp))

    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)

        return x


class Conv2DNet(nn.Module):
    def __init__(self):
        super(Conv2DNet, self).__init__()

        for i in range(16):
            self.add_module('conv_2d_block{}'.format(i), ConvBnRe2D(in_ch=96, out_ch=96))
        
        self.add_module('conv_2d_block{}'.format(16), Conv2DL(in_ch=96, out_ch=3))

    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)

        return x


class TfNet3D(nn.Module):
    def __init__(self):
        super(TfNet3D, self).__init__()

        self.conv_3d_net = Conv3DNet()
        self.conv_2d_net = Conv2DNet()

    def forward(self, x):
        x = self.conv_3d_net(x).squeeze(-3)
        x = self.conv_2d_net(x)

        return x

@NETWORK_REGISTRY.register()
class TfNet(BaseNet):
    def __init__(self, num_in = 3):
        super(TfNet, self).__init__()
        
        self.conv_net = TfNet3D()
    
    def forward(self, noisy, denoised):
        b, f, c, h, w = noisy.size()
        seq_in = torch.cat([noisy, denoised], dim = 2)
        seq_in = seq_in.permute(0, 2, 1, 3, 4)
        # seq_in = reflection_pad_t_3d(seq_in, 3)

        xn = seq_in[:, 3:6, 3, :, :].clone()
        x = self.conv_net(seq_in)
        x = xn - x

        return x
