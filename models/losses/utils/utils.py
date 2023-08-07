import torch
import torch.nn as nn
import numpy as np
try:
    from debayer import Debayer3x3
except:
    pass



def conv_4to3(x):
    """Demosaic from the stacked tensor"""
    assert(x.shape[1] == 4)
    b = x.shape[0]
    w = x.shape[2]
    h = x.shape[3]

    awb_tensor = torch.tensor([2.25, 1, 1, 1.7]).view(1, 4, 1, 1).cuda()
    x = x * awb_tensor
    x = nn.PixelShuffle(2)(x)

    # Demosaic
    demosaic_out = Debayer3x3().cuda()(x)
    # demosaic_out = torch.clamp(demosaic_out, -1, 1)
    # The inversed exponential would be troublesome with regard to gradients
    # demosaic_out = torch.clamp(((demosaic_out + 1) / 2), 0, 1) ** (1/2.2)
    # demosaic_out = torch.clamp(((demosaic_out + 1) / 2), 0, 1) ** (1/2.2)
    # demosaic_out = (demosaic_out - 0.5) * 2.0

    return demosaic_out



def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    HH.weight.requires_grad = False

    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return HH


def RGB2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray