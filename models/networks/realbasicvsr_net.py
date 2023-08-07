from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNet
from .basicvsr_net import BasicVSR, BasicVSRFw, ResidualBlockNoBN
from .basicvsrpp_net import BasicVSRPlusPlus
from ..utils.util import make_layer
from utils.registry import NETWORK_REGISTRY


@NETWORK_REGISTRY.register()
class RealBasicVSR(BaseNet):
    def __init__(self,
                 input_nc=3,
                 num_feat=64,
                 num_block=7,
                 num_block_f=5,
                 dynamic_refine_thres=255.,
                 is_sequential_cleaning=False,
                 recurrent_model='BasicVSR',
                 spynet_path=None
                 ):
        super(RealBasicVSR, self).__init__()
        self.dynamic_refine_thres = dynamic_refine_thres / 255.
        self.is_sequential_cleaning = is_sequential_cleaning
        self.clean_model = ResUNet(input_nc=input_nc, depth=2, num_feat=num_feat, num_block=num_block)
        if recurrent_model == 'BasicVSR':
            self.basicvsr = BasicVSR(
                num_in = input_nc,
                num_feat=num_feat,
                num_block=num_block,
                num_block_f=num_block_f,
                spynet_path=spynet_path
            )
        elif recurrent_model == 'BasicVSRFw':
            self.basicvsr = BasicVSRFw(
                num_feat=num_feat,
                num_block=num_block,
                num_block_f=num_block_f,
                spynet_path=spynet_path
            )
        elif recurrent_model == 'BasicVSRPlusPlus':
            self.basicvsr = BasicVSRPlusPlus(
                num_feat=num_feat,
                num_block=num_block,
                num_block_f=num_block_f,
                max_residue_magnitude=10,
                spynet_path=spynet_path
            )

    def forward(self, x, return_clean=False):
        n, t, c, h, w = x.size()

        for _ in range(0, 3):  # at most 3 cleaning, determined empirically
            if self.is_sequential_cleaning:
                residues = []
                for i in range(0, t):
                    residue_i = self.clean_model(x[:, i, :, :, :])
                    x[:, i, :, :, :] += residue_i
                    residues.append(residue_i)
                residues = torch.stack(residues, dim=1)
            else:  # time -> batch, then apply cleaning at once
                x = x.view(-1, c, h, w)
                residues = self.clean_model(x)
                x = (x + residues).view(n, t, c, h, w)

            # determine whether to continue cleaning
            if torch.mean(torch.abs(residues)) < self.dynamic_refine_thres:
                break

        # BasicVSRFw
        outputs = self.basicvsr(x)

        if return_clean:
            return outputs, x.view(-1, c, h, w)
        else:
            return outputs


class ResUNet(BaseNet):
    def __init__(self,
                 input_nc=3,
                 depth=2,
                 num_feat=64,
                 num_block=9):
        super(ResUNet, self).__init__()
        self.depth = depth

        self.conv_in = nn.Conv2d(input_nc, num_feat, 3, 1, 1)

        self.down_0 = conv_block(num_in_ch=num_feat, num_out_ch=num_feat)
        self.down_1 = conv_block(num_in_ch=num_feat, num_out_ch=num_feat)
        self.trunk_blocks = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)
        self.up_0 = up_block(num_feat*2, num_feat*4)
        self.up_1 = up_block(num_feat*2, num_feat*4)

        self.conv_out = nn.Conv2d(num_feat, input_nc, 3, 1, 1)

    def forward(self, x):
        skips = []
        feat = self.conv_in(x)
        feat_in = feat.clone()
        skips.append(feat)
        for i in range(self.depth):
            feat = getattr(self, 'down_{}'.format(i))(feat)
            skips.append(feat)
        # middle blocks
        feat = self.trunk_blocks(feat)
        for i in range(self.depth):
            feat = getattr(self, 'up_{}'.format(i))(torch.cat([skips.pop(), feat], dim=1))

        out = self.conv_out(feat+feat_in)
        return out

class ResUNet2(BaseNet):
    def __init__(self,
                 input_nc=3,
                 depth=2,
                 num_feat=64,
                 num_block=9):
        super(ResUNet2, self).__init__()
        self.depth = depth

        self.conv_in = nn.Conv2d(input_nc+1, num_feat, 3, 1, 1)

        self.down_0 = conv_block(num_in_ch=num_feat, num_out_ch=num_feat)
        self.down_1 = conv_block(num_in_ch=num_feat, num_out_ch=num_feat)
        self.trunk_blocks = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)
        self.up_0 = up_block(num_feat*2, num_feat*4)
        self.up_1 = up_block(num_feat*2, num_feat*4)

        self.conv_out = nn.Conv2d(num_feat, input_nc, 3, 1, 1)

    def forward(self, x):
        skips = []
        feat = self.conv_in(x)
        feat_in = feat.clone()
        skips.append(feat)
        for i in range(self.depth):
            feat = getattr(self, 'down_{}'.format(i))(feat)
            skips.append(feat)
        # middle blocks
        feat = self.trunk_blocks(feat)
        for i in range(self.depth):
            feat = getattr(self, 'up_{}'.format(i))(torch.cat([skips.pop(), feat], dim=1))

        out = self.conv_out(feat+feat_in)
        return out

def conv_block(num_in_ch, num_out_ch):
    model = nn.Sequential(
        nn.Conv2d(num_in_ch, num_out_ch, 3, 2, 1, bias=True),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
    )
    return model


def up_block(num_in_ch, num_out_ch):
    model = nn.Sequential(
        nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True),
        nn.PixelShuffle(upscale_factor=2),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )
    return model

