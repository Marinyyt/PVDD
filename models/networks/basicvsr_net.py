import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNet
from .spynet import SpyNet
from ..utils.warp import flow_warp
from ..utils.util import make_layer
from utils.registry import NETWORK_REGISTRY


class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR with same size output.
    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(

            nn.Conv2d(num_in_ch, num_out_ch, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(num_out_ch, num_out_ch, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch)
        )

    def forward(self, fea):
        return self.main(fea)


class ResidualBlocksWithInConv(nn.Module):
    """residual block used in BasicVSR for feats propagation.
    Args:
        num_in_ch (int): Number of input channels. Default: 64.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 5.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch)
        )

    def forward(self, fea):
        return self.main(fea)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.
    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
    """

    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


@NETWORK_REGISTRY.register()
class BasicVSR(BaseNet):
    """A recurrent network for video restoration
    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_in = 3, num_feat=64, num_block=7, num_block_f=5, spynet_path=None):
        super(BasicVSR, self).__init__()
        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)

        # feat extractor
        self.feat_extractor = ConvResidualBlocks(num_in, num_feat, num_block_f)

        # propagation
        self.backward_trunk = ResidualBlocksWithInConv(num_feat * 2, num_feat, num_block)
        self.forward_trunk = ResidualBlocksWithInConv(num_feat * 2, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, num_in, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1[:, :3, :, :], x_2[:,  :3, :, :]).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2[:,  :3, :, :], x_1[:,  :3, :, :]).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, x):
        """Forward function of BasicVSR.
        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        b, n, c, h, w = x.size()
        #x_down_sample = F.interpolate(x.reshape(-1, c, h, w),
        #                               scale_factor=1/4., mode='bilinear').view(-1, n, c, h//4, w//4)
        # flows_forward, flows_backward = self.get_flow(x_down_sample)
        b, n, _, h, w = x.size()

        # feature extractor
        feats_ = self.feat_extractor(x.view(-1, c, h, w))
        h_, w_ = feats_.shape[2:]
        feats_ = feats_.view(b, n, -1, h_, w_)
        feats = [feats_[:, i, :, :, :] for i in range(0, n)]

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h_, w_)
        for i in range(n - 1, -1, -1):
            feat_i = feats[i]
            # if i < n - 1:
            #     flow = flows_backward[:, i, :, :, :]
            #     feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([feat_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_i = feats[i]
            # if i > 0:
            #     flow = flows_forward[:, i - 1, :, :, :]
            #     feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([feat_i, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = torch.cat([out_l[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))

            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = x_i  # same size with output
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)


@NETWORK_REGISTRY.register()
class BasicVSRFw(BaseNet):
    """A recurrent network for video restoration, only forward propagation
    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=7, num_block_f=5, spynet_path=None):
        super(BasicVSRFw, self).__init__()
        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)

        # feat extractor
        self.feat_extractor = ConvResidualBlocks(3, num_feat, num_block_f)

        # propagation
        self.forward_trunk = ResidualBlocksWithInConv(num_feat * 2, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.
        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        b, n, c, h, w = x.size()
        x_down_sample = F.interpolate(x.reshape(-1, c, h, w),
                                      scale_factor=1/4., mode='bilinear').view(-1, n, c, h//4, w//4)
        flows_forward = self.get_flow(x_down_sample)
        b, n, _, h, w = x.size()

        # feature extractor
        feats_ = self.feat_extractor(x.view(-1, c, h, w))
        h_, w_ = feats_.shape[2:]
        feats_ = feats_.view(b, n, -1, h_, w_)
        feats = [feats_[:, i, :, :, :] for i in range(0, n)]

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h_, w_)
        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_i = feats[i]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([feat_i, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = feat_prop
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = x_i  # same size with output
            out += base
            out_l.append(out)

        return torch.stack(out_l, dim=1)



@NETWORK_REGISTRY.register()
class BasicVSR_S(BaseNet):
    """A recurrent network for video restoration
    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=7, num_block_f=5, spynet_path=None):
        super(BasicVSR_S, self).__init__()
        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)

        # feat extractor
        self.feat_extractor = ConvResidualBlocks(3, num_feat, num_block_f)

        # propagation
        self.backward_trunk = ResidualBlocksWithInConv(num_feat * 2, num_feat, num_block)
        self.forward_trunk = ResidualBlocksWithInConv(num_feat * 2, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat * 2, 32, 1, 1, 0, bias=True)

        self.upconv1 = nn.Conv2d(32, 32 * 2, 3, 1, 1, bias=True)
        self.dconv1 = nn.ConvTranspose2d(32 * 2, 32, 2, stride=2, padding=0)
        self.upconv2 = nn.Conv2d(32, 32 * 2, 3, 1, 1, bias=True)
        self.dconv2 = nn.ConvTranspose2d(32 * 2, 32, 2, stride=2, padding=0)

        self.conv_hr = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv_last = nn.Conv2d(32, 3, 3, 1, 1)

        # self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        # self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        # self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        # self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, x):
        """Forward function of BasicVSR.
        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        b, n, c, h, w = x.size()
        x_down_sample = F.interpolate(x.reshape(-1, c, h, w),
                                      scale_factor=1/4., mode='bilinear').view(-1, n, c, h//4, w//4)
        flows_forward, flows_backward = self.get_flow(x_down_sample)
        b, n, _, h, w = x.size()

        # feature extractor
        feats_ = self.feat_extractor(x.view(-1, c, h, w))
        h_, w_ = feats_.shape[2:]
        feats_ = feats_.view(b, n, -1, h_, w_)
        feats = [feats_[:, i, :, :, :] for i in range(0, n)]

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h_, w_)
        for i in range(n - 1, -1, -1):
            feat_i = feats[i]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([feat_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_i = feats[i]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([feat_i, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = torch.cat([out_l[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))

            out = self.lrelu(self.dconv1(self.upconv1(out)))
            out = self.lrelu(self.dconv2(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = x_i  # same size with output
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)
