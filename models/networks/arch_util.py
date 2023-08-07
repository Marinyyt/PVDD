import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class ResidualBlock_noBN22(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN22, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        # initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class DenseBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(DenseBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # initialization
        # initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x = self.lrelu(self.conv1(x), inplace=True)
        x = self.lrelu(self.conv2(x), inplace=True)
        x = self.lrelu(self.conv3(x), inplace=True)
        x = self.lrelu(self.conv4(x), inplace=True)
        x = self.lrelu(self.conv5(x), inplace=True)
        return x


class DenseBlock_noBN2(nn.Module):
    def __init__(self, nf=128, bias=True):
        super(DenseBlock_noBN2, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + nf, nf, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * nf, nf, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * nf, nf, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * nf, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5 + x


class ResidualBlock_noBN_LFB(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64, epison1=1.0, epison2=1.0):
        super(ResidualBlock_noBN_LFB, self).__init__()
        self.res1 = ResidualBlock_noBN(nf=nf)
        self.res2 = ResidualBlock_noBN(nf=nf)
        self.conv1 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.ep1 = epison1
        self.ep2 = epison2
        initialize_weights([self.res1, self.res2, self.conv1], 0.1)

    def forward(self, x):
        identity = x
        res1 = self.res1(x)
        res2 = self.res2(res1)
        res3 = torch.cat([res1, res2], dim=1)
        res3 = self.conv1(res3)
        out = identity * self.ep1 + res3 * self.ep2
        return out


import torch.nn.utils.weight_norm as wn
class ResidualBlock_noBN_wn(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN_wn, self).__init__()
        self.conv1 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        self.conv2 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class ResidualBlock_noBN_3d(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN_3d, self).__init__()
        self.conv1 = nn.Conv3d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv3d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class ResidualBlock_BN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_BN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv1_bn = nn.BatchNorm2d(nf)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2_bn = nn.BatchNorm2d(nf)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv1_bn(out)
        out = self.conv2(out)
        out = self.conv2_bn(out)
        return identity + out


class ResidualBlock_BN2(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_BN2, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv1_bn = nn.BatchNorm2d(nf)
        self.conv1_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        '''
        self.conv2 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
        self.conv2_bn = nn.BatchNorm2d(nf)
        self.conv2_2 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
        '''
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(self.conv1_bn(x)))
        out = self.conv1_2(out)
        identity = identity + out
        '''
        out2 = F.relu(self.conv2(self.conv2_bn(identity)))
        out2 = self.conv2_2(out2)
        identity = identity + out2
        '''
        return identity


class ResidualBlock_IN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_IN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv1_bn = nn.InstanceNorm2d(nf)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2_bn = nn.InstanceNorm2d(nf)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv1_bn(out)
        out = self.conv2(out)
        out = self.conv2_bn(out)
        return identity + out


class ResidualBlock_IN2(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_IN2, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv1_bn = nn.InstanceNorm2d(nf)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2_bn = nn.InstanceNorm2d(nf)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.lrelu(self.conv1_bn(self.conv1(x)))
        out = self.conv2_bn(self.conv2(out))
        return identity + out


class ConvBlock_IN(nn.Module):
    def __init__(self, nf=64):
        super(ConvBlock_IN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv1_bn = nn.InstanceNorm2d(nf)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2_bn = nn.InstanceNorm2d(nf)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv1_bn(out)
        out = F.relu(self.conv2(out), inplace=True)
        out = self.conv2_bn(out)
        return out


class ResidualBlock_BN_3d(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_BN_3d, self).__init__()
        self.conv1 = nn.Conv3d(nf, nf, 3, 1, 1, bias=True)
        self.conv1_bn = nn.BatchNorm3d(nf)
        self.conv2 = nn.Conv3d(nf, nf, 3, 1, 1, bias=True)
        self.conv2_bn = nn.BatchNorm3d(nf)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv1_bn(out)
        out = self.conv2(out)
        out = self.conv2_bn(out)
        return identity + out



def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output
