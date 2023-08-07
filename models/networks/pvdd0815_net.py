import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import BaseNet
from .spynet import SpyNet
from ..utils.warp import flow_warp
from ..utils.util import make_layer
from utils.registry import NETWORK_REGISTRY
from models.modules.layers import GELU, trunc_normal_, Mlp, Mlp02, Mlp03, Mlp04, get_window_size, window_partition, window_reverse, VSTSREncoderTransformerBlock, EncoderLayer
from models.networks.realbasicvsr_net import ResUNet
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


class WindowAttention3D02(nn.Module):
    """Window based multi-head self/cross attention (W-MSA/W-MCA) module with relative
    position bias.
    It supports both of shifted and non-shifted window.
    """
    def __init__(self, dim, num_frames_q, num_frames_kv, window_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        """Initialization function.
        Args:
            dim (int): Number of input channels.
            num_frames (int): Number of input frames.
            window_size (tuple[int]): The size of the window.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            attn_drop (float, optional): Dropout ratio of attention weight. Defaults to 0.0
            proj_drop (float, optional): Dropout ratio of output. Defaults to 0.0
        """
        super().__init__()
        self.dim = dim
        self.num_frames_q = num_frames_q # D1
        self.num_frames_kv = num_frames_kv # D2
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads # nH
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * num_frames_q - 1) * (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*D-1 * 2*Wh-1 * 2*Ww-1, nH

        # Get pair-wise relative position index for each token inside the window
        coords_d_q = torch.arange(self.num_frames_q)
        coords_d_kv = torch.arange(0, self.num_frames_q, int((self.num_frames_q + 1) // self.num_frames_kv))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_q = torch.stack(torch.meshgrid([coords_d_q, coords_h, coords_w]))  # 3, D1, Wh, Ww
        coords_kv = torch.stack(torch.meshgrid([coords_d_kv, coords_h, coords_w]))  # 3, D2, Wh, Ww
        coords_q_flatten = torch.flatten(coords_q, 1)  # 3, D1*Wh*Ww
        coords_kv_flatten = torch.flatten(coords_kv, 1)  # 3, D2*Wh*Ww
        relative_coords = coords_q_flatten[:, :, None] - coords_kv_flatten[:, None, :]  # 3, D1*Wh*Ww, D2*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # D1*Wh*Ww, D2*Wh*Ww, 3
        relative_coords[:, :, 0] += self.num_frames_q - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[0] - 1
        relative_coords[:, :, 2] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # D1*Wh*Ww, D2*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, kv=None, mask=None):
        """Forward function.
        Args:
            q (torch.Tensor): (B*nW, D1*Wh*Ww, C)
            kv (torch.Tensor): (B*nW, D2*Wh*Ww, C). Defaults to None.
            mask (torch.Tensor, optional): Mask for shifted window attention (nW, D1*Wh*Ww, D2*Wh*Ww). Defaults to None.
        Returns:
            torch.Tensor: (B*nW, D1*Wh*Ww, C)
        """
        kv = q if kv is None else kv
        B_, N1, C = q.shape # N1 = D1*Wh*Ww, B_ = B*nW
        B_, N2, C = kv.shape # N2 = D2*Wh*Ww, B_ = B*nW
        q_copy = q
        q = self.q(q).reshape(B_, N1, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(kv).reshape(B_, N2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1] # B_, nH, N1(2), C
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # B_, nH, N1, N2

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N1, N2, -1)  # D1*Wh*Ww, D2*Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, D1*Wh*Ww, D2*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, D1*Wh*Ww, D2*Wh*Ww

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N1, N2) + mask.unsqueeze(1).unsqueeze(0) # B, nW, nH, D1*Wh*Ww, D2*Wh*Ww
            attn = attn.view(-1, self.num_heads, N1, N2)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x + q_copy, attn


class VSTSREncoderTransformerBlock02(nn.Module):
    """Video spatial-temporal super-resolution encoder transformer block.
    """

    def __init__(self, dim, num_heads, num_frames=4, window_size=(8, 8),
                 shift_size=(0, 0), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=GELU,
                 norm_layer=nn.LayerNorm, mlp = None):
        """Initialization function.
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            num_frames (int): Number of input frames.
            window_size (tuple[int], optional): Window size. Defaults to 8.
            shift_size (tuple[int], optional): Shift size. Defaults to 0.
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop (float, optional): Dropout rate. Defaults to 0.
            attn_drop (float, optional): Attention dropout rate. Defaults to 0.
            drop_path (float, optional):  Stochastic depth rate. Defaults to 0.
            act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-win_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D02(
            dim, num_frames_q=1, num_frames_kv=1,
            window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop
        )

        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if mlp == '02':
            self.mlp = Mlp02(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif mlp == '03':
            self.mlp = Mlp03(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif mlp == '04':
            self.mlp = Mlp04(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix):
        """Forward function.
        Args:
            x (torch.Tensor): (B, D, H, W, C)
            mask_matrix (torch.Tensor): (nW*B, D*Wh*Ww, D*Wh*Ww)
        Returns:
            torch.Tensor: (B, D, H, W, C)
        """
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        shortcut = x[:, 0]
        x = self.norm1(x)

        # Padding
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0))
        _, _, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # nW*B, D, window_size, window_size, C
        # x_windows = x_windows.view(-1, D * window_size[0] * window_size[1], C)  # nW*B, D*window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows[:, 0].view(-1, window_size[0] * window_size[1], C), x_windows[:, 1].view(-1, window_size[0] * window_size[1], C), mask=attn_mask)[0]  # nW*B, D*window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, 1, window_size[0], window_size[1], C)
        shifted_x = window_reverse(attn_windows, window_size, B, 1, Hp, Wp)  # B, D, H, W, C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W, :].contiguous()

        # FFN
        x = shortcut + self.drop_path(x)
        x = (x + self.drop_path(self.mlp(self.norm2(x)))).squeeze(1)
        return x


class EncoderLayer02(nn.Module):
    def __init__(self, dim, depth, num_heads, num_frames, window_size=(8, 8),
                 mlp_ratio=4., mlp = None, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        """Encoder layer
        Args:
            dim (int): Number of feature channels
            depth (int): Depths of this stage.
            num_heads (int): Number of attention head.
            num_frames (int]): Number of input frames.
            window_size (tuple[int], optional): Window size. Defaults to (8, 8).
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop (float, optional): Dropout rate. Defaults to 0.
            attn_drop (float, optional): Attention dropout rate. Defaults to 0.
            drop_path (float, optional): Stochastic depth rate. Defaults to 0.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        """
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth

        # Build blocks
        self.blocks = nn.ModuleList([
            VSTSREncoderTransformerBlock(dim=dim, num_heads=num_heads,
            num_frames=num_frames,window_size=window_size,
            shift_size=(0, 0) if (i % 2 == 0) else self.shift_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer, mlp=mlp) for i in range(depth)])

        self.last_blk = VSTSREncoderTransformerBlock02(dim=dim, num_heads=num_heads,
            num_frames=num_frames, window_size=window_size,
            shift_size=(0, 0),
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop,
            norm_layer=norm_layer, mlp = mlp)
    def forward(self, x):
        """Forward function.
        Args:
            x (torch.Tensor): (B, D, C, H, W)
        Returns:
            torch.Tensor: (B, D, C, H, W)
        """
        B, D, C, H, W = x.shape
        x = x.permute(0, 1, 3, 4, 2) # B, D, H, W, C

        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        Hp = int(np.ceil(H / window_size[0])) * window_size[0]
        Wp = int(np.ceil(W / window_size[1])) * window_size[1]

        img_mask = torch.zeros((1, D, Hp, Wp, 1), device=x.device) # 1, D, H, W, 1
        h_slices = (slice(0, -window_size[0]),
                    slice(-window_size[0], -shift_size[0]),
                    slice(-shift_size[0], None))
        w_slices = (slice(0, -window_size[1]),
                    slice(-window_size[1], -shift_size[1]),
                    slice(-shift_size[1], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size) # nW, D, Wh, Ww, 1
        mask_windows = mask_windows.view(-1, D * window_size[0] * window_size[1]) # nW, D*Wh*Ww
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # nW, D*Wh*Ww, D*Wh*Ww
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.last_blk(x, attn_mask)
        x = x.permute(0, 3, 1, 2) # B, D, C, H, W

        return x

# 将q和kv的输入区分开的实验 & 更改mlp & 加入spatial trans
@NETWORK_REGISTRY.register()
class pvdd0815(BaseNet):
    """A recurrent network for video restoration
    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_in = 3, num_feat=64, num_block=1, num_block_f=5, num_block_pre = 3, spynet_path=None,
                 dynamic_refine_thres=255., is_sequential_cleaning=False,
                 depth_pre = 1, depth = 8, num_head = 4, num_frames = 2, window_size_pre = (8, 8), window_size = (4, 4), mlp_ratio = 2., mlp = None, qkv_bias = True,
                 qk_scale = None, drop_rate = 0., attn_drop_rate = 0., drop_path_rate = 0.,
                drop_path = 0., norm_layer = nn.LayerNorm):
        super(pvdd0815, self).__init__()
        self.num_feat = num_feat

        # feat extractor
        self.feat_extractor = ConvResidualBlocks(num_in, num_feat, num_block_f)
        self.feat_STTB = EncoderLayer(
            dim=num_feat,
            depth=depth_pre, num_heads=num_head,
            num_frames=1, window_size=window_size_pre, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path,
            norm_layer=norm_layer,
            mlp = mlp
        )

        # pre cleaning module
        self.dynamic_refine_thres = dynamic_refine_thres / 255.
        self.is_sequential_cleaning = is_sequential_cleaning
        self.clean_model = ResUNet(input_nc=num_in, depth=2, num_feat=num_feat, num_block=num_block_pre)

        # propagation
        self.backward_STTB = EncoderLayer02(
            dim=num_feat,
            depth=depth, num_heads=num_head,
            num_frames=num_frames, window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path,
            norm_layer=norm_layer,
            mlp = mlp
        )

        self.forward_STTB = EncoderLayer02(
            dim=num_feat,
            depth=depth, num_heads=num_head,
            num_frames=num_frames, window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path,
            norm_layer=norm_layer,
            mlp = mlp
        )

        self.backward_trunk = ResidualBlocksWithInConv(num_feat, num_feat, num_block)
        self.forward_trunk = ResidualBlocksWithInConv(num_feat, num_feat, num_block)

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

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def process(self, x):
        """Forward function of BasicVSR.
        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        b, n, c, h, w = x.size()

        # spatial feature extractor
        feats_ = self.feat_extractor(x.view(-1, c, h, w)) # down scale 4.
        h_, w_ = feats_.shape[2:]
        feats_ = feats_.view(b, n, self.num_feat, h_, w_)
        # feats_ = self.feat_STTB(feats_).view(b, n, -1, h_, w_)
        feats = [feats_[:, i, :, :, :] for i in range(0, n)]


        # temporal backward branch
        out_l = []
        feat_prop = feats[-1]
        for i in range(n - 1, -1, -1):
            feat_i = feats[i]
            feat_prop = torch.stack([feat_i, feat_prop], dim=1) # b, 2, num_feat, h_, w_
            feat_prop = self.backward_STTB(feat_prop)
            # feat_prop = torch.cat([feat_prop[:, 0, :, :, :], feat_prop[:, 1, :, :, :]], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # temporal forward branch
        feat_prop = feats[0]
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_i = feats[i]
            feat_prop = torch.stack([feat_i, feat_prop], dim=1)
            feat_prop = self.forward_STTB(feat_prop)
            # feat_prop = torch.cat([feat_prop[:, 0, :, :, :], feat_prop[:, 1, :, :, :]], dim=1)
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

    def forward(self, x):
        # x = x.unsqueeze(1).repeat(1, 5, 1, 1, 1)
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

        # forward_process
        outputs = self.process(x)
        return outputs
