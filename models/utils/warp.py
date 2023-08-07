import torch
import torch.nn as nn
import torch.nn.functional as F
from ..networks.warpnet import Warpnet_512


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.
    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).cuda()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)

    return output


def warp_fn(ref_image, grid):
    # warp image
    grid = grid.permute((0, 2, 3, 1))
    grid_no_grad = grid.clone()
    grid_no_grad[:, :, :, 0], grid_no_grad[:, :, :, 1] = grid[:, :, :, 1], grid[:, :, :, 0]
    warp_image = F.grid_sample(ref_image, grid_no_grad)
    # transform to the original order
    # grid = grid.permute((0, 3, 1, 2))
    return warp_image


class WarpModel:

    def __init__(self, warp_type, pretrain_model=None) -> None:
        # warp_type: warpnet | flow | no_warp
        self.warp_type = warp_type.lower()
        if self.warp_type.lower() == "warpnet":
            self.warpnet = Warpnet_512()
            self.warpnet.load_pretrain_model(pretrain_model)
            # self.warpnet.load_state_dict(torch.load())
            self.warpnet.eval()
    
    def __call__(self, d_img_y, ref_img_y, d_img_rgb=None, ref_img_rgb=None):
        """
        if warp_type is 'warpnet':
            d_img_rgb, ref_img_rgb: torch.Tensor, [NCHW], 使用warpnet的时候需要, warpnet需要rgb输入
            d_img_y, ref_img_y: np.array or torch.Tensor, ref_img_y是需要warp的y通道数据
        """
        if self.warp_type == "warpnet":
            with torch.no_grad():
                grid, _ = self.warpnet(d_img_rgb, ref_img_rgb)
            return warp_fn(ref_img_y, grid)
        elif self.warp_type == "flow":
            # TODO
            pass
        elif self.warp_type == "no_warp":
            return ref_img_y

    def wrap(self, parallel):
        if self.warp_type.lower() == "warpnet":
            self.warpnet = parallel.wrapper(self.warpnet)


