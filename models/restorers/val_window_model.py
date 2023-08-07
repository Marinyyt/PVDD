from copy import deepcopy

import imageio
import os
import cv2
from .video_base_model import VideoBaseModel
from ..utils.util import requires_grad
from utils.registry import MODEL_REGISTRY
import numpy as np
import torch

def rgb2yuv(rgb):
    rgb_ = rgb.transpose(1, 3)
    A = torch.tensor([[0.299, -0.14714119, 0.61497538], [0.587, -0.28886916, -0.51496512], [0.114, 0.43601035, -0.10001026]], dtype = torch.float32).cuda()
    yuv = torch.tensordot(rgb_, A, 1).transpose(1, 3)
    return yuv

def yuv2rgb(yuv):
    yuv_ = yuv.transpose(1, 3)
    A = torch.tensor([[1, 1, 1], [0, -0.39465, 2.03211], [1.13983, -0.58060, 0]], dtype = torch.float32).cuda()
    rgb = torch.tensordot(yuv_, A, 1).transpose(1, 3)
    return rgb

def resize_img(img_path, patch_size):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2YUV) / 255.0
        h, w, c = img.shape
        img = torch.from_numpy(img.astype('float32')).permute(2, 0, 1).unsqueeze(0)
        # patch_size = self.config.DatasetConfig.get('crop_size', 512)
        scale_img = torch.nn.functional.interpolate(img, size = [patch_size, patch_size], mode = 'bilinear')
        return scale_img[:, 0:1, :, :].cuda()
@MODEL_REGISTRY.register()
class ValWindowModel(VideoBaseModel):
    def __init__(self, config):
        super(ValWindowModel, self).__init__(config)

        self.g_net.cuda().eval()
        self.save_path = self.config.get('val_path', None)
        self.global_idx = 0

        ref_img_path = '/mnt/lustre/yuyitong/portrait_restoration/Test/Standard_Reference/face.png'
        self.ref_y = resize_img(ref_img_path, self.config.get('crop_size', 1024))
    def reduce_loss(self, loss_dict):
        """
        Used after `forward_loss_func`
        """
        loss_sum = 0.
        for k, v in loss_dict.items():
            loss_sum += v
            loss_dict[k] = round(v.mean().item(), 5)

        return loss_sum, loss_dict

    def train_step(self, data):
        """
        Args:
            data: a batch of data
        """
        requires_grad(self.g_net, False)
        if self.global_idx > 100:
            raise print('over')
        b, n, c, h, w = data['gt'].shape
        center_idx = n // 2

        lq_yuv = rgb2yuv(data['lq'][:, center_idx, :, :, :])
        output = self.g_net(lq_yuv[:, 0:1, :, :], self.ref_y)
        lq_yuv[:, 0:1, :, :] = output
        output = yuv2rgb(lq_yuv)

        lq = data['lq'][0, center_idx, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)
        res = output[0].detach().cpu().numpy().transpose(1, 2, 0)
        gt = data['gt'][0, center_idx, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)

        imageio.imsave(os.path.join(self.save_path, '{:04d}_lq.png'.format(self.global_idx)), (lq * 255).astype(np.uint8))
        imageio.imsave(os.path.join(self.save_path, '{:04d}_res.png'.format(self.global_idx)), (res * 255).astype(np.uint8))
        imageio.imsave(os.path.join(self.save_path, '{:04d}_gt.png'.format(self.global_idx)), (gt * 255).astype(np.uint8))

        # data reshape

        data['output'] = output.view(b, c, h, w)
        data['gt'] = data['gt'][:, center_idx, :, :, :].view(b, c, h, w)
        # data['gt'] = data['gt'].view(-1, c, h, w)
        data['lq'] = data['lq'].view(-1, c, h, w)



        # training GNet
        g_loss_dict = dict()
        # discriminator training
        d_loss_dict = dict()

        self.global_idx += 1
        return data, {**g_loss_dict, **d_loss_dict}
