import torch
import torch.nn as nn

import os
import os.path as osp

from models.utils.vggface import VGG19


class LossBase(nn.Module):

    def __init__(self) -> None:
        super(LossBase, self).__init__()


class VGGLossBase(LossBase):

    def __init__(self) -> None:
        super(VGGLossBase, self).__init__()
        _pretrain_model = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), 
                          'pretrain_models', 
                          'checkpoints', 'vgg19-dcbb9e9d.pth')
        self.vgg = VGG19(_pretrain_model)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1))

    def _normalization(self, img):
        return (img - self.mean) / self.std


class DistillLossBase(nn.Module):

    def __init__(self):
        super(DistillLossBase, self).__init__()


class GANLossBase(nn.Module):

    def __init__(self):
        super(GANLossBase, self).__init__()