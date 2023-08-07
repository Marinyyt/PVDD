import importlib
from os import path as osp
import os
from copy import deepcopy
import torch.nn as nn


from .base import LossBase, GANLossBase, DistillLossBase
from .loss import VGGLossBase
from utils.registry import LOSS_REGISTRY


# register pytorch internal loss
for k, v in vars(nn).items():
    if 'loss' in k.lower():
        LOSS_REGISTRY.register(getattr(nn, k))


# automatically scan and import loss modules for registry
# scan all the files under the 'losses' folder and collect files ending with
# 'loss.py'
loss_folder = osp.dirname(osp.abspath(__file__))
loss_filenames = [osp.splitext(osp.basename(v))[0] for v in os.listdir(loss_folder) if v.endswith('loss.py')]
# import all the loss modules
_loss_modules = [importlib.import_module(f'models.losses.{file_name}') for file_name in loss_filenames]


def build_loss(opt):
    """Build loss from config.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Loss type.
    """
    opt = deepcopy(opt)

    if hasattr(opt, 'args'):
        loss = LOSS_REGISTRY.get(opt['type'])(**opt.args)
    else:
        loss = LOSS_REGISTRY.get(opt['type'])()
    
    print(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
