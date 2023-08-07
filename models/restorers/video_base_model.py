from copy import deepcopy
import os.path as osp

import torch

from .basicsr_model import BasicSRModel
from utils.registry import MODEL_REGISTRY
from utils.dist_parallel import master_only

from models.optimizers import build_optimizer


@MODEL_REGISTRY.register()
class VideoBaseModel(BasicSRModel):

    def metric(self, data, output_dict, loss_dict=None, mode='val'):
        return 0.

    def log_image(self, writer, data, output_dict, mode='train', global_step=None):
        images = [data['lq'][0, 0, :, :, :], data['output'], data['gt']]
        writer.writer.add_images('lq', data['lq'][0, 0, :, :, :], global_step, dataformats = 'CHW')
        writer.writer.add_images('output', data['output'][0, 0, :, :, :], global_step, dataformats = 'CHW')
        # writer.visual_image('video_vis', images, global_step, normalize=True)
