import torchvision.utils as t_utils
import torch
import os
import os.path as osp

from .dist_parallel import master_only


class VisualBoard:

    def __init__(self, log_path, use_pavi=True):
        self._mkdir(log_path)
        self._create_writer(log_path, use_pavi)
    
    @master_only
    def _mkdir(self, log_path):
        if not osp.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

    @master_only
    def _create_writer(self, log_path, use_pavi):
        if use_pavi:
            try:
                from pavi2 import SummaryWriter
                self.writer = SummaryWriter(log_path, thread_safe_mode=True)
            except ImportError:
                from tensorboardX import SummaryWriter
                self.writer = SummaryWriter(log_path)
        else:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(log_path)

    @master_only
    def visual_image(self, tag, images, iters, normalize=False):
        if len(images) > 0:
            image_show = t_utils.make_grid(torch.cat(images, dim=0),
                                        nrow=images[0].size()[0],
                                        normalize=normalize,
                                        range=(-1, 1))
            self.writer.add_image(tag, image_show, iters)
    
    @master_only
    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)

    @master_only
    def add_scalar(self, tag, scalar, global_step=None):
        self.writer.add_scalar(tag, scalar, global_step)

    @master_only
    def close(self):
        self.writer.close()
