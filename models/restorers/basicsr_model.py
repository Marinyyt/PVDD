from copy import deepcopy
import os.path as osp

import torch
from torch.nn.parallel import DistributedDataParallel, DataParallel

from .base import BaseModel
from .. import losses as L
from utils.registry import MODEL_REGISTRY
from models.networks import build_network
from utils.dist_parallel import master_only

from models.optimizers import build_optimizer


@MODEL_REGISTRY.register()
class BasicSRModel(BaseModel):

    def init_network(self, network_config):
        self.g_net = build_network(network_config.GNet)
        self.g_net_ema = deepcopy(self.g_net)
        self.d_net = build_network(network_config.DNet)
        self.g_net.train()
        self.g_net_ema.eval()
        self.d_net.train()

    def init_loss(self, loss_config):
        self.pixel_loss_config = loss_config.PixelLoss
        self.gan_loss_config = loss_config.get('GANLoss')
        self.use_gan = True if loss_config.get('GANLoss') is not None else False
        # build gan loss
        if self.use_gan:
            setattr(self, 'gan_loss', L.build_loss(loss_config.GANLoss))
        # build pixel loss
        for k, v in loss_config.PixelLoss.items():
            func = L.build_loss(v)
            setattr(self, k, func)

    # cast data to target device
    def before_train_step(self, data, augment_func=None):
        return {k: v.to(self.device) for k, v in data.items()}

    def init_optimizer(self, optim_config):
        self.optimizer_g = build_optimizer(self.g_net.parameters(), optim_config.OptimG)
        self.optimizer_d = build_optimizer(self.d_net.parameters(), optim_config.OptimD)
    def forward_loss_func(self, data, mode='train'):
        pass

    def metric(self, data, output_dict, loss_dict=None, mode='val'):
        # TODO add metric for model selection or monitoring the experiment
        return 0.

    def resume(self):
        # TODO add training state for resume training
        g_net_path = self.config.Network.GNet.get('resume_path', None)
        d_net_path = self.config.Network.DNet.get('resume_path', None)

        self.g_net.load_pretrain_model(g_net_path)
        self.d_net.load_pretrain_model(d_net_path)
    
    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net    

    def accumulate(self, decay=0.999):
        """
        apply exponential moving average on generator
        """
        g_net = self.get_bare_model(self.g_net)
        g_net_ema = self.get_bare_model(self.g_net_ema)
        par1 = dict(g_net_ema.named_parameters())
        par2 = dict(g_net.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

    @master_only
    def save(self, save_dir, epoch=None, iter=None):
        g_net_save_path = osp.join(save_dir, 'g_net_{}.pth'.format(epoch if epoch is not None else iter))
        self._save_model(self.g_net, g_net_save_path)
        self._save_model(self.g_net_ema, g_net_save_path.replace('g_net', 'g_net_ema'))
        self._save_model(self.d_net, g_net_save_path.replace('g_net', 'd_net'))

    def log_loss_metric(self, writer, loss_dict, metric_dict, mode='train', global_step=None):
        # record loss values
        # 暂时将R1 regularization loss放到pixel loss中
        if mode == 'train':
            pixel_loss = {}
            for name, value in loss_dict.items():
                if name not in ['gan_loss', 'd_loss']:
                    pixel_loss[name] = loss_dict[name]
            if self.use_gan:
                writer.add_scalar('gan_loss', loss_dict['gan_loss'], global_step)
                writer.add_scalar('d_loss', loss_dict['d_loss'], global_step)
            writer.add_scalars("reg_loss", pixel_loss, global_step)
        elif mode == 'val':
            for name, value in metric_dict.items():
                writer.add_scalar(name, value, global_step)
    def log_image(self, writer, data, output_dict, mode='train', global_step=None):
        images = [output_dict['d_image_y'], output_dict['fake_image_y'], output_dict['gt_image_y']]
        writer.visual_image('image show', images, global_step, normalize=True)

    def forward_net(self):
        pass

    def init_scheduler(self, a):
        pass

    def optimize_param(self):
        pass
