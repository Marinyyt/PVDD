from copy import deepcopy

from models.networks import build_network

import torch.nn as nn
import torch.nn.functional as F
from .video_base_model import VideoBaseModel
from .basicvsr_model import BasicVSRModel
from .. import losses as L
from ..utils.util import requires_grad
from utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class RealVSRModel(BasicVSRModel):   
    def log_image(self, writer, data, output_dict, mode='train', global_step=None):
        images = [data['lq'], data['output_clean'], data['output'], data['gt']]
        writer.visual_image('video_vis', images, global_step, normalize=True)
        
    def train_step(self, data):
        """
        Args:
            data: a batch of data
        """
        requires_grad(self.g_net, True)
        if self.use_gan:
            requires_grad(self.d_net, False)
        # real vsrnet with pre-clean model
        to_rgb = self.config.Network.GNet.args.get('to_rgb')
        if not to_rgb:
            output, output_clean = self.g_net(data['lq'], return_clean=True)
        else:
            output, output_clean, output_rgbs = self.g_net(data['lq'], return_clean=True)
            
        if to_rgb:
            gt_rgbs = []
            for i in range(len(output_rgbs)):
                gt_rgbs.append(F.interpolate(data['gt'].view(-1, *data['gt'].shape[2:]), scale_factor=1/2**(i+1)))

        # data reshape
        c, h, w = data['gt'].shape[2:]
        data['output'] = output.view(-1, c, h, w)
        data['output_clean'] = output_clean
        data['gt'] = data['gt'].view(-1, c, h, w)
        data['lq'] = data['lq'].view(-1, c, h, w)

        if self.pixel_loss_config.get('FeatMatchLoss'):
            fake_pred, fake_feats = self.d_net(data['output'])
            real_pred, real_feats = self.d_net(data['gt'])
        else:
            fake_pred = self.d_net(data['output'])
            real_pred = self.d_net(data['gt'])

        # training GNet
        g_loss_dict = dict()
        for key, v in self.pixel_loss_config.items():
            if key == 'FeatMatchLoss':
                loss_item = getattr(self, key)(fake_feats, real_feats) * v.weight
            else:
                loss_item = getattr(self, key)(data['output'], data['gt']) * v.weight
            # clean loss
            if isinstance(getattr(self, key), nn.L1Loss):
                g_loss_dict['clean_loss'] = getattr(self, key)(data['output_clean'], data['gt']) * v.weight
                if to_rgb:
                    multi_clean_loss = 0.
                    for out_rgb, gt_rgb in zip(output_rgbs, gt_rgbs[::-1]):
                        multi_clean_loss += getattr(self, key)(out_rgb, gt_rgb)
                    g_loss_dict['multi_clean_loss'] = multi_clean_loss
            g_loss_dict[key] = loss_item
        # add gan loss
        if self.use_gan:
            gan_loss = self.gan_loss(fake_pred, real_pred, flag='G') * self.gan_loss_config.weight
            g_loss_dict['gan_loss'] = gan_loss
        g_loss, g_loss_dict = self.reduce_loss(g_loss_dict)
        # loss backward and update weights
        self.optimizer_g.zero_grad()
        g_loss.backward()
        self.optimizer_g.step()
        
        self.accumulate()
        # discriminator training
        d_loss_dict = dict()
        if self.use_gan:
            requires_grad(self.d_net, True)
            requires_grad(self.g_net, False)

            fake_pred = self.d_net(data['output'].detach())
            real_pred = self.d_net(data['gt'])
            d_loss = self.gan_loss(fake_pred, real_pred, flag='D')
            d_loss_dict['d_loss'] = d_loss.item()

            self.optimizer_d.zero_grad()
            d_loss.backward()
            self.optimizer_d.step()

        return data, {**g_loss_dict, **d_loss_dict}
