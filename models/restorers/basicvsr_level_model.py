from copy import deepcopy
import torch
from .video_base_model import VideoBaseModel
from ..utils.util import requires_grad
from utils.registry import MODEL_REGISTRY
from models.metrics.metric import psnr, ssim
import time
@MODEL_REGISTRY.register()
class BasicVSRLevelModel(VideoBaseModel):

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
        requires_grad(self.g_net, True)
        if self.use_gan:
            requires_grad(self.d_net, False)
        b, n, c, h, w = data['gt'].size()
        nl = data['noiselevel']
        nl = nl.view(b, 1, 1, 1) 
        noise_map = nl.cuda(non_blocking=True).repeat(1, 1, h, w)
        output = self.g_net(data['lq'], noise_map)

        # data reshape
        data['output'] = output.view(-1, c, h, w)
        data['gt'] = data['gt'].view(-1, c, h, w)
        data['lq'] = data['lq'].view(-1, c, h, w)
        if self.use_gan:
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
            elif key == 'EMVDLoss':
                loss_item = getattr(self, key)(self.g_net) * v.weight
            else:
                loss_item = getattr(self, key)(data['output'], data['gt']) * v.weight
            g_loss_dict[key] = loss_item
        # add gan loss
        if self.use_gan:
            gan_loss = self.gan_loss(fake_pred, real_pred, 'G') * self.gan_loss_config.weight
            g_loss_dict['gan_loss'] = gan_loss
        g_loss, g_loss_dict = self.reduce_loss(g_loss_dict)
        # loss backward and update weights
        self.optimizer_g.zero_grad()
        g_loss.backward(retain_graph = True)
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
            d_loss.backward(retain_graph = True)
            self.optimizer_d.step()
 
        data['output'] = output.view(b, n, c, h, w)
        data['gt'] = data['gt'].view(b, n, c, h, w)
        data['lq'] = data['lq'].view(b, n, c, h, w)

        return data, {**g_loss_dict, **d_loss_dict}

    def metric(self, data, output_dict, mode='val'):
        if mode == 'val':
            n, c, h, w = data['gt'].size()
            psnrs, ssims = 0, 0
            for i in range(n):
                psnrs += psnr(data['output'][i], data['gt'][i], pixel_max_cnt = 1.0)
                ssims += ssim(data['output'][i], data['gt'][i])
            psnrs = psnrs / n
            ssims = ssims / n
            return {'psnr': psnrs, 'ssim': ssims}
        else:
            return 0.
    def val_step(self, data):
        requires_grad(self.g_net, False)
        b, n, c, h, w = data['gt'].size()
        nl = data['noiselevel']
        nl = nl.view(b, 1, 1, 1)
        noise_map = nl.cuda(non_blocking=True).repeat(1, 1, h, w)
        output = self.g_net(data['lq'], noise_map)
        # data reshape
        data['output'] = output.view(b, n, c, h, w)[0]
        data['gt'] = data['gt'].view(b, n, c, h, w)[0]
        data['lq'] = data['lq'].view(b, n, c, h, w)[0]

        return data

