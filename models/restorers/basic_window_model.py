from copy import deepcopy
import torch
import imageio

from .video_base_model import VideoBaseModel
from ..utils.util import requires_grad, normalize_augment
from utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class BasicWindowModel(VideoBaseModel):

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
        requires_grad(self.d_net, True)
        if self.use_gan:
            requires_grad(self.d_net, False)
        # data['lq'], data['gt'] = normalize_augment(data['lq'], data['gt'])
        b, n, c, h, w = data['lq'].size()
        temp = torch.zeros((b, n-6, c, h, w)).cuda()
        for i in range(3, 10):
            temp[:, i-3, :, :, :] = self.g_net(data['lq'][:, i-3:i+4, :, :, :], 1)
        output = self.d_net(data['lq'][:, 3:10, :, :, :], temp)
        if False:
            lq_img = data['lq'][0, 2, :, :, :].detach().cpu().numpy().transpose(1, 2, 0) * 255
            gt_img = data['gt'][0, 2, :, :, :].detach().cpu().numpy().transpose(1, 2, 0) * 255
            imageio.imwrite('/home/SENSETIME/yuyitong/code/portrait_restoration/lq_img.png', lq_img)
            imageio.imwrite('/home/SENSETIME/yuyitong/code/portrait_restoration/gt_img.png', gt_img)

        # data reshape
        b, n, c, h, w = data['gt'].shape
        center_idx = n // 2
        data['output'] = output.view(b, c, h, w)
        data['gt'] = data['gt'][:, center_idx, :, :, :].view(b, c, h, w)
        # data['gt'] = data['gt'].view(-1, c, h, w)
        data['lq'] = data['lq'].view(-1, c, h, w)

        # if self.pixel_loss_config.get('FeatMatchLoss'):
        #     fake_pred, fake_feats = self.d_net(data['output'])
        #     if '3D' in self.config['Network']['DNet']['type']:
        #         real_pred, real_feats = self.d_net(data['gt'])
        #     else:
        #         read_pred, real_feats = self.d_net(data['gt'])
        # else:
        #    if '3D' in self.config['Network']['DNet']['type']:          
        #         fake_pred = self.d_net(data['output'])
        #         real_pred = self.d_net(data['gt'])
        #    else:
        #         fake_pred = self.d_net(data['output'])
        #         real_pred = self.d_net(data['gt'])
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
