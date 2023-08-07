from copy import deepcopy
import os.path as osp
from functools import partial

import torch

from .base import GANBaseModel
from utils.registry import MODEL_REGISTRY
from utils.dist_parallel import Parallel
from utils.visualboard import VisualBoard
from models.networks import build_network
from models.optimizers import build_optimizer
from .. import losses as L

from ..utils.diffaugment import DiffAugment
from ..utils.warp import WarpModel
from ..utils.distill_gt import DistillGT


@MODEL_REGISTRY.register()
class FaseSRModel(GANBaseModel):

    """
    人像修复模型基础训练流程
    """

    def __init__(self, config):
        super(FaseSRModel, self).__init__(config)

        if hasattr(self.config, 'distill_gt'):
            self.distill_gt = True
            self.distillGT = DistillGT(**self.config.distill_gt)
        else:
            self.distill_gt = False


    def init_network(self, network_config):
        self.g_net = build_network(network_config.GNet)
        self.g_net_ema = deepcopy(self.g_net)
        self.d_net = build_network(network_config.DNet)
        self.warp_model = WarpModel(warp_type='warpnet')
        self.g_net.train()
        self.d_net.train()
        self.g_net_ema.eval()

    def init_optimizer(self, optim_config):
        self.optimizer_g = build_optimizer(self.g_net.parameters(), optim_config.OptimG)
        self.optimizer_d = build_optimizer(self.d_net.parameters(), optim_config.OptimD)

    def init_loss(self, loss_config):
        self.pixel_loss_config = loss_config.PixelLoss
        # build pixel loss
        for k, v in self.pixel_loss_config.items():
            func = L.build_loss(v)
            setattr(self, k, func)
        # build gan loss
        if not hasattr(loss_config, 'GANLoss') or loss_config.GANLoss is None:
            self.train_gan = False
        else:
            self.gan_loss_config = loss_config.GANLoss
            self.train_gan = True
        
            for k, v in self.gan_loss_config.items():
                func = L.build_loss(v)
                setattr(self, k, func)
        print("Train GAN: ", self.train_gan)
        
    def resume(self):
        GNet_config = self.config.Network.GNet
        DNet_config = self.config.Network.DNet
        if hasattr(GNet_config, 'resume_path'):
            self.g_net.load_pretrain_model(GNet_config.resume_path)
            self.g_net_ema.load_pretrain_model(GNet_config.resume_path)
        if hasattr(DNet_config, 'resume_path'):
            self.d_net.load_pretrain_model(DNet_config.resume_path)

    def wrap(self, parallel: Parallel):
        super().wrap(parallel)
        self.warp_model.wrap(parallel)
        if self.distill_gt:
            self.distillGT.wrap(parallel)


    def before_train_G_step(self, data, augment_func=None):
        """ 
        get warped images
        augmentation
        """
        for k in data.keys():
            data[k] = data[k].to(self.device)
        
        data['warp_image_y'] = self.warp_model(d_img_y=None, ref_img_y=data['ref_image_y'], \
                                            d_img_rgb=data['d_image'], ref_img_rgb=data['ref_image'])

        # batch数据批量augment
        if augment_func is not None:
            [data['d_image_y'], data['gt_image_y'], data['ref_image_y'], data['warp_image_y']], data = \
                augment_func([data['d_image_y'], data['gt_image_y'], data['ref_image_y'], data['warp_image_y']], data)
        
        if self.distill_gt:
            data['gt_image_y'] = self.distillGT(data['d_image_y'], data['ref_image_y'], data['gt_image_y'], data['scores'])
            
        return data

    def forward_G_net(self, data, mode='train'):
        d_image = data['d_image_y']
        ref_image = data['warp_image_y']
        out = self.g_net(d_image, ref_image)
        return {'rec_image': out}


    def forward_G_loss_func(self, output_dict, data, mode='train'):
        gt_image = data['gt_image_y']
        ref_image = data['warp_image_y']
        rec_image = output_dict['rec_image']
        if self.train_gan:
            real_pred = output_dict['real_pred']
            fake_pred = output_dict['fake_pred']
            real_feat = output_dict['real_feat']
            fake_feat = output_dict['fake_feat']

        losses = {}
        for key, v in self.pixel_loss_config.items():
            loss_fn = getattr(self, key)
            if key == 'FeatMatchLoss':
                if self.train_gan:
                    loss = loss_fn(fake_feat, real_feat) * v.weight
                else:
                    continue
            elif key == "CXLoss":
                loss = loss_fn(rec_image, gt_image, ref_image)
            elif key == "TVRLoss":
                loss = loss_fn(rec_image)
            else:
                loss = loss_fn(rec_image, gt_image) * v.weight
            losses[key] = loss
        
        for k, v in self.gan_loss_config.items():
            loss_fn = getattr(self, k)
            if k == "R1Loss":
                continue
            else:
                loss = loss_fn(fake_pred, real_pred, 'G') * v.weight
            losses[k] = loss
        return losses

    def before_train_D_step(self, data, output_g_data):
        data['rec_image'] = output_g_data['rec_image']
        return data

    def forward_D_net(self, data, mode='train_g'):
        gt_image = data['gt_image_y']
        rec_image = data['rec_image']
        diffaugment = partial(DiffAugment, policy=self.config.get('diffaugment'))
        if mode == 'train_g':
            fake_pred, fake_feat = self.d_net(diffaugment(rec_image))
            real_pred, real_feat = self.d_net(diffaugment(gt_image))
            return {'fake_pred': fake_pred, 'real_pred': real_pred, 'fake_feat': fake_feat, 'real_feat': real_feat}
        elif mode == 'train_d':
            fake_pred, _ = self.d_net(diffaugment(rec_image.detach()))
            real_pred, _ = self.d_net(diffaugment(gt_image))
        return {'fake_pred': fake_pred, 'real_pred': real_pred}

    def forward_D_loss_func(self, output_dict, data, mode='train'):
        real_pred = output_dict['real_pred']
        fake_pred = output_dict['fake_pred']
        losses = {}
        for k, v in self.gan_loss_config.items():
            loss_func = getattr(self, k)
            if k == "R1Loss":
                # TODO
                continue
            else:
                loss = loss_func(fake_pred, real_pred, flag='D') * v.weight
            losses[k] = loss
        return losses

    def optimize_D_param(self, loss_sum):
        self.optimizer_d.zero_grad()
        loss_sum.backward()
        self.optimizer_d.step()

    def optimize_G_param(self, loss_sum):
        self.optimizer_g.zero_grad()
        loss_sum.backward()
        self.optimizer_g.step()


    def train_step(self, runing_iter, data, augment_func=None):
        """
        Train one step

        return:
            output_dict: output of networks
            loss_dict: loss dict
        """
        self.g_net.train()
        self.g_net_ema.train()
        self.d_net.train()

        # G training
        G_data = self.before_train_G_step(data, augment_func)
        output_G_dict = self.forward_G_net(G_data)
        # forward d net for computing feature match loss and GAN loss
        if self.train_gan:
            output_D_dict = self.forward_D_net({**G_data, **output_G_dict}, mode='train_g')
        else:
            output_D_dict = {}

        loss_G_dict = self.forward_G_loss_func({**output_G_dict, **output_D_dict}, data)
        loss_G_sum, loss_G_dict = self.reduce_loss(loss_G_dict)
        self.optimize_G_param(loss_G_sum)

        G_data, output_G_dict = self.after_train_G_step(G_data, output_G_dict)

        # D training
        if self.train_gan:
            D_data = self.before_train_D_step(data, output_G_dict)
            output_D_dict = self.forward_D_net(D_data, mode='train_d')
            loss_D_dict = self.forward_D_loss_func(output_D_dict, D_data, mode='train')
            loss_D_sum, loss_D_dict = self.reduce_loss(loss_D_dict)
            self.optimize_D_param(loss_D_sum)
            D_data, output_D_dict = self.after_train_D_step(D_data, output_D_dict)
        else:
            output_D_dict = {}
            loss_D_dict = {}

        # update generator ema
        self.accumulate(self.g_net_ema, self.g_net)

        # summarize output and loss
        output_dict = {}
        output_dict.update(output_G_dict)
        output_dict.update(output_D_dict)

        loss_dict = {"G_loss": loss_G_dict, "D_loss": loss_D_dict}
        return output_dict, loss_dict

    def metric(self, data, output_dict, loss_dict=None, mode='val'):
        pass

    def save(self, save_dir, epoch=None, iter=None):
        if epoch is not None:
            g_net_save_path = osp.join(save_dir, 'g_net_{}.pth'.format(epoch))
        elif iter is not None:
            g_net_save_path = osp.join(save_dir, 'g_net_iter_{}.pth'.format(iter))
        self._save_model(self.g_net, g_net_save_path)
        self._save_model(self.g_net_ema, g_net_save_path.replace('g_net', 'g_net_ema'))
        if self.train_gan:
            self._save_model(self.d_net, g_net_save_path.replace('g_net', 'd_net'))

    def log_loss_metric(self, writer: VisualBoard, loss_dict, metric_dict, mode='train', global_step=None):
        writer.add_scalars("G_loss", loss_dict['G_loss'], global_step=global_step)
        writer.add_scalars("D_loss", loss_dict['D_loss'], global_step=global_step)

    def log_image(self, writer: VisualBoard, data, output_dict, mode='train', global_step=None):
        show_images = [data['d_image_y'], data['rec_image'], data['gt_image_y'], data['warp_image_y']]
        writer.visual_image("show_image", show_images, iters=global_step)
