from abc import ABCMeta, abstractmethod
import os
import os.path as osp
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from .. import networks as N
from .. import losses as L
from utils.dist_parallel import master_only, Parallel
from utils.visualboard import VisualBoard


class BaseModel(metaclass=ABCMeta):

    def __init__(self, config):
        self.config = config
        self.init_network(config.Network)
        self.init_loss(config.Loss)
        self.init_optimizer(config.Optimizer)
        self.init_scheduler(config.Scheduler)
        self.resume()

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    @abstractmethod
    def init_network(self, network_config):
        """
        init your network
        """

    def init_loss(self, loss_config):
        self.loss_config = loss_config
        for k, v in loss_config.items(): 
            func = L.build_loss(v)
            setattr(self, k, func)


    @abstractmethod
    def init_optimizer(self, optim_config):
        """
        init your optimizer
        """


    def init_scheduler(self, scheduler_config):
        pass


    def before_train_step(self, data, augment_func=None):
        """
        Perform data preprocess before training.
        """
        return {k: v.to(self.device) for k, v in data.items()}


    def train_step(self, running_iter, data, augment_func=None):
        """
        Train one step

        return:
            output_dict: output of networks
            loss_dict: loss dict
        """
        data = self.before_train_step(data, augment_func)

        output_dict = self.forward_net(data, mode='train')
        loss_dict = self.forward_loss_func(output_dict, data, mode='train')
        loss_sum, loss_dict = self.reduce_loss(loss_dict)
        self.optimize_param(loss_sum)

        data, output_dict = self.after_train_step(data, output_dict)

        return output_dict, loss_dict

    
    def after_train_step(self, data, output_dict):
        """
        Perform data postprocess after training.

        args:
            data: dataset output
            output_dict: network output
        """
        return data, output_dict


    @abstractmethod
    def forward_net(self, data, mode="train"):
        """
        forward network in training or testing phase

        args:
            data: output of `before_train_step` or `before_val_step`
            mode: train | val

        return:
            output_dict: output of networks
        """

    
    @abstractmethod
    def forward_loss_func(self, output_dict, data, mode='train'):
        """
        forward loss

        args:
            output_dict: output of `forward_net`
            mode: train | val

        return:
            loss_dict: loss for each loss func
        """


    def reduce_loss(self, loss_dict):
        """
        Used after `forward_loss_func`
        """
        loss_sum = 0.
        for k, v in loss_dict.items():
            if len(v.shape) > 0:
                loss_sum += v.mean()
                loss_dict[k] = round(v.mean().item(), 5)
            else:
                loss_sum += v
                loss_dict[k] = round(v.item(), 5)
        
        return loss_sum, loss_dict


    @abstractmethod
    def optimize_param(self, loss_sum):
        """
        call `optimizer.step()` and `loss_sum.backward()` to BP
        """


    def before_val_step(self, data, augment_func=None):
        """
        Perform data preprocess before evaluating.
        """
        return data


    def val_step(self, data, augment_func=None):
        """
        Val one step

        return:
            output_dict: output of networks
            loss_dict: loss or None, some model may use loss as metric.
        """
        data = self.before_val_step(data, augment_func)

        output_dict = self.forward_net(data, mode='val')

        data, output_dict = self.after_val_step(data, output_dict)
        return output_dict, None


    def after_val_step(self, data, output_dict):
        """
        Perform data postprocess after validation.

        args:
            data: dataset output
            output_dict: network output
        
        return:
            data: dataset output after postprocess
            output_dict: network output after postprocess
        """

        return data, output_dict


    @abstractmethod
    def metric(self, data, output_dict, loss_dict=None, mode='val'):
        """
        metric for network.

        return:
            metric_dict: {k: v}, v should be float.
        """


    @master_only
    def _save_model(self, net, save_path):
        if isinstance(net, nn.DataParallel) or isinstance(net, DistributedDataParallel):
            save_state_dict = net.module.state_dict()
        else:
            save_state_dict = net.state_dict()
        
        torch.save(save_state_dict, save_path)
        print('Successfully save %s' % save_path)


    @abstractmethod
    def save(self, save_dir, epoch=None, iter=None):
        """
        Save model or optimizer
        """


    @abstractmethod
    def resume(self):
        """
        Resume model or optimizer, if possible
        """


    @abstractmethod
    def log_loss_metric(self, writer: VisualBoard, loss_dict, metric_dict, mode='train', global_step=None):
        """
        Print loss and metric log on tensorboard.
        args:
            loss_dict: loss dict from `train_step`
            metric_dict: metric result dict from `metric`
            mode: train | val
        return:
            metric_dict: dict of metric result, value should be `float` or `tensor.Tensor`
        """


    @abstractmethod
    def log_image(self, writer: VisualBoard, data, output_dict, mode='train', global_step=None):
        """
        Print image log on tensorboard.
        args:
            data: dict or tuple from your dataset
            output_dict: output of networks from `train_step`
            mode: train | val
        """


    def wrap(self, parallel: Parallel):
        '''
        Wrap Net and Loss to devices (CPU, GPU, or distributed GPU).
        parallel is class `Parallel` defined in utils.dist_parallel.py
        '''
        self.device = parallel.device
        for name, module in vars(self).items():
            # wrap Loss
            if isinstance(module, L.GANLossBase):
                setattr(self, name, module.to(parallel.device))
            elif isinstance(module, L.DistillLossBase):
                setattr(self, name, parallel.wrapper(module))
            elif isinstance(module, L.LossBase):
                setattr(self, name, module.to(parallel.device))
            # wrap Network
            elif isinstance(module, N.BaseNet):
                setattr(self, name, parallel.wrapper(module))
            # wrap torch in-defined loss
            elif isinstance(module, torch.nn.Module):
                setattr(self, name, module.to(parallel.device))


class GANBaseModel(BaseModel):

    """
    GAN基础训练流程
    """

    @staticmethod
    def accumulate(model1, model2, decay=0.999):
        """
        apply exponential moving average on generator
        """
        par1 = dict(model1.named_parameters())
        par2 = dict(model2.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

    def forward_net(self, data, mode="train"):
        raise ValueError(
            'GANBaseModel does not supprot `forward_net` function.')

    def forward_loss_func(self, output_dict, data, mode='train'):
        raise ValueError(
            'GANBaseModel does not supprot `forward_loss_func` function.')

    def optimize_param(self, loss_sum):
        raise ValueError(
            'GANBaseModel does not supprot `optimize_param` function.')


    def before_train_G_step(self, data, augment_func=None):
        """
        Perform G network data preprocess before training.
        """
        
        return data

    def after_train_G_step(self, data, output_dict):
        """
        Perform G network data postprocess after training.

        args:
            data: dataset output
            output_dict: network output
        """
        return data, output_dict


    @abstractmethod
    def forward_G_net(self, data, mode='train'):
        """
        forward G network in training or testing phase

        args:
            data: output of `before_train_step` or `before_val_step`
            mode: train | val

        return:
            output_dict: output of networks
        """

    @abstractmethod
    def forward_G_loss_func(self, output_dict, data, mode='train'):
        """
        forward G network loss

        args:
            output_dict: output of `forward_net`
            mode: train | val

        return:
            loss_dict: loss for each loss func
        """

    @abstractmethod
    def optimize_G_param(self, loss_sum):
        """
        backward G loss and optimize G param
        """


    def before_train_D_step(self, data, output_G_dict):
        """
        Perform D network data preprocess before training.
        """
        
        return data, output_G_dict


    def after_train_D_step(self, data, output_D_dict):
        """
        Perform D network data postprocess after training.

        args:
            data: dataset output
            output_D_dict: D network output
        """
        return data, output_D_dict


    @abstractmethod
    def forward_D_net(self, data, mode='train_d'):
        """
        forward D network in training or testing phase

        args:
            data: output of `before_train_step` or `before_val_step`
            mode: train_g | train_d | val

        return:
            output_dict: output of networks
        """

    @abstractmethod
    def forward_D_loss_func(self, output_dict, data, mode='train'):
        """
        forward D network loss

        args:
            output_dict: output of `forward_net`
            mode: train | val

        return:
            loss_dict: loss for each loss func
        """

    @abstractmethod
    def optimize_D_param(self, loss_sum):
        """
        backward D loss and optimize D param
        """


    def train_step(self, running_iter, data, augment_func=None):
        """
        Train one step

        return:
            output_dict: output of networks
            loss_dict: loss dict
        """

        # G training
        G_data = self.before_train_G_step(data, augment_func)

        output_G_dict = self.forward_G_net(G_data, mode='train')
        output_D_dict = self.forward_D_net({**G_data, **output_G_dict}, mode='train_g')

        loss_G_dict = self.forward_G_loss_func({**output_G_dict, **output_D_dict}, G_data, mode='train')
        loss_G_sum, loss_G_dict = self.reduce_loss(loss_G_dict)
        self.optimize_G_param(loss_G_sum)

        G_data, output_G_dict = self.after_train_G_step(G_data, output_G_dict)

        # D training
        D_data, output_G_dict = self.before_train_D_step(data, output_G_dict)

        output_D_dict = self.forward_D_net(D_data, mode='train_d')
        loss_D_dict = self.forward_D_loss_func(output_D_dict, D_data, mode='train')
        loss_D_sum, loss_D_dict = self.reduce_loss(loss_D_dict)
        self.optimize_D_param(loss_D_sum)

        D_data, output_D_dict = self.after_train_D_step(D_data, output_D_dict)

        # summarize output and loss
        output_dict = {}
        output_dict.update(output_G_dict)
        output_dict.update(output_D_dict)

        loss_dict = {}
        loss_dict.update(loss_G_dict)
        loss_dict.update(loss_D_dict)

        return output_dict, loss_dict

