import os
import os.path as osp
import numpy as np

from torch.utils.data.distributed import DistributedSampler
import torch

from models.restorers import build_model
from datasets import collate, build_dataset
from utils.visualboard import VisualBoard
from utils.dist_parallel import Parallel, master_only
from utils.registry import RUNNER_REGISTRY


@RUNNER_REGISTRY.register()
class BasicRunner:

    def __init__(self, config):
        self.config = config
        self.dataset_config = config.DatasetConfig
        self.model_config = config.ModelConfig

        log_path = osp.join(config.save_path, 'tb_log')
        self.visualboard = VisualBoard(log_path, use_pavi=config.use_pavi)
        self.parallel = Parallel(config.num_gpus, distributed_training=config.distributed_training, port=config.port)

        self.save_model_path = osp.join(config.save_path, 'model')
        self.mkdir(self.config)

        self.model = self.create_model(self.model_config)
        self.model.wrap(self.parallel)

        self.train_dataset, self.val_dataset = self.create_dataset(self.dataset_config)

        self.train_dataloader, self.train_sampler = self.create_train_dataloader(self.config, self.train_dataset, self.parallel)
        self.val_dataloader = self.create_val_dataloader(self.config, self.val_dataset, self.parallel)


    @master_only
    def mkdir(self, config):
        if not osp.exists(self.save_model_path):
            os.makedirs(self.save_model_path, exist_ok=True)

    
    def create_dataset(self, dataset_config):
        """
        Create train and val dataset
        """
        return build_dataset(dataset_config.TrainDataset), build_dataset(dataset_config.ValDataset)


    def create_train_dataloader(self, config, train_dataset, parallel):
        """
        config: RunnerConfig, with some attribute attatched in `train.py`
        parallel: Parallel
        """
        if parallel.distributed_training:
            train_sampler = DistributedSampler(train_dataset, num_replicas=parallel.world_size, rank=parallel.local_rank)
            shuffle = True
        else:
            train_sampler = None
            shuffle = True

        train_batch_size = config.train_batch_size
        # 非分布式且多GPU训练，需要手动扩充batchsize
        if config.num_gpus > 0 and not parallel.distributed_training:
            train_batch_size *= config.num_gpus
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
                                    num_workers=config.num_workers, pin_memory=False, sampler=train_sampler, 
                                    shuffle=shuffle)
        return train_loader, train_sampler
                

    def create_val_dataloader(self, config, val_dataset, parallel):
        """
        config: RunnerConfig, with some attribute attatched in `train.py`
        parallel: Parallel
        """
        if parallel.distributed_training:
            val_sampler = DistributedSampler(val_dataset, num_replicas=parallel.world_size, rank=parallel.local_rank)
            shuffle = False
        else:
            val_sampler = None
            shuffle = False
    
        val_batch_size = config.val_batch_size
        # 非分布式且多GPU训练，需要手动扩充batchsize
        if config.num_gpus > 0 and not parallel.distributed_training:
            val_batch_size *= config.num_gpus
        
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size,
                                    num_workers=config.num_workers, pin_memory=False, sampler=val_sampler, 
                                    shuffle=shuffle)
        
        return val_loader


    def create_model(self, model_config):
        return build_model(model_config)


    @master_only
    def print_log(self, iters, losses, epoch):
        print('[%d|%d] training, iter: %d, %s' % (epoch, self.config.epochs, iters, str(losses)))


    def train(self, running_iters, running_epoch):
        for i, data in enumerate(self.train_dataloader, 1):
            
            if data is None:
                print('data is None')
                continue

            running_iters += 1

            output_dict, loss_dict = self.model.train_step(running_iters, data, self.train_dataset.augment_batch)
            metric_dict = self.model.metric(data, output_dict, loss_dict, mode='train')

            # record loss
            if running_iters % self.config.show_loss_iter == 0:
                self.print_log(running_iters, loss_dict, running_epoch)
                self.model.log_loss_metric(self.visualboard, loss_dict, metric_dict, 
                               mode='train', global_step=running_iters)
            if running_iters % self.config.show_img_iter == 0:
                self.model.log_image(self.visualboard, data, output_dict,
                               mode='train', global_step=running_iters)
            
            # save model
            if self.config.save_mode == 'epoch':
                if running_epoch % self.config.save_by_epoch == 0 and \
                    running_iters % len(self.train_dataloader) == 0:
                    self.model.save(self.save_model_path, epoch=running_epoch+1)
            elif self.config.save_mode == 'iter':
                if running_iters % self.config.save_by_iter == 0:
                    self.model.save(self.save_model_path, iter=running_iters)
            else:
                raise ValueError('save_mode %s is not supported.' % self.config.save_mode)

    @torch.no_grad()
    def val(self, running_iters, running_epoch):

        metric_info = {}

        for i, data in enumerate(self.val_dataloader, 1):
            
            if data is None:
                print('data is None')
                continue

            running_iters += 1

            output_dict, loss_dict = self.model.val_step(data)
            metric_dict = self.model.metric(data, output_dict, loss_dict, mode='val')

            print('Validation, iter: %f, %s' % (running_iters, str(metric_dict)))

            for k, v in metric_dict.items():
                if isinstance(v, float):
                    if k not in metric_info:
                        metric_info[k] = []
                    metric_info[k].append(v)
        
        for k, v in metric_info.items():
            metric_info[k] = np.mean(v)

        # record metric sum
        if len(metric_info) > 0:
            self.print_log(running_iters, metric_info, running_epoch)
            self.model.log_loss_metric(self.visualboard, loss_dict=None, metric_dict=metric_info, 
                                       mode='val', global_step=running_epoch)
            # record val img
            if data is not None and output_dict is not None:
                self.log_image(self.visualboard, data, output_dict,
                                mode='val', global_step=running_iters)
            
    
    def run(self):
        iters_down = 0
        for epoch in range(self.config.start_epoch, self.config.epochs):
            if self.parallel.distributed_training:
                self.train_sampler.set_epoch(epoch)
            self.train(iters_down, epoch)
            self.val(iters_down, epoch)
            iters_down += len(self.train_dataloader)


    def finish(self):
        self.visualboard.close()
        self.parallel.close()
