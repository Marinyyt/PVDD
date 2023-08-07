import numpy as np

import torch

from utils.registry import RUNNER_REGISTRY
from .basic_runner import BasicRunner
from utils.dist_parallel import master_only


@RUNNER_REGISTRY.register()
class VideoRunner(BasicRunner):

    def train(self, running_iters, running_epoch):
        for i, data in enumerate(self.train_dataloader, 1):

            running_iters += 1

            if data is None:
                print('data is None')
                continue
            data = self.model.before_train_step(data)
            # with torch.autograd.detect_anomaly():
            output_dict, loss_dict = self.model.train_step(data)
            data, output_dict = self.model.after_train_step(data, output_dict)
            metric_dict = self.model.metric(data, loss_dict, mode='train')

            # record loss
            if running_iters % self.config.show_loss_iter == 0:
                self.print_log(running_iters, loss_dict, running_epoch)
                self.model.log_loss_metric(self.visualboard, loss_dict, metric_dict,
                                           mode='train', global_step=running_iters)
            # if running_iters % self.config.show_img_iter == 0:
            #     self.model.log_image(self.visualboard, data, output_dict,
            #                          mode='train', global_step=running_iters)

            # save model
            if self.config.save_mode == 'epoch':
                if running_epoch % self.config.save_by_epoch == 0 and \
                        running_iters % len(self.train_dataloader) == 0:
                    self.model.save(self.save_model_path, epoch=running_epoch + 1)
            elif self.config.save_mode == 'iter':
                if running_iters % self.config.save_by_iter == 0:
                    self.model.save(self.save_model_path, iter=running_iters)
            else:
                raise ValueError('save_mode %s is not supported.' % self.config.save_mode)

    @master_only
    def print_log(self, iters, losses, epoch):
        print('[%d|%d] training, iter: %d, %s' % (epoch, self.config.epochs, iters, str(losses)))

    @master_only
    def print_val_log(self, iters, metric, epoch):
        print('[%d|%d] eval, iter: %d, %s' % (epoch, self.config.epochs, iters, str(metric)))


    @torch.no_grad()
    def val(self, running_iters, running_epoch):
        metric_info = {}
        for i, data in enumerate(self.val_dataloader, 1):
            # running_iters += 1
            if data is None:
                print('data is None')
                continue
            data = self.model.before_train_step(data)
            output_dict = self.model.val_step(data)
            metric_dict = self.model.metric(data, output_dict, mode='val')
            for k, v in metric_dict.items():
                if isinstance(v, float):
                    if k not in metric_info:
                        metric_info[k] = []
                    metric_info[k].append(v)

        for k, v in metric_info.items():
            metric_info[k] = np.mean(v)

        self.print_val_log(running_iters, metric_info, running_epoch)
        
        # record metric sum
        if len(metric_info) > 0:
            self.model.log_loss_metric(self.visualboard, loss_dict=None, metric_dict=metric_info,
                                       mode='val', global_step=running_iters)
    
    def run(self):
        iters_down = 0
        for epoch in range(self.config.start_epoch, self.config.epochs):
            if self.parallel.distributed_training:
                self.train_sampler.set_epoch(epoch)
            self.train(iters_down, epoch)
            # if epoch % 10 == 0:
            #     self.val(iters_down, epoch)
            iters_down += len(self.train_dataloader)
            
