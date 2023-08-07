import sys
sys.path.append('/mnt/lustre/share/pymc/py3')
import numpy as np
import cv2

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from . import augmentations as A
from . import distortion as D


class BaseDataset(data.Dataset):

    def __init__(self, config):
        self.config = config
        if hasattr(config, 'memcache') and config.memcache.enable:
            self.memcache_enable = True
            self.memcache_server_path = config.memcache.server_path
            self.memcache_client_path = config.memcache.client_path
        else:
            self.memcache_enable = False

        self.transforms = self.create_transforms(config)
        self.augmentation = self.create_augmentation(config.Augmentation)
        self.distortion = None
        self.augment_batch = self.create_augmentation_batch(config.BatchAugment)

    def create_transforms(self, config):
        transform_list = [
            transforms.ToPILImage(),
            transforms.ToTensor()
        ]

        if config.normalize:
            transform_list.append(
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                transforms.Normalize((0.5,), (0.5,))
            )
        
        transform = transforms.Compose(transform_list)
        return transform

    def create_augmentation(self, Augmentation):

        def apply_augmentation(img):
            if Augmentation is None:
                return img

            for aug_func in Augmentation:
                args = Augmentation.get(aug_func).args
                func = getattr(A, aug_func)
                img = func(img, **args)
            return img
        
        return apply_augmentation

    def create_distortion(self, Distortion):
        if Distortion is None:
            return None
        return getattr(D, Distortion.type)(Distortion.args)


    def create_augmentation_batch(self, BatchAugment):

        def apply_augment_batch(imgs):
            """
            对一个batch的图片或者数据进行augment, 在训练流程`**_model.py`中执行
            args:
                imgs: list or np.array or torch.Tensor
            """
            if BatchAugment is None:
                return imgs
                
            for aug_func in BatchAugment:
                args = BatchAugment.get(aug_func).args
                func = getattr(A, aug_func)
                imgs = func(imgs, **args)
            return imgs
        
        return apply_augment_batch


    def read_img(self, path):
        if self.memcache_enable:
            import mc
            mclient = mc.MemcachedClient.GetInstance(
                self.memcache_server_path, self.memcache_client_path)
            value = mc.pyvector()
            mclient.Get(path, value)
            value = mc.ConvertBuffer(value)
            img = np.frombuffer(value)
        else:
            img = cv2.imread(path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
