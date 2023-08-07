import os
import cv2
import json
import numpy as np
import glob
import random
import torch

from .base import BaseDataset
from . import distortion as D

from utils.registry import DATASET_REGISTRY


def random_crop(img_gt, num_frame, gt_patch_size):
    """Paired random crop. Support Numpy array and Tensor inputs.
    It crops lists of lq and gt images with corresponding locations.
    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        gt_path (str): Path to ground-truth. Default: None.
    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    # determine input type: Numpy array or Tensor

    h_gt, w_gt, _ = img_gt.shape
    img_gts = []

    # crop gt patch
    top_gt, left_gt = random.randint(10 * num_frame, h_gt - gt_patch_size - 10 * num_frame), random.randint(10 * num_frame, w_gt - gt_patch_size - 10 * num_frame)
    img_gts.append(img_gt[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, :])
    for _ in range(num_frame - 1):
        top_gt = top_gt + np.random.randint(-10, 10)
        left_gt = left_gt + np.random.randint(-10, 10)
        img_gts.append(img_gt[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, :])
    return img_gts



@DATASET_REGISTRY.register()
class ImageDataset(BaseDataset):
    """ Video data set for recurrent video processing model training

    Args:
        config: configuration for video data set. contains flowing key attributes:
        data_root:  path to data folders, HQ data
        num_frame: frame numbers for each training sequence
        meta_info_file: path to meta info, which including total frame info
        distortion: configs of online degradation

    """
    def __init__(self, config):
        super(ImageDataset, self).__init__(config)
        
        self.root = config['data_path']
        self.num_frame = config['num_frame']
        self.keys = []
        self.y_only = config['y_only']
        for i in range(len(self.root)):
            files = glob.glob(self.root[i] + '/*.jpg')
            self.keys += files
    def create_augmentation(self, Augmentation):
        return None

    def create_augmentation_batch(self, BatchAugment):
        return None

    def get_face_annotations(self):
        pass

    def img_to_tensor(self, imgs):
        imgs_tensor = [torch.from_numpy(img.astype('float32')).permute(2, 0, 1) for img in imgs]
        return imgs_tensor

    def _read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _add_distortion(self, imgs):
        return self.distortion(imgs)
    
    def create_transforms(self, config):
        return None

    def __getitem__(self, item):
        frame_name = self.keys[item]

        # get the GT frames (as the center frame)

        img_gt = (self._read_image(frame_name) / 255.).astype(np.float32)
        img_gts = random_crop(img_gt, self.num_frame, self.config.crop_size)

        img_lqs = self._add_distortion(img_gts)
        if type(img_lqs) == tuple:
            degradation_parameters = img_lqs[1]
            degradation_parameters = torch.FloatTensor(degradation_parameters)
            img_lqs = img_lqs[0]
        else:
            degradation_parameters = None

        if self.y_only:
            img_lqs = [np.expand_dims(cv2.cvtColor(img_lq, cv2.COLOR_RGB2GRAY), axis = -1) for img_lq in img_lqs]
            img_gts = [np.expand_dims(cv2.cvtColor(img_gt, cv2.COLOR_RGB2GRAY), axis = -1) for img_gt in img_gts]

        if True:
            img_lqs = [cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2YUV) / 255. for img in img_lqs]
            img_gts = [cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2YUV) / 255. for img in img_gts]
        # to tensor
        img_lqs = self.img_to_tensor(img_lqs)
        img_gts = self.img_to_tensor(img_gts)

        img_lqs = torch.stack(img_lqs, dim=0)  # b, n, c, h, w
        img_gts = torch.stack(img_gts, dim=0)

        return {'lq': img_lqs, 'gt': img_gts, 'params': degradation_parameters}

    def __len__(self):
        return len(self.keys)
