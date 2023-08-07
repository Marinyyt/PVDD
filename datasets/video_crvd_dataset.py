import os
import cv2
import json
import numpy as np
import glob
import random
import torch

from .base import BaseDataset
from . import distortion as D
from natsort import natsorted
from utils.registry import DATASET_REGISTRY


def random_crop(img_gts, img_lqs, gt_patch_size, gt_path=None):
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

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_gt, w_gt = img_gts[0].shape[0:2]

    # crop gt patch
    top_gt, left_gt = random.randint(0, h_gt - gt_patch_size), random.randint(0, w_gt - gt_patch_size)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
        img_lqs = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_lqs]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, :] for v in img_gts]
        img_lqs = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, :] for v in img_lqs]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    return img_gts, img_lqs



@DATASET_REGISTRY.register()
class VideoCRVDDataset(BaseDataset):
    """ Video data set for recurrent video processing model training

    Args:
        config: configuration for video data set. contains flowing key attributes:
        data_root:  path to data folders, HQ data
        num_frame: frame numbers for each training sequence
        meta_info_file: path to meta info, which including total frame info
        distortion: configs of online degradation

    """
    def __init__(self, config):
        super(VideoCRVDDataset, self).__init__(config)
        
        self.root = config['data_path']
        self.num_frame = config['num_frame']
        self.keys = []

        dirs = glob.glob(self.root[0] + '/*')
        for dir in dirs:
            self.keys += glob.glob(dir + '/*')
        
        self.noiseLevels = {}
        f = open('/mnt/lustrenew/share_data/yuyitong/data/PVDD/rgb/noiseLevel.txt')
        for line in f.readlines():
            self.noiseLevels[line.split(' ')[0]] = float(line.split(' ')[1])
        f.close()


    def create_augmentation(self, Augmentation):
        return None

    def create_augmentation_batch(self, BatchAugment):
        return None

    def get_face_annotations(self):
        pass

    def img_to_tensor(self, imgs):
        if isinstance(imgs, list):
            imgs_tensor = [torch.from_numpy(img.astype('float32')).permute(2, 0, 1) for img in imgs]
        else:
            imgs_tensor = torch.from_numpy(imgs.astype('float32')).permute(2, 0, 1)
        return imgs_tensor

    def _read_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _add_distortion(self, imgs):
        return self.distortion(imgs)
    
    def create_transforms(self, config):
        return None

    def __getitem__(self, item):
        folder_name = self.keys[item]
        # print(folder_name)
        neighbor_list_clean = natsorted(glob.glob(folder_name + '/*clean.png'))
        neighbor_list_noisy = [file.replace('clean.png', 'noisy0.png') for file in neighbor_list_clean]
        start_idx = np.random.randint(0, self.num_frame)
        neighbor_list_clean = [neighbor_list_clean[(start_idx + j)%7] for j in range(self.num_frame)]
        neighbor_list_noisy = [neighbor_list_noisy[(start_idx + j)%7] for j in range(self.num_frame)]

        # print(neighbor_list_clean)

        # get the GT frames (as the center frame)
        img_gts = []
        for neighbor in neighbor_list_clean:
            img_gt = self._read_image(neighbor)
            img_gts.append((img_gt / 255.).astype(np.float32))
        # get the noisy frames (as the center frame)
        img_lqs = []
        for neighbor in neighbor_list_noisy:
            img_lq = self._read_image(neighbor)
            img_lqs.append((img_lq / 255.0).astype(np.float32))
        # frames crop and resize
        img_gts, img_lqs = random_crop(img_gts, img_lqs, gt_patch_size=self.config.crop_size)
           

        # img_lqs = self._add_distortion(img_gts)

        # to tensor
        img_lqs = self.img_to_tensor(img_lqs)
        img_gts = self.img_to_tensor(img_gts)

        img_lqs = torch.stack(img_lqs, dim=0)  # b, n, c, h, w
        img_gts = torch.stack(img_gts, dim=0)

        
        return {'lq': img_lqs, 'gt': img_gts}

    def __len__(self):
        return len(self.keys)

