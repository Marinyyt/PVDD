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


def random_crop(img_gts, gt_patch_size, gt_path=None):
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
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    return img_gts





@DATASET_REGISTRY.register()
class VideoJointDataset(BaseDataset):
    """ Video data set for recurrent video processing model training

    Args:
        config: configuration for video data set. contains flowing key attributes:
        data_root:  path to data folders, HQ data
        num_frame: frame numbers for each training sequence
        meta_info_file: path to meta info, which including total frame info
        distortion: configs of online degradation

    """
    def __init__(self, config):
        super(VideoJointDataset, self).__init__(config)
        
        self.root = config['data_path']
        self.num_frame = config['num_frame']
        self.keys = []
        self.y_only = config['y_only']
        for i in range(len(self.root)):
            with open(config['meta_info_file'][i], 'r') as fin:
                fin = json.load(fin)
                self.meta_info = fin
                for key, v in self.meta_info.items():
                    folder, frame_num = key, v['frame_num']
                    self.keys.extend([f'{self.root[i]}@{folder}@{j:08d}' for j in range(int(frame_num))])

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
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _add_distortion(self, imgs):
        return self.distortion(imgs)
    
    def create_transforms(self, config):
        return None

    def __getitem__(self, item):
        data_path, folder_name, frame_name = self.keys[item].split('@')
        start_frame_idx = int(frame_name)

        folder_frame_num = (int(len(glob.glob(os.path.join(data_path, folder_name) + '/*.png')))-1)
        if start_frame_idx > folder_frame_num - self.num_frame:
            start_frame_idx = random.randint(0, folder_frame_num - self.num_frame)

        end_frame_idx = start_frame_idx + self.num_frame
        neighbor_list = list(range(start_frame_idx, end_frame_idx))

        # random reverse
        if self.config['random_reverse'] and random.random() < 0.5:
            neighbor_list.reverse()

        assert len(neighbor_list) == self.num_frame, f'Wrong length of neighbor list: {len(neighbor_list)}'

        # get the GT frames (as the center frame)
        img_gts = []
        for neighbor in neighbor_list:
            img_gt_path = os.path.join(data_path, folder_name, f'{neighbor:08d}.png')
            img_gt = self._read_image(img_gt_path)
            img_gts.append((img_gt / 255.).astype(np.float32))
        # frames crop and resize
        try:
            img_gts = random_crop(img_gts, gt_patch_size=self.config.crop_size)
            if self.config.crop_size != self.config.img_size:
                img_gts = [cv2.resize(img, dsize=(self.config.img_size, self.config.img_size)) for img in img_gts]
        except Exception as ex:
            img_gts = [cv2.resize(img, dsize=(self.config.img_size, self.config.img_size)) for img in img_gts]

        img_lqs = self._add_distortion(img_gts)
        if self.y_only:
            img_lqs = [np.expand_dims(cv2.cvtColor(img_lq, cv2.COLOR_RGB2GRAY), axis = -1) for img_lq in img_lqs]
            img_gts = [np.expand_dims(cv2.cvtColor(img_gt, cv2.COLOR_RGB2GRAY), axis = -1) for img_gt in img_gts]
        # to tensor
        img_lqs = self.img_to_tensor(img_lqs)
        img_gts = self.img_to_tensor(img_gts)

        img_lqs = torch.stack(img_lqs, dim=0)  # b, n, c, h, w
        img_gts = torch.stack(img_gts, dim=0)

        return {'lq': img_lqs, 'gt': img_gts}

    def __len__(self):
        return len(self.keys)
