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
import tifffile as tiff

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


def pack_gbrg_raw2(im):
	im = np.expand_dims(im, axis=2)
	img_shape = im.shape
	H = img_shape[0]
	W = img_shape[1]
	out = np.concatenate((im[1:H:2, 0:W:2, :],
						  im[1:H:2, 1:W:2, :],
						  im[0:H:2, 1:W:2, :],
						  im[0:H:2, 0:W:2, :]), axis=2)
	return out




@DATASET_REGISTRY.register()
class VideoRAWDataset(BaseDataset):
    """ Video data set for recurrent video processing model training

    Args:
        config: configuration for video data set. contains flowing key attributes:
        data_root:  path to data folders, HQ data
        num_frame: frame numbers for each training sequence
        meta_info_file: path to meta info, which including total frame info
        distortion: configs of online degradation

    """
    def __init__(self, config):
        super(VideoRAWDataset, self).__init__(config)
        
        self.root = config['data_path']
        self.num_frame = config['num_frame']
        self.keys = []

        for dir in self.root:
            self.keys += glob.glob(os.path.join(dir, 'clean') + '/*_1.npy')

        self.noiseLevels = {}
        f = open('/mnt/lustrenew/share_data/yuyitong/data/PVDD/raw/noiseLevel.txt')
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
        print(path)
        frame_this = np.load(path)
        # print(frame_this.shape)
        # frame_this = np.array(frame_this).astype(np.float32)
        return frame_this

    def _add_distortion(self, imgs):
        return self.distortion(imgs)
    
    def create_transforms(self, config):
        return None

    def __getitem__(self, item):
        frame_name = self.keys[item]

        folder_frame_num = self.num_frame

        neighbor_list_clean = [frame_name.replace('_1.npy', '_{}.npy'.format(i)) for i in range(1, folder_frame_num + 1)]
        neighbor_list_noisy = [name.replace('clean', 'noisy') for name in neighbor_list_clean]
        # get the GT frames (as the center frame)
        img_gts = []
        for neighbor in neighbor_list_clean:
            img_gt = self._read_image(neighbor)
            if np.nan in img_gt or np.inf in img_gt:
                print('img_gt has nan.')
            img_gts.append(pack_gbrg_raw2(img_gt))
        # get the noisy frames (as the center frame)
        img_lqs = []
        for neighbor in neighbor_list_noisy:
            img_lq = self._read_image(neighbor)
            if np.nan in img_lq or np.inf in img_lq:
                print('img_lq has nan.')
            img_lqs.append(pack_gbrg_raw2(img_lq))
        # frames crop and resize
        img_gts, img_lqs = random_crop(img_gts, img_lqs, gt_patch_size=self.config.crop_size)
           

        # img_lqs = self._add_distortion(img_gts)

        # to tensor
        img_lqs = self.img_to_tensor(img_lqs)
        img_gts = self.img_to_tensor(img_gts)

        img_lqs = torch.stack(img_lqs, dim=0)  # b, n, c, h, w
        img_gts = torch.stack(img_gts, dim=0)

        
        return {'lq': img_lqs, 'gt': img_gts, 'noiselevel': np.float32(self.noiseLevels[os.path.basename(frame_name)[:-6]])}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class VideoRAWDatasetVal(BaseDataset):
    """ Video data set for recurrent video processing model training

    Args:
        config: configuration for video data set. contains flowing key attributes:
        data_root:  path to data folders, HQ data
        num_frame: frame numbers for each training sequence
        meta_info_file: path to meta info, which including total frame info
        distortion: configs of online degradation

    """
    def __init__(self, config):
        super(VideoRAWDatasetVal, self).__init__(config)
        
        self.root = config['data_path']
        self.num_frame = config['num_frame']
        self.keys = []

        for dir in self.root:
            self.keys += glob.glob(os.path.join(dir, 'clean') + '/*_1_L.tif')
            self.keys += glob.glob(os.path.join(dir, 'clean') + '/*_1_S.png')
            self.keys += glob.glob(os.path.join(dir, 'clean') + '/*_1_M.png')
        self.noiseLevels = {}
        f = open('/mnt/lustrenew/share_data/yuyitong/data/PVDD/test/synNoiseData/noiseLevel.txt')
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
        frame_name = self.keys[item]

        folder_frame_num = self.num_frame
        if 'L' in frame_name:
            neighbor_list_clean = [frame_name.replace('_1_L.png', '_{}_L.png'.format(i)) for i in range(5)]
            noiselevel = 0.00547
        elif 'S' in frame_name:
            neighbor_list_clean = [frame_name.replace('_1_S.png', '_{}_S.png'.format(i)) for i in range(5)]
            noiselevel = 0.000687765
        elif 'M' in frame_name:
            neighbor_list_clean = [frame_name.replace('_1_M.png', '_{}_M.png'.format(i)) for i in range(5)]
            noiselevel = 0.00219122
        neighbor_list_noisy = [name.replace('clean', 'noisy') for name in neighbor_list_clean]
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
        img_gts = [img[512:512+256, 512:512+256, :] for img in img_gts] 
        img_lqs = [img[512:512+256, 512:512+256, :] for img in img_lqs] 
           

        # img_lqs = self._add_distortion(img_gts)

        # to tensor
        img_lqs = self.img_to_tensor(img_lqs)
        img_gts = self.img_to_tensor(img_gts)

        img_lqs = torch.stack(img_lqs, dim=0)  # n, c, h, w
        img_gts = torch.stack(img_gts, dim=0)

        return {'lq': img_lqs, 'gt': img_gts, 'noiselevel': noiselevel}

    def __len__(self):
        return len(self.keys)
