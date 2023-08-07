from copy import deepcopy
import numpy as np
import random
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F


# ========== helper func =================
def filter2D(img, kernel):
    """cv2.filter2D
    Args:
        img: (h, w, c), type: float32, 0-1
        kernel: (b, k, k)
    """
    img_blur = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    return img_blur.clip(0, 1)


def get_H_W(imgs):
    """
    获得图片的长宽
    args:
        imgs: list | np.ndarray | torch.Tensor
             np.ndarray: [HWC] or [NHWC],
             torch.Tensor: [CHW] or [NCHW]
    """
    if isinstance(imgs, list):
        if isinstance(imgs[0], np.ndarray):
            assert len(imgs[0].shape) == 4 or len(imgs[0].shape) == 3
            if len(imgs[0].shape) == 4:
                _, H, W, _ = imgs[0].shape
            else:
                H, W, _ = imgs[0].shape
        elif(imgs[0], torch.Tensor):
            assert len(imgs[0].shape) == 4 or len(imgs[0].shape) == 3
            if len(imgs[0].shape) == 4:
                _, _, H, W = imgs[0].shape
            else:
                _, H, W = imgs[0].shape
    else:
        if isinstance(imgs, np.ndarray):
            assert len(imgs.shape) == 4 or len(imgs.shape) == 3
            if len(imgs.shape) == 4:
                _, H, W, _ = imgs.shape
            else:
                H, W, _ = imgs.shape
        elif isinstance(imgs, torch.Tensor):
            assert len(imgs.shape) == 4 or len(imgs.shape) == 3
            if len(imgs.shape) == 4:
                _, _, H, W = imgs.shape
            else:
                _, H, W = imgs.shape
    
    return H, W



# ============= augment func =================

def haze(img, prob=1.0):
    """
    对图片增加雾感
    args:
        img: np ndarray [HWC] or list of np ndarray, 如果是list, 则每张图都使用相同的退化(要求每张图大小一致)，并返回list
             数值在0~255，uint8
        prob: 概率值， 0~1
    """

    if random.random() < prob:
        if random.random() > 0.5:
            if isinstance(img, list):
                h, w = img[0].shape[:2]
            else:
                h, w = img.shape[:2]

            # Base light mask
            light_mask = np.zeros(img.shape, dtype=np.float32)

            # Calculate end points
            left_ = random.randint(h // 3, h)
            right_ = random.randint(h // 3, h)
            k = (right_ - left_) / float(w - 1)
            b = left_
            end_points = np.floor(k * np.asarray(list(range(0, w))) + b).astype(np.int)
            end_points = np.clip(end_points, 0, h - 1)

            for idx, ep in enumerate(end_points):
                v_slice = ((np.linspace(255, 0, ep + 1)) / 255.0) ** 0.7 * 255.0
                light_mask[0:ep + 1, idx, :] = v_slice.reshape((ep + 1, 1))
        else:
            light_mask = np.ones(img.shape, dtype=np.float32) * 255.0

        a = random.uniform(0.5, 0.8)
        if isinstance(img, list):
            for i in range(len(img)):
                img[i] = np.clip(img[i] * a + light_mask * (1 - a), 0, 255).astype(np.uint8)
            out_img = img
        else:
            out_img = np.clip(img * a + light_mask * (1 - a), 0, 255).astype(np.uint8)
    else:
        out_img = img

    return out_img


def enhancement(image, k_size=[3,5,7], strength=[0,3], prob=1.0):
    """
    对图片进行细节增强
    args:
        img: np ndarray [HWC] or list of np ndarray, 如果是list, 则每张图都使用相同的退化, 并返回list
             数值在0~255,uint8
        k_size: 模糊核大小,list or int, 如果是list就随机从中选一个数值
        strength: 细节增强力度,list or int, list需要提供两个数,表示随机取值范围
        prob: 概率值, 0~1
    """
    if random.random() < prob:
        if isinstance(k_size, list):
            k_size = random.choice(k_size)
        if k_size % 2 == 0:
            k_size += 1
        
        if isinstance(strength, list):
            strength = random.uniform(strength[0], strength[1])
        
        if isinstance(image, list):
            for i in range(len(image)):
                image[i] = image[i].astype(np.float32) / 255.
                detail = image[i] - cv2.GaussianBlur(image[i], ksize=(k_size, k_size), sigmaX=0.8)
                image[i] = np.clip((image[i] + strength * detail) * 255., 0., 255.)
                image[i] = image[i].astype(np.uint8)
        else:
            image = image.astype(np.float) / 255
            detail = image - cv2.GaussianBlur(image, ksize=(k_size, k_size), sigmaX=0.8)
            image = np.clip((image + strength * detail) * 255, 0., 255.).astype(np.uint8)
    
    return image


def randomaffine(img, degree=[0, 2], translate=(0.2, 0.2), scale=(0.7, 0.9), prob=1.0):
    """
    对图片进行随机平移、旋转、缩放
    args:
        img: np ndarray [HWC] or list of np ndarray, 如果是list, 则每张图都使用相同的退化, 并返回list
             数值在0~255，uint8
        prob: 概率值， 0~1
    """
    if random.random() < prob:
        func = transforms.RandomAffine(degrees=degree,
                                       translate=tuple(translate),
                                       scale=tuple(scale))
        if isinstance(img, list):
            for i in range(len(img)):
                img[i] = func(Image.fromarray(img[i]))
                img[i] = np.array(img[i])
        else:
            img = func(Image.fromarray(img))
            img = np.array(img)
    
    return img



def sharpen(img, prob=1.0):
    """
    对图片进行锐化操作
    args:
        img: np ndarray [HWC] or list of np ndarray, 如果是list，则每张图都使用相同的退化，并返回list
             数值在0~255，uint8
        prob: 概率值， 0~1
    """
    def _apply_sharp(_img):
        img_float = _img.astype(np.float32) / 255.0
        img_blur = cv2.GaussianBlur(img_float, (5, 5), sigmaX=1.0, sigmaY=1.0)
        details = img_float - img_blur
        sharpen_rate = random.uniform(0.3, 1.0)
        img_sharpen = np.clip(img_float + sharpen_rate * details, 0.0, 1.0)
        img = (img_sharpen * 255.0).astype(np.uint8)
        return img

    if random.random() < prob:
        if isinstance(img, list):
            for i in range(len(img)):
                img[i] = _apply_sharp(img[i])
        else:
            img = _apply_sharp(img)

    return img


def advancedSharpen(img, prob=1.0):
    """
    对图片进行高级sharpen操作
    args:
        img: np ndarray [HWC] or list of np ndarray, 如果是list，则每张图都使用相同的退化，并返回list
             数值在0~255，uint8
        prob: 概率值， 0~1
    """

    def _apply_sharp(_img, layer1_hp_filter, layer2_hp_filter, weight1, weight2):
        img = _img.astype(np.float32) / 255.
        layer1_sharpen = filter2D(img, layer1_hp_filter/1232)
        img_down = cv2.resize(img, dsize=(img.shape[0]//2, img.shape[1]//2), interpolation=cv2.INTER_LINEAR)
        layer2_sharpen = cv2.resize(filter2D(img_down, layer2_hp_filter/220), dsize=(img.shape[0], img.shape[1]),
                                    interpolation=cv2.INTER_CUBIC)
        img = img + weight1 * layer1_sharpen + weight2 * layer2_sharpen
        return (img * 255).clip(0, 255).astype(np.uint8)

    if random.random() < prob:
        layer1_hp_filter = np.array(
            [[0, 0, -3, -5, -3, 0, 0],
            [0, -10, -57, -106, -57, -10, 0],
            [-3, -57, -132, 65, -132, -57, -3],
            [-5, -106, 65, 1232, 65, -106, -5],
            [-3, -57, -132, 65, -132, -57, -3],
            [0, -10, -57, -106, -57, -10, 0],
            [0, 0, -3, -5, -3, 0, 0]]
        )

        layer2_hp_filter = np.array(
            [[-1, -4, -6, -4, -1],
            [-4, -16, -24, -16, -4],
            [-6, -24, 220, -24, -6],
            [-4, -16, -24, -16, -4],
            [-1, -4, -6, -4, -1]]
        )

        weight1 = max(random.random(), 0.2)
        weight2 = max(random.random(), 0.1)

        if isinstance(img, list):
            for i in range(len(img)):
                img[i] = _apply_sharp(img[i], layer1_hp_filter, layer2_hp_filter, weight1, weight2)
        else:
            img = _apply_sharp(img, layer1_hp_filter, layer2_hp_filter, weight1, weight2)


    return img


def darken(img, dark_factor=1.25, prob=1.0):
    """
    调低图片亮度
    args:
        img: np ndarray [HWC] or list of np ndarray, 如果是list，则每张图都使用相同的退化，并返回list
             数值在0~255，uint8
        dark_factor: 数值越大，图片越暗，list or int, 如果是list就随机从中选一个数值
        prob: 概率值， 0~1
    """
    def _apply_dark(_img):
        img = _img.astype(np.float32) / 255.
        img = np.power(img, dark_factor)
        img = (img * 255.).astype(np.uint8)
        return img

    if random.random() < prob:
        if isinstance(dark_factor, list):
            dark_factor = random.sample(dark_factor, k=1)[0]
        
        if isinstance(img, list):
            for i in range(len(img)):
                img[i] = _apply_dark(img[i])
        else:
            img = _apply_dark(img)
    
    return img



def cutblur(img_cut, img_paste, ratio=0.3, prob=1.0):
    """
    CutBlur数据增强: https://arxiv.org/abs/2004.00448
    args:
        img_cut/img_paste: [HWC],数值在0~255,uint8
        ratio: [0~1], cut的比例,数值越大cut的区域越大
        prob: 概率值, 0~1
    """
    if random.random() < prob:
        target_img = img_cut.copy()
        w, h = img_cut.shape[:2]
        crop_w = int(w * ratio)
        crop_h = int(h * ratio)
        start_w = int((w - crop_w - 1) * random.random())
        start_h = int((h - crop_h - 1) * random.random())

        target_img[start_h:start_h+crop_h, start_w:start_w+crop_w] = \
            img_paste[start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        return target_img
    
    else:
        return img_cut


def cutblurMulti(imgs_cut, imgs_paste, ratio=0.3, prob=1.0):
    """
    CutBlur数据增强: https://arxiv.org/abs/2004.00448
    args:
        img_cut/img_paste: [HWC],数值在0~255,uint8
        ratio: [0~1], cut的比例,数值越大cut的区域越大
        prob: 概率值, 0~1
    """
    if random.random() < prob:
        target_imgs = imgs_cut.copy()
        w, h = imgs_cut[0].shape[:2]
        crop_w = int(w * ratio)
        crop_h = int(h * ratio)
        start_w = int((w - crop_w - 1) * random.random())
        start_h = int((h - crop_h - 1) * random.random())

        target_imgs[:][start_h:start_h + crop_h, start_w:start_w + crop_w] = \
            imgs_paste[:][start_h:start_h + crop_h, start_w:start_w + crop_w]

        return target_imgs

    else:
        return imgs_cut

def random_crop(imgs, crop_size, start_h=None, start_w=None, target_size=None, prob=1.0):
    """
    对图片随机crop到crop_size大小，如果size不是None，则在crop后再resize到target_size大小
    args:
        imgs: list | np.ndarray | torch.Tensor，数值类型没有要求
             np.ndarray: [HWC] or [NHWC]，
             torch.Tensor: [CHW] or [NCHW]
        crop_size: crop大小
        start_h/start_w: crop的起始点坐标，如果没有提供，就随机生成
        target_size: crop后重新resize
        prob: 概率值， 0~1
    """

    def _random_crop(img, crop_size, start_h, start_w, size=None):
        if isinstance(img, np.ndarray):
            if len(img.shape) == 4:
                img = img[:, start_h:start_h+crop_size, start_w:start_w+crop_size]
                if size is not None:
                    for i in range(img.shape[0]):
                        img[i] = cv2.resize(img[i], tuple(size), interpolation=cv2.INTER_LINEAR)
            else:
                img = img[start_h:start_h+crop_size, start_w:start_w+crop_size]
                if size is not None:
                    img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_LINEAR)
        elif isinstance(img, torch.Tensor):
            if len(img.shape) == 4:
                img = img[:, :, start_h:start_h+crop_size, start_w:start_w+crop_size]
                if size is not None:
                    img = F.interpolate(img, size, mode='bilinear')
            else:
                img = img[:, start_h:start_h+crop_size, start_w:start_w+crop_size]
                if size is not None:
                    img = F.interpolate(img.unsqueeze(0), size, mode='bilinear')[0]
        return img
    
    if random.random() < prob:
        H, W = get_H_W(imgs)
        if start_h is None:
            start_h = random.randint(0, max(0, H - crop_size))
        if start_w is None:
            start_w = random.randint(0, max(0, W - crop_size))
        
        if isinstance(imgs, list):
            for i in range(len(imgs)):
                imgs[i] = _random_crop(imgs[i], crop_size, start_h, start_w, target_size)
        else:
            imgs = _random_crop(imgs, crop_size, start_h, start_w, target_size)
    
    return imgs


def random_resize(imgs, scales, prob=1.0):
    """
    对图片随机resize
    args:
        imgs: list | np.ndarray | torch.Tensor，数值类型没有要求
             np.ndarray: [HWC] or [NHWC]，
             torch.Tensor: [CHW] or [NCHW]
        scales: list or float, 如果是list就从中选择一个缩放比例
        prob: 概率值， 0~1
    """

    def _random_resize(img, scale):
        if isinstance(img, np.array):
            if len(img.shape) == 4:
                for i in range(img.shape[0]):
                    h, w = img[i].shape[0] * scale, img[i].shape[1] * scale
                    img[i] = cv2.resize(img[i], (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                h, w = img.shape[0] * scale, img.shape[1] * scale
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        elif isinstance(img, torch.Tensor):
            if len(img.shape) == 4:
                img = F.interpolate(img, scale_factor=scale)
            else:
                img = F.interpolate(img, scale_factor=scale)
        
        return img
    
    if random.random() < prob:
        if isinstance(scales, list):
            scale = random.sample(scales, k=1)[0]
        else:
            scale = scales
        if isinstance(imgs, list):
            for i in range(len(imgs)):
                imgs[i] = _random_resize(imgs[i], scale)
        else:
            imgs = _random_resize(imgs, scale)
    
    return imgs



