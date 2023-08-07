import numpy as np
import random
import cv2
import math


def rotate(img, angle, prob=1.0):
    """
    对图片进行旋转
    args:
        img: np ndarray [HWC] or list of np ndarray, 如果是list，则每张图都使用相同的退化，并返回list
             数值在0~255，uint8
        angle: 旋转角度，list or int, 如果是list就随机从中选一个数值
        prob: 概率值， 0~1
    """
    def _rotate(_img):
        w, h = _img.shape[:2]
        center = (w / 2, h / 2)
        scale = 1.0
        # Perform the counter clockwise rotation holding at the center
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated_img = cv2.warpAffine(_img, M, (h, w))
        return rotated_img


    if random.random() < prob:
        if isinstance(angle, list):
            angle = random.sample(angle, k=1)[0]
        
        if isinstance(img, list):
            for i in range(len(img)):
                img[i] = _rotate(img[i])
        else:
            img = _rotate(img)
    
    return img


def rotate_coordinate(origin, point, angle, row):
    """ rotate coordinates in image coordinate system
    :param origin: tuple of coordinates,the rotation center
    :param point: tuple of coordinates, points to rotate
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated coordinates of point
    """
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)


