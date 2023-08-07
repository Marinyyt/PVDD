import torch
import numpy as np
import skimage
import skimage.measure


def psnr(pred, target, pixel_max_cnt = 255):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p


def grey_psnr(pred, target, pixel_max_cnt = 255):
    pred = torch.sum(pred, dim = 0)
    target = torch.sum(target, dim = 0)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt * 3 / rmse_avg)
    return p


def ssim(pred, target):
    pred = pred.clone().data.permute(1, 2, 0).cpu().numpy()
    target = target.clone().data.permute(1, 2, 0).cpu().numpy()
    ssim = skimage.measure.compare_ssim(target, pred, multichannel = True)
    return ssim