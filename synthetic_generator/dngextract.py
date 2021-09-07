import numpy as np
from pipeline_utils import get_visible_raw_image, get_metadata, normalize, white_balance, demosaic, apply_lens_correction, apply_hue_correction, apply_exposure_compensation, \
    apply_hsv_enhance, apply_color_space_transform, transform_xyz_to_srgb, transform_xyz_to_prorgb, apply_brighten, apply_gamma, apply_tone_map, fix_orientation, transform_prorgb_to_srgb, raw_rgb_to_cct, \
    transform_rgb_to_raw, apply_degamma, de_white_balance, apply_mosaic, add_noise
from matplotlib import pyplot as plt
import imageio
from bm3d import bm3d
import colour
import cv2
from cv2.ximgproc import guidedFilter

m = get_metadata('G:/NIKON/static/group20/YYT_0310.dng')
a = 1