

import os
from fractions import Fraction
import math
import cv2
import numpy as np
import exifread
from exifread.utils import Ratio
import rawpy
from scipy.io import loadmat
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007, demosaicing_CFA_Bayer_bilinear
import struct
from exif_data_formats import exif_formats
from exif_utils import parse_exif_tag, parse_exif, get_tag_values_from_ifds
from skimage.color import rgb2hsv, hsv2rgb
import colour
import torch
def get_visible_raw_image(image_path):
    raw_image = rawpy.imread(image_path).raw_image_visible.copy()
    return raw_image


def get_image_tags(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
    return tags


def get_image_ifds(image_path):
    ifds = parse_exif(image_path, verbose=False)
    return ifds


def get_metadata(image_path):
    metadata = {}
    tags = get_image_tags(image_path)
    ifds = get_image_ifds(image_path)
    # camera info
    metadata['camera_model'] = get_camera_model(ifds)
    metadata['lens_info'] = None

    # normalize
    metadata['linearization_table'] = get_linearization_table(tags)
    metadata['black_level'] = get_black_level(tags, ifds)
    metadata['white_level'] = get_white_level(tags, ifds)

    # demosaic
    metadata['cfa_pattern'] = get_cfa_pattern(tags, ifds)

    # white_balance
    metadata['as_shot_neutral'] = get_as_shot_neutral(tags)

    # color_space_trans
    color_matrix_1, color_matrix_2 = get_color_matrices(tags)
    color_matrix_1 = np.reshape(np.asarray(color_matrix_1, dtype=np.float64), (3, 3)) if color_matrix_1 else None
    color_matrix_2 = np.reshape(np.asarray(color_matrix_2, dtype=np.float64), (3, 3)) if color_matrix_2 else None
    metadata['color_matrix_1'] = color_matrix_1
    metadata['color_matrix_2'] = color_matrix_2
    metadata['analog_balance'] = get_tag_values_from_ifds(50727, ifds)
    metadata['calibration_illuminant1'] = get_tag_values_from_ifds(50778, ifds)
    metadata['calibration_illuminant2'] = get_tag_values_from_ifds(50779, ifds)


    # hsv_correction
    hue_map1, hue_map2 = get_hue_map(ifds)
    metadata['hue_map1'] = hue_map1
    metadata['hue_map2'] = hue_map2

    # exposure_compensation
    metadata['base_exposure'] = get_tag_values_from_ifds(50730, ifds)

    # hsv_enhancement
    metadata['hsv_map'] = get_hsv_enhance_map(ifds)

    # others
    metadata['orientation'] = get_orientation(tags)
    metadata['noise_profile'] = get_noise_profile(tags, ifds)


    # ...
    # fall back to default values, if necessary
    if metadata['black_level'] is None:
        metadata['black_level'] = 0
        print("Black level is None; using 0.")
    if metadata['white_level'] is None:
        metadata['white_level'] = 2 ** 16
        print("White level is None; using 2 ** 16.")
    if metadata['cfa_pattern'] is None:
        metadata['cfa_pattern'] = [0, 1, 1, 2]
        print("CFAPattern is None; using [0, 1, 1, 2] (RGGB)")
    if metadata['as_shot_neutral'] is None:
        metadata['as_shot_neutral'] = [1, 1, 1]
        print("AsShotNeutral is None; using [1, 1, 1]")
    if metadata['color_matrix_1'] is None:
        #metadata['color_matrix_1'] = [1] * 9
        print("ColorMatrix1 is None; using [1, 1, 1, 1, 1, 1, 1, 1, 1]")
    if metadata['color_matrix_2'] is None:
        #metadata['color_matrix_2'] = [1] * 9
        print("ColorMatrix2 is None; using [1, 1, 1, 1, 1, 1, 1, 1, 1]")
    if metadata['orientation'] is None:
        metadata['orientation'] = 0
        print("Orientation is None; using 0.")
    # ...
    return metadata


def get_camera_model(ifds):
    temp = get_tag_values_from_ifds(50708, ifds)
    camera_model = []
    for i in range(len(temp)-1):
        ch = temp[i]
        camera_model.append(str(temp[i])[2])
    return ''.join(camera_model)


def get_linearization_table(tags):
    possible_keys = ['Image Tag 0xC618', 'Image Tag 50712', 'LinearizationTable', 'Image LinearizationTable']
    return get_values(tags, possible_keys)


def get_black_level(tags, ifds):
    possible_keys = ['Image Tag 0xC61A', 'Image Tag 50714', 'BlackLevel', 'Image BlackLevel']
    vals = get_values(tags, possible_keys)
    if vals is None:
        vals = get_tag_values_from_ifds(50714, ifds)
    vals = [np.float32(val) for val in vals]


    repeat_dims = get_tag_values_from_ifds(50713, ifds)
    try:
        if repeat_dims is not None:
            black_level = np.array(vals).reshape(repeat_dims[0], repeat_dims[1])
    except:
        black_level = np.array([[251.52344 , 251.52344], [251.52344 , 251.52344]])
    #black_level = vals[0]

    delta_h = get_tag_values_from_ifds(50715, ifds)
    if delta_h is not None:
        delta_h =np.reshape(delta_h, (len(delta_h), 1))
        rows = len(delta_h)
        black_level = np.tile(black_level, (rows // repeat_dims[0], 1))
        black_level += delta_h


    delta_w = get_tag_values_from_ifds(50716, ifds)
    if delta_w is not None:
        cols = len(delta_w)
        black_level = np.tile(black_level, (1, cols // repeat_dims[1]))
        black_level += delta_w
    return black_level


def get_white_level(tags, ifds):
    possible_keys = ['Image Tag 0xC61D', 'Image Tag 50717', 'WhiteLevel', 'Image WhiteLevel']
    vals = get_values(tags, possible_keys)
    if vals is None:
        print("White level not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(50717, ifds)
    return vals


def get_cfa_pattern(tags, ifds):
    possible_keys = ['CFAPattern', 'Image CFAPattern']
    vals = get_values(tags, possible_keys)
    if vals is None:
        print("CFAPattern not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(33422, ifds)
    return vals


def get_as_shot_neutral(tags):
    possible_keys = ['Image Tag 0xC628', 'Image Tag 50728', 'AsShotNeutral', 'Image AsShotNeutral']
    return get_values(tags, possible_keys)



def get_color_matrices(tags):
    possible_keys_1 = ['Image Tag 0xC621', 'Image Tag 50721', 'ColorMatrix1', 'Image ColorMatrix1']
    color_matrix_1 = get_values(tags, possible_keys_1)
    possible_keys_2 = ['Image Tag 0xC622', 'Image Tag 50722', 'ColorMatrix2', 'Image ColorMatrix2']
    color_matrix_2 = get_values(tags, possible_keys_2)
    return color_matrix_1, color_matrix_2


def get_hue_map(ifds):
    hue_map_dims = get_tag_values_from_ifds(50937, ifds)

    hue_map1 = get_tag_values_from_ifds(50938, ifds)
    hue_map2 = get_tag_values_from_ifds(50939, ifds)
    if not hue_map1 and not hue_map2:
        return None, None
    hue_map1 = np.reshape(hue_map1, (hue_map_dims[2], hue_map_dims[0], hue_map_dims[1], 3))
    hue_map2 = np.reshape(hue_map2, (hue_map_dims[2], hue_map_dims[0], hue_map_dims[1], 3))
    return hue_map1, hue_map2

def get_hsv_enhance_map(ifds):
    hsv_map_dims = get_tag_values_from_ifds(50981, ifds)
    hsv_map = get_tag_values_from_ifds(50982, ifds)
    if not hsv_map:
        return None
    hsv_map = np.reshape(hsv_map, (hsv_map_dims[2], hsv_map_dims[0], hsv_map_dims[1], 3))
    return hsv_map



def get_orientation(tags):
    possible_tags = ['Orientation', 'Image Orientation']
    return get_values(tags, possible_tags)


def get_noise_profile(tags, ifds):
    possible_keys = ['Image Tag 0xC761', 'Image Tag 51041', 'NoiseProfile', 'Image NoiseProfile']
    vals = get_values(tags, possible_keys)
    if vals is None:
        print("Noise profile not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(51041, ifds)
    return vals


def get_values(tags, possible_keys):
    values = None
    for key in possible_keys:
        if key in tags.keys():
            values = tags[key].values
    return values


def normalize(raw_image, black_level, white_level):
    if black_level.shape[0] * black_level.shape[1] == 1:
        black_level = float(black_level[0])
    if type(white_level) is list and len(white_level) == 1:
        white_level = float(white_level[0])

    black_level_mask = black_level
    if black_level.shape[0]*black_level.shape[1] == 4:
        if type(black_level[0]) is Ratio:
            black_level = ratios2floats(black_level)
        black_level_mask = np.zeros(raw_image.shape)
        idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        step2 = 2
        for i, idx in enumerate(idx2by2):
            black_level_mask[idx[0]::step2, idx[1]::step2] = black_level[idx[0]][idx[1]]
        black_level_mask = np.tile(black_level_mask, (raw_image.shape[0] // black_level_mask.shape[0], raw_image.shape[1] // black_level_mask.shape[1]))

    if black_level.shape[0] == raw_image.shape[0]:
        black_level_mask = np.tile(black_level, (1, raw_image.shape[1] // black_level.shape[1]))

    if black_level.shape[1] == raw_image.shape[1]:
        black_level_mask = np.tile(black_level, (raw_image.shape[0] // black_level.shape[0], 1))

    normalized_image = raw_image.astype(np.float32) - black_level_mask
    # if some values were smaller than black level

    normalized_image = normalized_image / (white_level - black_level_mask)
    normalized_image = np.clip(normalized_image, 0, 1)
    return normalized_image



def ratios2floats(ratios):
    floats = []
    for ratio in ratios:
        floats.append(float(ratio.num) / ratio.den)
    return floats


def white_balance(normalized_image, as_shot_neutral):
    if type(as_shot_neutral[0]) is Ratio:
        as_shot_neutral = ratios2floats(as_shot_neutral)

    white_balanced_image = np.zeros(normalized_image.shape)
    white_balanced_image[:, :, 0] = normalized_image[:, :, 0] / as_shot_neutral[0]
    white_balanced_image[:, :, 1] = normalized_image[:, :, 1] / as_shot_neutral[1]
    white_balanced_image[:, :, 2] = normalized_image[:, :, 2] / as_shot_neutral[2]

    white_balanced_image = np.clip(white_balanced_image, 0.0, 1.0)
    return white_balanced_image

def de_white_balance(raw_image, as_shot_neutral):
    if type(as_shot_neutral[0]) is Ratio:
        as_shot_neutral = ratios2floats(as_shot_neutral)

    de_white_balanced_image = np.zeros(raw_image.shape)
    de_white_balanced_image[:, :, 0] = raw_image[:, :, 0] * as_shot_neutral[0]
    de_white_balanced_image[:, :, 1] = raw_image[:, :, 1] * as_shot_neutral[1]
    de_white_balanced_image[:, :, 2] = raw_image[:, :, 2] * as_shot_neutral[2]

    de_white_balanced_image = np.clip(de_white_balanced_image, 0.0, 1.0)
    return de_white_balanced_image

def get_opencv_demsaic_flag(cfa_pattern, output_channel_order, alg_type='VNG'):
    # using opencv edge-aware demosaicing
    if alg_type != '':
        alg_type = '_' + alg_type
    if output_channel_order == 'BGR':
        if cfa_pattern == [0, 1, 1, 2]:  # RGGB
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2BGR' + alg_type)
        elif cfa_pattern == [2, 1, 1, 0]:  # BGGR
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_RG2BGR' + alg_type)
        elif cfa_pattern == [1, 0, 2, 1]:  # GRBG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GB2BGR' + alg_type)
        elif cfa_pattern == [1, 2, 0, 1]:  # GBRG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GR2BGR' + alg_type)
        else:
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2BGR' + alg_type)
            print("CFA pattern not identified.")
    else:  # RGB
        if cfa_pattern == [0, 1, 1, 2]:  # RGGB
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2RGB' + alg_type)
        elif cfa_pattern == [2, 1, 1, 0]:  # BGGR
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_RG2RGB' + alg_type)
        elif cfa_pattern == [1, 0, 2, 1]:  # GRBG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GB2RGB' + alg_type)
        elif cfa_pattern == [1, 2, 0, 1]:  # GBRG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GR2RGB' + alg_type)
        else:
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2RGB' + alg_type)
            print("CFA pattern not identified.")
    return opencv_demosaic_flag


def demosaic(white_balanced_image, maxval, cfa_pattern, output_channel_order='BGR', alg_type='VNG' ):
    """
    Demosaic a Bayer image.
    :param white_balanced_image:
    :param cfa_pattern:
    :param output_channel_order:
    :param alg_type: algorithm type. options: '', 'EA' for edge-aware, 'VNG' for variable number of gradients
    :return: Demosaiced image
    """
    max_val = maxval
    if alg_type == 'VNG':
        max_val = 255
        wb_image = (white_balanced_image * max_val).astype(dtype=np.uint8)
    else:
        max_val = 16383
        wb_image = (white_balanced_image * max_val).astype(dtype=np.uint16)

    if alg_type in ['', 'EA', 'VNG']:
        opencv_demosaic_flag = get_opencv_demsaic_flag(cfa_pattern, output_channel_order, alg_type=alg_type)
        demosaiced_image = cv2.cvtColor(wb_image, opencv_demosaic_flag)
    elif alg_type == 'menon2007':
        cfa_pattern_str = "".join(["RGB"[i] for i in cfa_pattern])
        demosaiced_image = demosaicing_CFA_Bayer_Menon2007(white_balanced_image, pattern=cfa_pattern_str)

    demosaiced_image = np.clip(demosaiced_image, 0, 1)

    return demosaiced_image

def apply_mosaic(de_white_balance_image):
    h, w = de_white_balance_image.shape[0], de_white_balance_image.shape[1]
    mosaic_image = np.zeros((h, w), dtype=np.float32)
    mosaic_image[0:h:2, 0:w:2] = de_white_balance_image[0:h:2, 0:w:2, 0]
    mosaic_image[0:h:2, 1:w:2] = de_white_balance_image[0:h:2, 1:w:2, 1]
    mosaic_image[1:h:2, 0:w:2] = de_white_balance_image[1:h:2, 0:w:2, 1]
    mosaic_image[1:h:2, 1:w:2] = de_white_balance_image[1:h:2, 1:w:2, 2]
    return np.clip(mosaic_image, 0, 1)

def apply_color_space_transform(demosaiced_image, metadata, temp):
    color_matrix_1 = metadata['color_matrix_1']
    color_matrix_2 = metadata['color_matrix_2']

    CC = temp*color_matrix_1 + (1-temp)*color_matrix_2
    #CC = color_matrix_1
    CC = CC / np.sum(CC, axis=1, keepdims=True)
    CC = np.linalg.inv(CC)
    xyz_image = CC[np.newaxis, np.newaxis, :, :] * demosaiced_image[:, :, np.newaxis, :]
    xyz_image = np.sum(xyz_image, axis=-1)
    xyz_image = np.clip(xyz_image, 0.0, 1.0)
    return xyz_image


def transform_xyz_to_srgb(xyz_image):
    # srgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
    #                      [0.2126729, 0.7151522, 0.0721750],
    #                      [0.0193339, 0.1191920, 0.9503041]])


    xyz2srgb = np.array([[3.2404542, -1.5371385, -0.4985314],
                         [-0.9692660, 1.8760108, 0.0415560],
                         [0.0556434, -0.2040259, 1.0572252]])


    xyz2srgb = xyz2srgb / np.sum(xyz2srgb, axis=-1, keepdims=True)

    srgb_image = xyz2srgb[np.newaxis, np.newaxis, :, :] * xyz_image[:, :, np.newaxis, :]
    srgb_image = np.sum(srgb_image, axis=-1)
    srgb_image = np.clip(srgb_image, 0.0, 1.0)
    return srgb_image

def transform_xyz_to_prorgb(xyz_image):
    xyz2prorgb = np.array([[1.3459433, -0.2556075, -0.0511118],
                           [-0.5445989, 1.5081673, 0.0205351],
                           [0.0, 0.0, 1.2118128]
                           ])
    xyz2prorgb = xyz2prorgb / np.sum(xyz2prorgb, axis=-1, keepdims=True)

    prorgb_image = xyz2prorgb[np.newaxis, np.newaxis, :, :] * xyz_image[:, :, np.newaxis, :]
    prorgb_image = np.sum(prorgb_image, axis=-1)
    prorgb_image = np.clip(prorgb_image, 0.0, 1.0)
    return prorgb_image

def transform_rgb_to_raw(degamma, temp, metadata):
    xyz2srgb = np.array([[3.2404542, -1.5371385, -0.4985314],
                         [-0.9692660, 1.8760108, 0.0415560],
                         [0.0556434, -0.2040259, 1.0572252]])

    xyz2srgb = xyz2srgb / np.sum(xyz2srgb, axis=-1, keepdims=True)

    xyz_image=  np.linalg.inv(xyz2srgb[np.newaxis, np.newaxis, :, :]) * degamma[:, :, np.newaxis, :]
    xyz_image = np.sum(xyz_image, axis=-1)
    xyz_image = np.clip(xyz_image, 0.0, 1.0)



    color_matrix_1 = metadata['color_matrix_1']
    color_matrix_2 = metadata['color_matrix_2']

    CC = temp * color_matrix_1 + (1 - temp) * color_matrix_2
    CC = CC / np.sum(CC, axis=1, keepdims=True)
    raw_image=  CC[np.newaxis, np.newaxis, :, :] * xyz_image[:, :, np.newaxis, :]
    raw_image = np.sum(raw_image, axis=-1)
    raw_image = np.clip(raw_image, 0.0, 1.0)
    return raw_image



def transform_prorgb_to_srgb(prorgb_image):
    prorgb2xyz = np.array([[0.7976749, 0.1351917, 0.0313534],
                           [0.2880402, 0.7118741, 0.0000857],
                           [0.0, 0.0, 0.82521]
                           ])
    prorgb2xyz = prorgb2xyz / np.sum(prorgb2xyz, axis=-1, keepdims=True)

    xyz_image = prorgb2xyz[np.newaxis, np.newaxis, :, :] * prorgb_image[:, :, np.newaxis, :]
    xyz_image = np.sum(xyz_image, axis=-1)
    xyz_image = np.clip(xyz_image, 0.0, 1.0)
    srgb_image = transform_xyz_to_srgb(xyz_image)
    return srgb_image

def apply_hue_correction(prorgb_image, hue_map1, hue_map2, temp):
    hua_map = temp * hue_map1 + (1-temp) * hue_map2
    hsv_image = rgb2hsv(prorgb_image)
    hsv_image[:, :, 0] = hsv_image[:, :, 0] * 360
    Hn = np.minimum(np.floor(hsv_image[:, :, 0]/360 * hue_map1.shape[1]), hue_map1.shape[1]-1).astype(np.int16)
    Sn = np.minimum(np.floor(hsv_image[:, :, 1] * hue_map1.shape[2]), hue_map1.shape[2]-1).astype(np.int16)
    Vn = np.minimum(np.floor(hsv_image[:, :, 2] * hue_map1.shape[0]), hue_map1.shape[0]-1).astype(np.int16)

    for i in range(prorgb_image.shape[0]):
        for j in range(prorgb_image.shape[1]):
            hsv_image[i, j, 0] = hsv_image[i, j, 0] + hua_map[Vn[i][j], Hn[i][j], Sn[i][j]][0]
            hsv_image[i, j, 1] = hsv_image[i, j, 1] * hua_map[Vn[i][j], Hn[i][j], Sn[i][j]][1]
            hsv_image[i, j, 2] = hsv_image[i, j, 2] * hua_map[Vn[i][j], Hn[i][j], Sn[i][j]][2]
    hsv_image[:, :, 0] = hsv_image[:, :, 0] / 360.0
    hue_corrected_image = hsv2rgb(hsv_image)
    return hue_corrected_image


def apply_exposure_compensation(hue_corrected_image, ev):
    ev = np.float64(ev)
    return np.clip(hue_corrected_image * 2**ev, 0, 1)


def apply_hsv_enhance(ev_image, hsv_map):
    hsv_image = rgb2hsv(ev_image)
    hsv_image[:, :, 0] = hsv_image[:, :, 0] * 360
    Hn = np.minimum(np.floor(hsv_image[:, :, 0]/360 * hsv_map.shape[1]), hsv_map.shape[1]-1).astype(np.int16)
    Sn = np.minimum(np.floor(hsv_image[:, :, 1] * hsv_map.shape[2]), hsv_map.shape[2]-1).astype(np.int16)
    Vn = np.minimum(np.floor(hsv_image[:, :, 2] * hsv_map.shape[0]), hsv_map.shape[0]-1).astype(np.int16)

    for i in range(ev_image.shape[0]):
        for j in range(ev_image.shape[1]):
            hsv_image[i, j, 0] = hsv_image[i, j, 0] + hsv_map[Vn[i][j], Hn[i][j], Sn[i][j]][0]
            hsv_image[i, j, 1] = hsv_image[i, j, 1] * hsv_map[Vn[i][j], Hn[i][j], Sn[i][j]][1]
            hsv_image[i, j, 2] = hsv_image[i, j, 2] * hsv_map[Vn[i][j], Hn[i][j], Sn[i][j]][2]
    hsv_image[:, :, 0] = hsv_image[:, :, 0] / 360.0
    hsv_enhanced_image = hsv2rgb(hsv_image)
    return hsv_enhanced_image

def fix_orientation(image, orientation):
    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW

    if type(orientation) is list:
        orientation = orientation[0]

    if orientation == 1:
        pass
    elif orientation == 2:
        image = cv2.flip(image, 0)
    elif orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        image = cv2.flip(image, 1)
    elif orientation == 5:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


def reverse_orientation(image, orientation):
    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW
    rev_orientations = np.array([1, 2, 3, 4, 5, 8, 7, 6])
    return fix_orientation(image, rev_orientations[orientation - 1])


def apply_gamma(x):
    return x ** (1.0 / 2.2)

def apply_degamma(x):
    return x ** (2.2)


_aces_a = 2.51
_aces_b = 0.03
_aces_c = 2.43
_aces_d = 0.59
_aces_e = 0.14

def acesfilm(x, adapted_lum=1.):
    x *= adapted_lum
    return (x * (_aces_a * x +_aces_b)) / (x * (_aces_c * x + _aces_d) + _aces_e)


def apply_tone_map(x):
    tone_mapped_img = acesfilm(x)
    return tone_mapped_img


def apply_ACES(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return x*(a*x + b) / (x*(c*x+d)+e)


def apply_brighten(x):
    # simple brightness control
    gray = 0.30 * x[:, :, 0] + 0.59 * x[:, :, 1] + 0.11 * x[:, :, 2]
    brighten_image = x * (0.25 / np.mean(gray))
    brighten_image = np.clip(brighten_image, 0, 1)
    return brighten_image




def raw_rgb_to_cct(metadata):
    """Convert raw-RGB triplet to corresponding correlated color temperature (CCT)"""

    AsShotNeutral = metadata['as_shot_neutral']

    color_matrix_1 = metadata['color_matrix_1']
    color_matrix_2 = metadata['color_matrix_2']


    white_point_xyz = np.asarray(AsShotNeutral, np.float64)
    pxyz = [0.5, 1.0, 0.5]
    loss = 1e10
    k = 1
    cct1, cct2 = 2855, 6504
    while loss > 1e-4:
        UCS = colour.XYZ_to_UCS(pxyz)
        uv = colour.UCS_to_uv(UCS)
        cct, dist= colour.uv_to_CCT((uv[0], uv[1]), 'robertson1968')
        alpha = ((1/cct) - (1/cct2))/((1/cct1) - (1/cct2))
        C = alpha * color_matrix_1 + (1-alpha) * color_matrix_2
        C = np.linalg.inv(C)

        new_xyz = np.matmul(C, white_point_xyz)
        loss = l2_norm(new_xyz - pxyz)
        pxyz = new_xyz
        print('k = %d, loss = %f\n', [k, loss])
        k = k + 1
    return np.clip(alpha, 0, 1)


def l2_norm(x):
    return np.sqrt(np.sum(x**2))

def add_noise(raw_image, noise_profile, as_shot_neutral, ratio):

    if type(as_shot_neutral[0]) is Ratio:
        as_shot_neutral = ratios2floats(as_shot_neutral)
    h, w = raw_image.shape[0], raw_image.shape[1]
    result = np.zeros(shape = (h, w), dtype=np.float32)
    noise_profile = noise_profile.split(' ')

    r_shot = float(noise_profile[0])*ratio
    r_read = float(noise_profile[1])*ratio
    r_noise = np.random.randn(raw_image.shape[0], raw_image.shape[1]) * np.sqrt(raw_image * r_shot + r_read)


    g_shot = float(noise_profile[2])*ratio
    g_read = float(noise_profile[3])*ratio
    g_noise = np.random.randn(raw_image.shape[0], raw_image.shape[1]) * np.sqrt(raw_image * g_shot + g_read)


    b_shot = float(noise_profile[4])*ratio
    b_read = float(noise_profile[5])*ratio
    b_noise = np.random.randn(raw_image.shape[0], raw_image.shape[1]) * np.sqrt(raw_image * b_shot + b_read)



    result[0:h:2, 0:w:2] = raw_image[0:h:2, 0:w:2] + r_noise[ 0:h:2, 0:w:2]
    result[0:h:2, 1:w:2] = raw_image[0:h:2, 1:w:2] + g_noise[ 0:h:2, 1:w:2]
    result[1:h:2, 0:w:2] = raw_image[1:h:2, 0:w:2] + g_noise[ 1:h:2, 0:w:2]
    result[1:h:2, 1:w:2] = raw_image[1:h:2, 1:w:2] + b_noise[ 1:h:2, 1:w:2]

    return np.clip(result, 0, 1), g_shot, g_read






