import numpy as np
from pipeline_utils import get_visible_raw_image, get_metadata, normalize, white_balance, demosaic, apply_hue_correction, apply_exposure_compensation, \
    apply_hsv_enhance, apply_color_space_transform, transform_xyz_to_srgb, transform_xyz_to_prorgb, apply_brighten, apply_gamma, apply_tone_map, fix_orientation, transform_prorgb_to_srgb, raw_rgb_to_cct, \
    transform_rgb_to_raw, apply_degamma, de_white_balance, apply_mosaic, add_noise
from matplotlib import pyplot as plt
import imageio
from bm3d import bm3d
import colour
import cv2
from cv2.ximgproc import guidedFilter
def inverse_pipeline(image, meta_path, ratio, noise_profile, add = False):
    metadata = get_metadata(meta_path)
    img = image / 255.0

    degamma_image = apply_degamma(img)

    temp = raw_rgb_to_cct(metadata)
    rawrgb_image = transform_rgb_to_raw(degamma_image, temp, metadata)

    de_white_balance_image = de_white_balance(rawrgb_image, metadata['as_shot_neutral'])

    raw_image = apply_mosaic(de_white_balance_image)
    if add:
        raw_image, shot, read = add_noise(raw_image, noise_profile, metadata['as_shot_neutral'], ratio)
        return raw_image, shot, read
    return raw_image, None



def run_pipeline(noisy_raw_image, image_path, params):

    # metadata
    metadata = get_metadata(image_path)
    h, w = noisy_raw_image.shape[0], noisy_raw_image.shape[1]

    ##################  linearization  ###########################################################################
    linearization_table = metadata['linearization_table']
    if linearization_table is not None:
        print('Linearization table found. Not handled.')
        # TODO

    normalized_image = noisy_raw_image
    if params['output_stage'] == 'normal':
        return normalized_image
    ##############################################################################################################


    ##################  demosaicing  #############################################################################
    demosaiced_image = demosaic(normalized_image, metadata['white_level'], metadata['cfa_pattern'], output_channel_order='BGR',
                                alg_type=params['demosaic_type'] )
    # fix image orientation, if needed
    demosaiced_image = fix_orientation(demosaiced_image, metadata['orientation'])
    if params['output_stage'] == 'demosaic':
        return demosaiced_image
    ##############################################################################################################



    ##################  white_balance  ###########################################################################
    white_balanced_image = white_balance(demosaiced_image, metadata['as_shot_neutral'])

    if params['output_stage'] == 'white_balance':
        return white_balanced_image
    ##############################################################################################################


    ##################  color_space_trans  #######################################################################
    temp = raw_rgb_to_cct(metadata)

    xyz_image = apply_color_space_transform(white_balanced_image, metadata, temp)

    if params['output_stage'] == 'xyz':
        return xyz_image

    prorgb_image = transform_xyz_to_prorgb(xyz_image)

    if params['output_stage'] == 'prorgb':
        return prorgb_image
    ##############################################################################################################


    ##################  tone_mapping  ############################################################################
    tone_mapped_image = apply_tone_map(prorgb_image)
    if params['output_stage'] == 'tone':
        return tone_mapped_image
    ##############################################################################################################


    ##################  color_trans & gamma_correction  ##########################################################
    srgb_image = transform_prorgb_to_srgb(tone_mapped_image)
    if params['output_stage'] == 'sRGB':
        return srgb_image
    gamma_corrected_image = apply_gamma(srgb_image)
    if params['output_stage'] == 'gamma':
        return gamma_corrected_image

    ##############################################################################################################

    output_image = None
    return output_image





