import cv2
from pipeline import run_pipeline, inverse_pipeline
from pipeline_utils import get_metadata
import os
import numpy as np
import glob
import imageio
import cv2


def batch_generator(**args):

    params = {
    'output_stage': 'gamma',  # options: 'normal', 'demosaic', 'white_balance', 'xyz', 'prorgb', 'hue', 'ev', 'hsv_enhance', 'tone', 'sRGB', 'gamma', 'lens_correction', 'denoise'
    'save_as': 'tif',  # options: 'jpg', 'png', 'tif', etc.
    'demosaic_type': 'menon2007',
    'save_dtype': np.uint8
    }


    f = open('./noiseLevel.txt')
    noiselevel = []
    for line in f.readlines():
        noiselevel.append(line[:-1])


    noisyPath = os.path.join(args['outputFrame_path'], 'noisy')
    if not os.path.exists(noisyPath):
        os.makedirs(noisyPath)
    cleanPath = os.path.join(args['outputFrame_path'], 'clean')
    if not os.path.exists(cleanPath):
        os.makedirs(cleanPath)


    f = open(args['output_nl_path'], 'w')

    dirs = os.listdir(args['inputFrame_path'])


    for i, dir in enumerate(dirs):

        clean_imgs = glob.glob(dir + '/*.png')
        meta_path = glob.glob(dir + '/*.dng')[0]
        step = 100
        if len(clean_imgs) > 400:
            step = 250
        for j in range(30, len(clean_imgs), step):
            idx = np.random.randint(0, 6)
            ratio = np.sqrt(np.random.uniform(0, 1))
            for k in range(30):

                clean_img = cv2.imread(clean_imgs[j-k], -1)[:, :, :3]/65535.0*255
                clean_raw_image, n = inverse_pipeline(clean_img, meta_path, ratio=ratio, noise_profile=noiselevel[idx], add=False)
                clean_rgb_image = run_pipeline(clean_raw_image, meta_path, params)


                noisy_raw_image, shot, read = inverse_pipeline(clean_img, meta_path, ratio=ratio, noise_profile=noiselevel[idx], add=True)
                noisy_rgb_image = run_pipeline(noisy_raw_image, meta_path, params)


                cv2.imwrite('{}/noisy_{}_frame{}_{}.png'.format(noisyPath, dir[-4:], j, 30-k), noisy_rgb_image*255)
                cv2.imwrite('{}/clean_{}_frame{}_{}.png'.format(cleanPath, dir[-4:], j, 30-k), clean_rgb_image*255)

            f.write('clean_{}_frame{}'.format(dir[-4:], j) + ' ' + str(shot) + ' ' + str(read) + '\n')
    f.close()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate a synthetic video denoising dataset from log-videos.")
    parser.add_argument("--inputFrame_path", type=str,\
						default="X:/TrainingData/SmoreRVD/logVideoFrame", \
						help='path to input video frame data.')

    parser.add_argument("--outputFrame_path", type=str, \
                        default="./", \
                        help='path to output synthetic video frame data.')

    parser.add_argument("--output_nl_path", type=str, default="F:/NIKON/SmoreRVD_RAW/noiseLevel.txt", \
						help='output path for the recording noise level.')



    argspar = parser.parse_args()


    batch_generator(**vars(argspar))

