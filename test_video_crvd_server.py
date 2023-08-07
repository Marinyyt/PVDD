#!/bin/sh
"""
Denoise all the sequences existent in a given folder using FastDVDnet.

@author: Matias Tassano <mtassano@parisdescartes.fr>
"""

import os
import argparse
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from natsort import natsorted
from models.networks.pvdd0803_net import pvdd0803
from models.networks.pvdd0815_net import pvdd0815
from utils.test_utils import batch_psnr, init_logger_test, \
				variable_to_cv2_image, open_sequence, close_logger, get_imagenames, open_image,blocks2image_v2, divide_block_v2
import glob

#import hiddenlayer as h
# model_temp = shuffleVR(n_chs=16, in_blocks=1, fuse_blocks=1, out_blocks=1)
# vis_graph = h.build_graph(model_temp, torch.zeros([1 ,2, 1, 28, 28]))   # 获取绘制图像的对象
# vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
# vis_graph.save("./demo1.png")


NUM_IN_FR_EXT = 5 # temporal size of patch
MC_ALGO = 'DeepFlow' # motion estimation algorithm
OUTIMGEXT = '.png' # output images format

torch.backends.cudnn.enabled = True
IDX = 0


def temp_denoise(model, noisyframe, sigma_noise1):
	'''Encapsulates call to denoising model and handles padding.
		Expects noisyframe to be normalized in [0., 1.]
	'''
	# make size a multiple of four (we have two scales in the denoiser)

	sh_im = noisyframe.size()
	pad_start = time.time()
	expanded_h = sh_im[-2] % 16
	if expanded_h:
		expanded_h = 16 - expanded_h
	expanded_w = sh_im[-1] % 16
	if expanded_w:
		expanded_w = 16 - expanded_w
	padexp = (0, expanded_w, 0, expanded_h)
	noisyframe = noisyframe[0]
	noisyframe = F.pad(input=noisyframe, pad=padexp, mode='reflect')
	noisyframe = noisyframe.unsqueeze(0).cuda()
	sigma_noise1 = F.pad(input=sigma_noise1, pad=padexp, mode='reflect')

	pad_end = time.time()
	# print('padding time ' + str(pad_end - pad_start))
	# sigma_noise2 = F.pad(input=sigma_noise2, pad=padexp, mode='reflect')
	# sigma_noise3 = F.pad(input=sigma_noise3, pad=padexp, mode='reflect')
	# sigma_noise4 = F.pad(input=sigma_noise4, pad=padexp, mode='reflect')
	# sigma_noise5 = F.pad(input=sigma_noise5, pad=padexp, mode='reflect')

	# denoise
	start = time.time()
	out = torch.clamp(model(noisyframe), 0., 1.)
	end = time.time()
	# print('inference time '  + str(start-end))
	if expanded_h:
		out = out[:, :, :, :-expanded_h, :]
	if expanded_w:
		out = out[:, :, :, :, :-expanded_w]

	return out


def hook_fn_forward(model, input, output):
	global IDX
	print(model)
	#input = input.cpu().detach().numpy() * 255
	output = (output.cpu().detach().numpy() * 255.0).transpose(0, 2, 3, 1)
	output = output + 50
	temp_save = './wonp_temp_feature1'
	if not os.path.exists(temp_save):
		os.mkdir(temp_save)
	cv2.imwrite(temp_save + '/test_{}.png'.format(IDX), output[0, :, :, :])
	IDX+=1



def save_out_seq(seqclean, save_dir, sigmaval, suffix, save_noisy, i):
	"""Saves the denoised and noisy sequences under save_dir
	"""
	seq_len = seqclean.size()[0]
	for idx in range(i, i+seq_len):
		# Build Outname
		fext = OUTIMGEXT
		noisy_name = os.path.join(save_dir,\
						('n{}_{:04d}').format(sigmaval, idx) + fext)
		if len(suffix) == 0:
			out_name = os.path.join(save_dir,\
					('n{}_FastDVDnet_{:04d}').format(sigmaval, idx) + fext)
		else:
			out_name = os.path.join(save_dir,\
					('n{}_FastDVDnet_{}_{:04d}').format(sigmaval, suffix, idx) + fext)

		# Save result
		if save_noisy:
			noisyimg = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.))
			cv2.imwrite(noisy_name, noisyimg)

		outimg = variable_to_cv2_image(seqclean[idx%seq_len].unsqueeze(dim=0))
		cv2.imwrite(out_name, outimg)



def test_fastdvdnet(**args):
	"""Denoises all sequences present in a given folder. Sequences must be stored as numbered
	image sequences. The different sequences must be stored in subfolders under the "test_path" folder.

	Inputs:
		args (dict) fields:
			"model_file": path to model
			"test_path": path to sequence to denoise
			"suffix": suffix to add to output name
			"max_num_fr_per_seq": max number of frames to load per sequence
			"noise_sigma": noise level used on test set
			"dont_save_results: if True, don't save output images
			"no_gpu": if True, run model on CPU
			"save_path": where to save outputs as png
			"gray": if True, perform denoising of grayscale images instead of RGB
	"""
	# Start time
	start_time = time.time()

	# If save_path does not exist, create it
	if not os.path.exists(args['save_path']):
		os.makedirs(args['save_path'])
	logger = init_logger_test(args['save_path'])

	# Sets data type according to CPU or GPU modes

	# Create models
	print('Loading models ...')
	model_temp = pvdd0815(          # input_nc: 3
          num_feat= 64,
          num_block= 3,
          num_block_f= 3,
          num_block_pre= 3,
          dynamic_refine_thres= 255.,
          is_sequential_cleaning= False,
          depth= 2, depth_pre = 1,
          num_head= 8,
          num_frames= 2,
          window_size= [8, 8], window_size_pre = [16, 16],
          mlp_ratio= 2.,
          qkv_bias= True,
          qk_scale= None,
          drop_rate= 0.,
          attn_drop_rate= 0,
          drop_path_rate= 0.,
          drop_path= 0., 
          mlp = '04'
	).cuda()

	# Load saved weights
	model_temp.load_pretrain_model(args['model_file'])


	# Sets the model in evaluation mode (e.g. it removes BN)
	model_temp.eval()
	with torch.no_grad():
		dirs = glob.glob(args['test_path'] + '/*')
		psnr_fin = []
		ssim_fin = []
		for dir in dirs:
			if os.path.isdir(dir):
				sub_dirs = glob.glob(dir + '/*')
				for sub_dir in sub_dirs:
					if os.path.isdir(sub_dir):
						files = natsorted(glob.glob(sub_dir + '/*clean.png'))
						gt_files = [cv2.cvtColor(cv2.imread(file, -1), cv2.COLOR_BGR2RGB) for file in files]
						noisy_files = [cv2.cvtColor(cv2.imread(file.replace('clean', 'noisy0'), -1), cv2.COLOR_BGR2RGB) for file in files]
						if np.max(gt_files[0]) > 256:
							m = 65535
						else:
							m = 255
						gt_files = [gt_file / m for gt_file in gt_files]
						seq_list = [img.transpose(2, 0, 1) / m for img in noisy_files]

						seq = np.stack(seq_list, axis=0)
						seq = np.clip(seq, 0, 1)
						seq = torch.from_numpy(seq).type(torch.FloatTensor)
						numframes, C, H, W = seq.shape

						inframes_t = seq.reshape((1, args['num_frame'], C, H, W)).type(torch.FloatTensor).cuda()
						noiseLevel = torch.FloatTensor([args['noise_sigma']])

						denframe = torch.zeros((1, args['num_frame'], 3, H, W))
						rects, _ = divide_block_v2(H, W, H // 4, W // 4, 20)
						blocks = [[] for _ in range(args['num_frame'])]

						for rect in rects:
							noisemap = noiseLevel.expand((1, 1, rect[1] - rect[0], rect[3] - rect[2])).cuda()
							out = torch.clamp(temp_denoise(model_temp, inframes_t[:, :, :, rect[0]:rect[1], rect[2]:rect[3]], noisemap),
												  0, 1)[0, :, :, :, :]
							for j in range(args['num_frame']):
								blocks[j].append(out[j, :, :, :])
						# print('one processing time' + str(one_end - one_start))
						for j in range(args['num_frame']):
							denframe[0, j] = blocks2image_v2(blocks[j], H, W, rects, H // 4, W // 4, 20, 1)

						res = denframe[0].detach().cpu().numpy()
						p = []
						s = []
						for j in range(args['num_frame']):
							p.append(compare_psnr(gt_files[j], res[j].transpose(1, 2, 0), data_range=1.0))
							s.append(compare_ssim(gt_files[j], res[j].transpose(1, 2, 0), data_range=1.0, multichannel=True))
						psnr_fin.append(np.mean(np.array(p)))
						ssim_fin.append(np.mean(np.array(s)))
						for j in range(args['num_frame']):
							out_name = args['save_path'] + '/' + os.path.basename(dir) + '_' + os.path.basename(sub_dir) + '_'  + os.path.basename(files[j])
							outimg = np.clip(res[j].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
							outimg = cv2.cvtColor(outimg, cv2.COLOR_RGB2BGR)
							cv2.imwrite(out_name, outimg)

							# out_name = args['save_path'] + '/' + os.path.basename(os.path.join(dir, 'clean.png'))
							# outimg = np.clip(gt_files[j] * 255, 0, 255).astype(np.uint8)
							# outimg = cv2.cvtColor(outimg, cv2.COLOR_RGB2BGR)
							# cv2.imwrite(out_name, outimg)
		print('CRVD PSNR: ' + str(np.mean(np.array(psnr_fin))))
		print('CRVD SSIM: ' + str(np.mean(np.array(ssim_fin))))

		# total_end = time.time()
			# print('total time' + str(total_end - start_time))
			# out_path = os.path.join(args['save_path'], video_name)
			# for i in range(len(files)):
			# 	if not os.path.exists(out_path):
			# 		os.makedirs(out_path)
			# 	out_name = out_path + '/' + os.path.basename(files[i])
			# 	outimg = np.clip(res[i].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
			# 	# outimg[:, :, 1] = cv2.ximgproc.guidedFilter(outimg[:, :, 0], outimg[:, :, 1], 10, 2, -1)
			# 	# outimg[:, :, 2] = cv2.ximgproc.guidedFilter(outimg[:, :, 0], outimg[:, :, 2], 10, 2, -1)
			# 	outimg = cv2.cvtColor(outimg, cv2.COLOR_RGB2BGR)
			# 	#outimg = variable_to_cv2_image(denframe)
			# 	print(outimg.shape)
			#
			# 	cv2.imwrite(out_name, outimg)

	# close logger
	close_logger(logger)



if __name__ == "__main__":
	# Parse arguments
	parser = argparse.ArgumentParser(description="Denoise a sequence with FastDVDnet")
	parser.add_argument("--model_file", type=str,\
						default="./videoDenoise/logs-rvdv1-edvr-np-ssim-png/ckpt_e1561.pth", \
						help='path to model of the pretrained denoiser')
	parser.add_argument("--test_path", type=str, default="/mnt/lustrenew/share_data/yuyitong/data/4KHDR/test0415/portrait_red_billboard_ori_dnv2", \
						help='path to sequence to denoise')
	parser.add_argument("--suffix", type=str, default="", help='suffix to add to output name')
	parser.add_argument("--max_num_fr_per_seq", type=int, default=45, \
						help='max number of frames to load per sequence')
	parser.add_argument("--noise_sigma", type=float, default=10, help='noise level used on test set')
	parser.add_argument("--num_frame", type=int, default=7)
	parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")
	parser.add_argument("--save_noisy", action='store_true', help="save noisy frames")
	parser.add_argument("--no_gpu", action='store_true', help="run model on CPU")
	parser.add_argument("--save_path", type=str, default='./videoDenoise/trainingData-release/clean-denoise/', \
						 help='where to save outputs as png')
	parser.add_argument("--gray", action='store_true',\
						help='perform denoising of grayscale images instead of RGB')

	argspar = parser.parse_args()
	# Normalize noises ot [0, 1]
	argspar.noise_sigma = 0.003

	# use CUDA?
	argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

	print("\n### Testing FastDVDnet model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	test_fastdvdnet(**vars(argspar))
