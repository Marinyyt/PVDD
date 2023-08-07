import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.registry import NETWORK_REGISTRY
from models.networks.base import BaseNet
class DNCNN(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(DNCNN, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, 16, kernel_size=3, padding=1, stride=1, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(16, out_ch, kernel_size=3, padding=1, stride=1, bias=True))

	def forward(self, x):
		# noise = self.convblock(x)
		# channel = noise.shape[1]
		# output = x[:, 0:channel, :, :] - noise
		output = self.convblock(x)
		return output


class RCNN(nn.Module):
	def __init__(self, in_ch):
		super(RCNN, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, 16, kernel_size=3, padding=1, stride=1, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=True))

	def forward(self, x):
		output = self.convblock(x)
		return output


class FCNN(nn.Module):
	def __init__(self, in_ch):
		super(FCNN, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, 16, kernel_size=3, padding=1, stride=1, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=True))

	def forward(self, x):
		output = self.convblock(x)
		return output




class EMVD_network8(nn.Module):
	def __init__(self, C = 3):
		super(EMVD_network8, self).__init__()
		self.C = C
		self.color_transform = nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0, bias=False)
		self.color_transform_inverse = nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0, bias=False)
		self.frequency_transform_L = nn.Parameter(torch.Tensor([0.7071067, 0.7071067]).view(2, 1))
		self.frequency_transform_H = nn.Parameter(torch.Tensor([-0.7071067, 0.7071067]).view(2, 1))

		self.frequency_transform_L_inverse = nn.Parameter(torch.Tensor([0.7071067, 0.7071067]).view(2, 1))
		self.frequency_transform_H_inverse = nn.Parameter(torch.Tensor([-0.7071067, 0.7071067]).view(2, 1))

		self.fcnn1 = FCNN(C + 1)
		self.fcnn = FCNN(C+1+1)
		self.scale_for_fusion = 2
		self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

		self.denoise = DNCNN(4*C+C+C+1, 4*C)
		self.refine = RCNN(4*C+4*C+1)

		# self.reset_params()

		color_matrix = torch.Tensor([[0.299,0.587,0.114], [-0.1678,-0.3313,0.5],[0.5,-0.4187,-0.0813]]).float()
		color_matrix = color_matrix.view(C, C, 1, 1)
		self.color_transform.weight.data = color_matrix

		color_matrix = torch.Tensor([[0.299, 0.587, 0.114], [-0.1678, -0.3313, 0.5], [0.5, -0.4187, -0.0813]]).float()
		color_matrix = color_matrix.view(C, C)
		color_matrix_inverse = torch.inverse(color_matrix)
		color_matrix_inverse = color_matrix_inverse.view(C, C, 1, 1)
		self.color_transform_inverse.weight.data = color_matrix_inverse
		# self.color_transform_inverse.weight.data =  torch.inverse(self.color_transform_inverse.weight.data)

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, zt=None, yt_1=None, return_weight=False, a_value=0, b_value=0):

		if return_weight:
			conv_weight_list1 = [self.color_transform.weight, self.color_transform_inverse.weight]
			conv_weight_list2 = [self.frequency_transform_L, self.frequency_transform_H]
			conv_weight_list3 = [self.frequency_transform_L_inverse, self.frequency_transform_H_inverse]
			return conv_weight_list1, conv_weight_list2, conv_weight_list3

		zt = self.color_transform(zt)
		batch_size = zt.shape[0]
		channel = zt.shape[1]
		height = zt.shape[2]
		width = zt.shape[3]
		zt = zt.view(batch_size * channel, 1, height, width)
		zt_LL = F.conv2d(zt, torch.mm(self.frequency_transform_L,self.frequency_transform_L.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
		zt_HL = F.conv2d(zt, torch.mm(self.frequency_transform_H,self.frequency_transform_L.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
		zt_LH = F.conv2d(zt, torch.mm(self.frequency_transform_L,self.frequency_transform_H.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
		zt_HH = F.conv2d(zt, torch.mm(self.frequency_transform_H,self.frequency_transform_H.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)

		yt_1 = self.color_transform(yt_1)
		yt_1 = yt_1.view(batch_size * channel, 1, height, width)
		yt_1_LL = F.conv2d(yt_1, torch.mm(self.frequency_transform_L,self.frequency_transform_L.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
		yt_1_HL = F.conv2d(yt_1, torch.mm(self.frequency_transform_H,self.frequency_transform_L.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
		yt_1_LH = F.conv2d(yt_1, torch.mm(self.frequency_transform_L,self.frequency_transform_H.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
		yt_1_HH = F.conv2d(yt_1, torch.mm(self.frequency_transform_H,self.frequency_transform_H.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)


		zt_LL_fuse = zt_LL
		yt_1_LL_fuse = yt_1_LL
		zt_list = []
		yt_list = []
		for mm in range(self.scale_for_fusion):
			batch_size = zt_LL_fuse.shape[0]
			channel = zt_LL_fuse.shape[1]
			height = zt_LL_fuse.shape[2]
			width = zt_LL_fuse.shape[3]

			zt_LL_fuse = zt_LL_fuse.view(batch_size*channel, 1, height, width)
			zt_LL_fuse_this = F.conv2d(zt_LL_fuse, torch.mm(self.frequency_transform_L,self.frequency_transform_L.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
			zt_LH_fuse = F.conv2d(zt_LL_fuse, torch.mm(self.frequency_transform_L,self.frequency_transform_H.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
			zt_HL_fuse = F.conv2d(zt_LL_fuse, torch.mm(self.frequency_transform_H,self.frequency_transform_L.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
			zt_HH_fuse = F.conv2d(zt_LL_fuse, torch.mm(self.frequency_transform_H,self.frequency_transform_H.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)

			yt_1_LL_fuse = yt_1_LL_fuse.view(batch_size * channel, 1, height, width)
			yt_1_LL_fuse_this = F.conv2d(yt_1_LL_fuse, torch.mm(self.frequency_transform_L,self.frequency_transform_L.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
			yt_1_LH_fuse = F.conv2d(yt_1_LL_fuse, torch.mm(self.frequency_transform_L,self.frequency_transform_H.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
			yt_1_HL_fuse = F.conv2d(yt_1_LL_fuse, torch.mm(self.frequency_transform_H,self.frequency_transform_L.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
			yt_1_HH_fuse = F.conv2d(yt_1_LL_fuse, torch.mm(self.frequency_transform_H,self.frequency_transform_H.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)

			zt_list.append([zt_LL_fuse_this, zt_LH_fuse, zt_HL_fuse, zt_HH_fuse])
			yt_list.append([yt_1_LL_fuse_this, yt_1_LH_fuse, yt_1_HL_fuse, yt_1_HH_fuse])
			zt_LL_fuse = zt_LL_fuse_this
			yt_1_LL_fuse = yt_1_LL_fuse_this


		for mm in range(self.scale_for_fusion):
			yt_1_LL_this, yt_1_LH_this, yt_1_HL_this, yt_1_HH_this = yt_list[self.scale_for_fusion-mm-1]
			zt_LL_this, zt_LH_this, zt_HL_this, zt_HH_this = zt_list[self.scale_for_fusion-mm-1]

			if mm == 0:
				zt_input = zt_LL_this

			sigma_t = torch.mean(zt_LL_this, dim=1).unsqueeze(dim=1) * a_value + b_value
			# print(torch.mean(sigma_t), torch.max(sigma_t), torch.min(sigma_t))
			# gamma_t_this = gamma_t[:, 0:1, :, :]
			if mm == 0:
				gammt_input = torch.cat([torch.abs(zt_LL_this - yt_1_LL_this), sigma_t], dim=1)
				gamma_t = self.fcnn1.forward(gammt_input)
				gamma_t = nn.Sigmoid()(gamma_t)
			else:
				gammt_input = torch.cat([torch.abs(zt_LL_this - yt_1_LL_this), gamma_t, sigma_t], dim=1)
				gamma_t = self.fcnn.forward(gammt_input)
				gamma_t = nn.Sigmoid()(gamma_t)

			channel = zt_LL_this.shape[1]
			gamma_t_this = gamma_t.repeat(1, channel, 1, 1)
			yt_par_LL_this = yt_1_LL_this * (1 - gamma_t_this) + zt_LL_this * gamma_t_this
			yt_par_LH_this = yt_1_LH_this * (1 - gamma_t_this) + zt_LH_this * gamma_t_this
			yt_par_HL_this = yt_1_HL_this * (1 - gamma_t_this) + zt_HL_this * gamma_t_this
			yt_par_HH_this = yt_1_HH_this * (1 - gamma_t_this) + zt_HH_this * gamma_t_this

			yt_par_input = torch.cat([yt_par_LL_this, yt_par_HL_this, yt_par_LH_this, yt_par_HH_this], dim=1)
			sigma_t_1 = torch.mean(yt_1_LL_this, dim=1).unsqueeze(dim=1) * a_value + b_value
			sigma_t_2 = torch.mean(zt_LL_this, dim=1).unsqueeze(dim=1) * a_value + b_value

			# gamma_t_this = gamma_t[:, 0:1, :, :]
			sigma_input = (1 - gamma_t) * (1 - gamma_t) * sigma_t_1 + gamma_t * gamma_t * sigma_t_2
			# print(torch.mean(sigma_t_1), torch.mean(sigma_t_2), 'sigma')
			denoise_input = torch.cat([yt_par_input, zt_LL_this, zt_input, sigma_input], dim=1)
			yt_war = self.denoise.forward(denoise_input)

			zt_input_LL = yt_war[:, 0:self.C, :, :]
			zt_input_HL = yt_war[:, self.C:self.C * 2, :, :]
			zt_input_LH = yt_war[:, self.C * 2:self.C * 3, :, :]
			zt_input_HH = yt_war[:, self.C * 3:self.C * 4, :, :]

			batch_size = zt_input_LL.shape[0]
			channel = zt_input_LL.shape[1]
			height = zt_input_LL.shape[2]
			width = zt_input_LL.shape[3]
			zt_input_LL = zt_input_LL.contiguous().view(batch_size * channel, 1, height, width)
			zt_input_HL = zt_input_HL.contiguous().view(batch_size * channel, 1, height, width)
			zt_input_LH = zt_input_LH.contiguous().view(batch_size * channel, 1, height, width)
			zt_input_HH = zt_input_HH.contiguous().view(batch_size * channel, 1, height, width)

			zt_input = F.conv_transpose2d(zt_input_LL, torch.mm(self.frequency_transform_L_inverse, self.frequency_transform_L_inverse.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2) + \
					   F.conv_transpose2d(zt_input_HL, torch.mm(self.frequency_transform_H_inverse, self.frequency_transform_L_inverse.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2) + \
					   F.conv_transpose2d(zt_input_LH, torch.mm(self.frequency_transform_L_inverse, self.frequency_transform_H_inverse.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2) + \
					   F.conv_transpose2d(zt_input_HH, torch.mm(self.frequency_transform_H_inverse, self.frequency_transform_H_inverse.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2)
			zt_input = zt_input.view(batch_size, channel, height*2, width*2)
			# zt_input = zt_input * 0.25
			gamma_t = self.upsample(gamma_t)

		sigma_t = torch.mean(zt_LL, dim=1).unsqueeze(dim=1) * a_value + b_value
		gammt_input = torch.cat([torch.abs(zt_LL - yt_1_LL), gamma_t, sigma_t], dim=1)
		gamma_t = self.fcnn.forward(gammt_input)
		gamma_t = nn.Sigmoid()(gamma_t)

		channel = zt_LL.shape[1]
		gamma_t_this = gamma_t.repeat(1, channel, 1, 1)
		yt_par_LL = yt_1_LL * (1 - gamma_t_this) + zt_LL * gamma_t_this
		yt_par_LH = yt_1_LH * (1 - gamma_t_this) + zt_LH * gamma_t_this
		yt_par_HL = yt_1_HL * (1 - gamma_t_this) + zt_HL * gamma_t_this
		yt_par_HH = yt_1_HH * (1 - gamma_t_this) + zt_HH * gamma_t_this

		yt_par_input = torch.cat([yt_par_LL, yt_par_HL, yt_par_LH, yt_par_HH], dim=1)
		sigma_t_1 = torch.mean(yt_1_LL, dim=1).unsqueeze(dim=1) * a_value + b_value
		sigma_t_2 = torch.mean(zt_LL, dim=1).unsqueeze(dim=1) * a_value + b_value
		# gamma_t_this = gamma_t[:, 0:1, :, :]
		sigma_input = (1 - gamma_t) * (1 - gamma_t) * sigma_t_1 + gamma_t * gamma_t * sigma_t_2
		denoise_input = torch.cat([yt_par_input, zt_LL, zt_input, sigma_input], dim=1)
		yt_war = self.denoise.forward(denoise_input)

		refine_input = torch.cat([yt_war, yt_par_input, sigma_input], dim=1)
		refine_output = self.refine.forward(refine_input)
		refine_output = nn.Sigmoid()(refine_output)
		channel = yt_par_input.shape[1]
		refine_output = refine_output.repeat(1, channel, 1, 1)
		final_output = yt_par_input * refine_output + yt_war * (1-refine_output)

		# print(torch.mean(final_output), 'output1')
		final_output_LL = final_output[:, 0:self.C, :, :]
		final_output_HL = final_output[:, self.C:self.C*2, :, :]
		final_output_LH = final_output[:, self.C*2:self.C * 3, :, :]
		final_output_HH = final_output[:, self.C*3:self.C * 4, :, :]

		batch_size = final_output_LL.shape[0]
		channel = final_output_LL.shape[1]
		height = final_output_LL.shape[2]
		width = final_output_LL.shape[3]
		final_output_LL = final_output_LL.contiguous().view(batch_size * channel, 1, height, width)
		final_output_HL = final_output_HL.contiguous().view(batch_size * channel, 1, height, width)
		final_output_LH = final_output_LH.contiguous().view(batch_size * channel, 1, height, width)
		final_output_HH = final_output_HH.contiguous().view(batch_size * channel, 1, height, width)

		final_output1 = F.conv_transpose2d(final_output_LL, torch.mm(self.frequency_transform_L_inverse, self.frequency_transform_L_inverse.permute(1, 0)).unsqueeze(0).view(1,1, 2, 2), stride=2) + \
				   		F.conv_transpose2d(final_output_HL, torch.mm(self.frequency_transform_H_inverse, self.frequency_transform_L_inverse.permute(1, 0)).unsqueeze(0).view(1,1, 2, 2), stride=2) + \
				   		F.conv_transpose2d(final_output_LH, torch.mm(self.frequency_transform_L_inverse, self.frequency_transform_H_inverse.permute(1, 0)).unsqueeze(0).view(1,1, 2, 2), stride=2) + \
				   		F.conv_transpose2d(final_output_HH, torch.mm(self.frequency_transform_H_inverse, self.frequency_transform_H_inverse.permute(1, 0)).unsqueeze(0).view(1,1, 2, 2), stride=2)
		final_output1 = final_output1.view(batch_size, channel, height*2, width*2)
		# final_output1 = final_output1 * 0.25
		final_output2 = self.color_transform_inverse(final_output1)

		batch_size = yt_par_LL.shape[0]
		channel = yt_par_LL.shape[1]
		height = yt_par_LL.shape[2]
		width = yt_par_LL.shape[3]
		yt_par_LL = yt_par_LL.contiguous().view(batch_size * channel, 1, height, width)
		yt_par_HL = yt_par_HL.contiguous().view(batch_size * channel, 1, height, width)
		yt_par_LH = yt_par_LH.contiguous().view(batch_size * channel, 1, height, width)
		yt_par_HH = yt_par_HH.contiguous().view(batch_size * channel, 1, height, width)

		yt_par = F.conv_transpose2d(yt_par_LL, torch.mm(self.frequency_transform_L_inverse, self.frequency_transform_L_inverse.permute(1,0)).unsqueeze(0).view(1,1, 2, 2), stride=2) + \
				 F.conv_transpose2d(yt_par_HL, torch.mm(self.frequency_transform_H_inverse, self.frequency_transform_L_inverse.permute(1,0)).unsqueeze(0).view(1,1, 2, 2), stride=2) + \
				 F.conv_transpose2d(yt_par_LH, torch.mm(self.frequency_transform_L_inverse, self.frequency_transform_H_inverse.permute(1,0)).unsqueeze(0).view(1,1, 2, 2), stride=2) + \
				 F.conv_transpose2d(yt_par_HH, torch.mm(self.frequency_transform_H_inverse, self.frequency_transform_H_inverse.permute(1,0)).unsqueeze(0).view(1,1, 2, 2), stride=2)
		yt_par = yt_par.view(batch_size, channel, height * 2, width * 2)
		# yt_par = yt_par * 0.25
		yt_par = self.color_transform_inverse(yt_par)

		return final_output2, yt_par



@NETWORK_REGISTRY.register()
class EMVD_network8_level_raw(nn.Module):
	def __init__(self, C = 4):
		super(EMVD_network8_level_raw, self).__init__()
		self.C = C
		self.color_transform = nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0, bias=False)
		self.color_transform_inverse = nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0, bias=False)
		self.frequency_transform_L = nn.Parameter(torch.Tensor([0.7071067, 0.7071067]).view(2, 1))
		self.frequency_transform_H = nn.Parameter(torch.Tensor([-0.7071067, 0.7071067]).view(2, 1))

		self.frequency_transform_L_inverse = nn.Parameter(torch.Tensor([0.7071067, 0.7071067]).view(2, 1))
		self.frequency_transform_H_inverse = nn.Parameter(torch.Tensor([-0.7071067, 0.7071067]).view(2, 1))

		self.fcnn1 = FCNN(C + 1)
		self.fcnn = FCNN(C+1+1)
		self.scale_for_fusion = 2
		self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

		self.denoise = DNCNN(4*C+C+C+1, 4*C)
		self.refine = RCNN(4*C+4*C+1)

		# self.reset_params()

		color_matrix = torch.Tensor([[0.5, 0.5, 0.5, 0.5], [-0.5, 0.5, 0.5, -0.5],[0.65, 0.2784,-0.2784, -0.65],[-0.2784,0.65,-0.65,0.2784]]).float()
		color_matrix = color_matrix.view(C, C, 1, 1)
		self.color_transform.weight.data = color_matrix

		color_matrix = torch.Tensor([[0.5, 0.5, 0.5, 0.5], [-0.5, 0.5, 0.5, -0.5], [0.65, 0.2784, -0.2784, -0.65],[-0.2784, 0.65, -0.65, 0.2784]]).float()
		color_matrix = color_matrix.view(C, C)
		color_matrix_inverse = torch.inverse(color_matrix)
		color_matrix_inverse = color_matrix_inverse.view(C, C, 1, 1)
		self.color_transform_inverse.weight.data = color_matrix_inverse
		# self.color_transform_inverse.weight.data =  torch.inverse(self.color_transform_inverse.weight.data)

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def load_pretrain_model(self, model_path):
		if model_path is not None and model_path != "":
			state_dict = torch.load(model_path, map_location=torch.device('cpu'))['state_dict']
		state_dict_tmp = {}
		for n, p in state_dict.items():
			if 'module' in n:
				state_dict_tmp[n[7:]] = p
			else:
				state_dict_tmp[n] = p
		key_tmp = set(list(state_dict_tmp.keys()))
		self.load_state_dict(state_dict_tmp)
		print("Load checkpoint {} successfully!".format(model_path))

	def forward(self, zts=None, return_weight=False, a_value=0, b_value=0):
		b, n, c, h, w = zts.size()
		outputs = []
		yt_1 = zts[:, 0, :, :, :]
		for i in range(n):
			if return_weight:
				conv_weight_list1 = [self.color_transform.weight, self.color_transform_inverse.weight]
				conv_weight_list2 = [self.frequency_transform_L, self.frequency_transform_H]
				conv_weight_list3 = [self.frequency_transform_L_inverse, self.frequency_transform_H_inverse]
				return conv_weight_list1, conv_weight_list2, conv_weight_list3
			zt = zts[:, i, :, :, :]
			zt = self.color_transform(zt)
			batch_size = zt.shape[0]
			channel = zt.shape[1]
			height = zt.shape[2]
			width = zt.shape[3]
			zt = zt.view(batch_size * channel, 1, height, width)
			zt_LL = F.conv2d(zt, torch.mm(self.frequency_transform_L,self.frequency_transform_L.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
			zt_HL = F.conv2d(zt, torch.mm(self.frequency_transform_H,self.frequency_transform_L.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
			zt_LH = F.conv2d(zt, torch.mm(self.frequency_transform_L,self.frequency_transform_H.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
			zt_HH = F.conv2d(zt, torch.mm(self.frequency_transform_H,self.frequency_transform_H.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)

			yt_1 = self.color_transform(yt_1)
			yt_1 = yt_1.view(batch_size * channel, 1, height, width)
			yt_1_LL = F.conv2d(yt_1, torch.mm(self.frequency_transform_L,self.frequency_transform_L.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
			yt_1_HL = F.conv2d(yt_1, torch.mm(self.frequency_transform_H,self.frequency_transform_L.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
			yt_1_LH = F.conv2d(yt_1, torch.mm(self.frequency_transform_L,self.frequency_transform_H.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
			yt_1_HH = F.conv2d(yt_1, torch.mm(self.frequency_transform_H,self.frequency_transform_H.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)


			zt_LL_fuse = zt_LL
			yt_1_LL_fuse = yt_1_LL
			zt_list = []
			yt_list = []
			for mm in range(self.scale_for_fusion):
				batch_size = zt_LL_fuse.shape[0]
				channel = zt_LL_fuse.shape[1]
				height = zt_LL_fuse.shape[2]
				width = zt_LL_fuse.shape[3]

				zt_LL_fuse = zt_LL_fuse.view(batch_size*channel, 1, height, width)
				zt_LL_fuse_this = F.conv2d(zt_LL_fuse, torch.mm(self.frequency_transform_L,self.frequency_transform_L.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
				zt_LH_fuse = F.conv2d(zt_LL_fuse, torch.mm(self.frequency_transform_L,self.frequency_transform_H.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
				zt_HL_fuse = F.conv2d(zt_LL_fuse, torch.mm(self.frequency_transform_H,self.frequency_transform_L.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
				zt_HH_fuse = F.conv2d(zt_LL_fuse, torch.mm(self.frequency_transform_H,self.frequency_transform_H.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)

				yt_1_LL_fuse = yt_1_LL_fuse.view(batch_size * channel, 1, height, width)
				yt_1_LL_fuse_this = F.conv2d(yt_1_LL_fuse, torch.mm(self.frequency_transform_L,self.frequency_transform_L.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
				yt_1_LH_fuse = F.conv2d(yt_1_LL_fuse, torch.mm(self.frequency_transform_L,self.frequency_transform_H.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
				yt_1_HL_fuse = F.conv2d(yt_1_LL_fuse, torch.mm(self.frequency_transform_H,self.frequency_transform_L.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)
				yt_1_HH_fuse = F.conv2d(yt_1_LL_fuse, torch.mm(self.frequency_transform_H,self.frequency_transform_H.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2).view(batch_size, channel, height//2, width//2)

				zt_list.append([zt_LL_fuse_this, zt_LH_fuse, zt_HL_fuse, zt_HH_fuse])
				yt_list.append([yt_1_LL_fuse_this, yt_1_LH_fuse, yt_1_HL_fuse, yt_1_HH_fuse])
				zt_LL_fuse = zt_LL_fuse_this
				yt_1_LL_fuse = yt_1_LL_fuse_this


			for mm in range(self.scale_for_fusion):
				yt_1_LL_this, yt_1_LH_this, yt_1_HL_this, yt_1_HH_this = yt_list[self.scale_for_fusion-mm-1]
				zt_LL_this, zt_LH_this, zt_HL_this, zt_HH_this = zt_list[self.scale_for_fusion-mm-1]

				if mm == 0:
					zt_input = zt_LL_this

				sigma_t = torch.mean(zt_LL_this, dim=1).unsqueeze(dim=1) * a_value + b_value
				# print(torch.mean(sigma_t), torch.max(sigma_t), torch.min(sigma_t))
				# gamma_t_this = gamma_t[:, 0:1, :, :]
				if mm == 0:
					gammt_input = torch.cat([torch.abs(zt_LL_this - yt_1_LL_this), sigma_t], dim=1)
					gamma_t = self.fcnn1.forward(gammt_input)
					gamma_t = nn.Sigmoid()(gamma_t)
				else:
					gammt_input = torch.cat([torch.abs(zt_LL_this - yt_1_LL_this), gamma_t, sigma_t], dim=1)
					gamma_t = self.fcnn.forward(gammt_input)
					gamma_t = nn.Sigmoid()(gamma_t)

				channel = zt_LL_this.shape[1]
				gamma_t_this = gamma_t.repeat(1, channel, 1, 1)
				yt_par_LL_this = yt_1_LL_this * (1 - gamma_t_this) + zt_LL_this * gamma_t_this
				yt_par_LH_this = yt_1_LH_this * (1 - gamma_t_this) + zt_LH_this * gamma_t_this
				yt_par_HL_this = yt_1_HL_this * (1 - gamma_t_this) + zt_HL_this * gamma_t_this
				yt_par_HH_this = yt_1_HH_this * (1 - gamma_t_this) + zt_HH_this * gamma_t_this

				yt_par_input = torch.cat([yt_par_LL_this, yt_par_HL_this, yt_par_LH_this, yt_par_HH_this], dim=1)
				sigma_t_1 = torch.mean(yt_1_LL_this, dim=1).unsqueeze(dim=1) * a_value + b_value
				sigma_t_2 = torch.mean(zt_LL_this, dim=1).unsqueeze(dim=1) * a_value + b_value

				# gamma_t_this = gamma_t[:, 0:1, :, :]
				sigma_input = (1 - gamma_t) * (1 - gamma_t) * sigma_t_1 + gamma_t * gamma_t * sigma_t_2
				# print(torch.mean(sigma_t_1), torch.mean(sigma_t_2), 'sigma')
				denoise_input = torch.cat([yt_par_input, zt_LL_this, zt_input, sigma_input], dim=1)
				yt_war = self.denoise.forward(denoise_input)

				zt_input_LL = yt_war[:, 0:self.C, :, :]
				zt_input_HL = yt_war[:, self.C:self.C * 2, :, :]
				zt_input_LH = yt_war[:, self.C * 2:self.C * 3, :, :]
				zt_input_HH = yt_war[:, self.C * 3:self.C * 4, :, :]

				batch_size = zt_input_LL.shape[0]
				channel = zt_input_LL.shape[1]
				height = zt_input_LL.shape[2]
				width = zt_input_LL.shape[3]
				zt_input_LL = zt_input_LL.contiguous().view(batch_size * channel, 1, height, width)
				zt_input_HL = zt_input_HL.contiguous().view(batch_size * channel, 1, height, width)
				zt_input_LH = zt_input_LH.contiguous().view(batch_size * channel, 1, height, width)
				zt_input_HH = zt_input_HH.contiguous().view(batch_size * channel, 1, height, width)

				zt_input = F.conv_transpose2d(zt_input_LL, torch.mm(self.frequency_transform_L_inverse, self.frequency_transform_L_inverse.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2) + \
						   F.conv_transpose2d(zt_input_HL, torch.mm(self.frequency_transform_H_inverse, self.frequency_transform_L_inverse.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2) + \
						   F.conv_transpose2d(zt_input_LH, torch.mm(self.frequency_transform_L_inverse, self.frequency_transform_H_inverse.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2) + \
						   F.conv_transpose2d(zt_input_HH, torch.mm(self.frequency_transform_H_inverse, self.frequency_transform_H_inverse.permute(1, 0)).unsqueeze(0).view(1, 1, 2, 2), stride=2)
				zt_input = zt_input.view(batch_size, channel, height*2, width*2)
				# zt_input = zt_input * 0.25
				gamma_t = self.upsample(gamma_t)

			sigma_t = torch.mean(zt_LL, dim=1).unsqueeze(dim=1) * a_value + b_value
			gammt_input = torch.cat([torch.abs(zt_LL - yt_1_LL), gamma_t, sigma_t], dim=1)
			gamma_t = self.fcnn.forward(gammt_input)
			gamma_t = nn.Sigmoid()(gamma_t)

			channel = zt_LL.shape[1]
			gamma_t_this = gamma_t.repeat(1, channel, 1, 1)
			yt_par_LL = yt_1_LL * (1 - gamma_t_this) + zt_LL * gamma_t_this
			yt_par_LH = yt_1_LH * (1 - gamma_t_this) + zt_LH * gamma_t_this
			yt_par_HL = yt_1_HL * (1 - gamma_t_this) + zt_HL * gamma_t_this
			yt_par_HH = yt_1_HH * (1 - gamma_t_this) + zt_HH * gamma_t_this

			yt_par_input = torch.cat([yt_par_LL, yt_par_HL, yt_par_LH, yt_par_HH], dim=1)
			sigma_t_1 = torch.mean(yt_1_LL, dim=1).unsqueeze(dim=1) * a_value + b_value
			sigma_t_2 = torch.mean(zt_LL, dim=1).unsqueeze(dim=1) * a_value + b_value
			# gamma_t_this = gamma_t[:, 0:1, :, :]
			sigma_input = (1 - gamma_t) * (1 - gamma_t) * sigma_t_1 + gamma_t * gamma_t * sigma_t_2
			denoise_input = torch.cat([yt_par_input, zt_LL, zt_input, sigma_input], dim=1)
			yt_war = self.denoise.forward(denoise_input)

			refine_input = torch.cat([yt_war, yt_par_input, sigma_input], dim=1)
			refine_output = self.refine.forward(refine_input)
			refine_output = nn.Sigmoid()(refine_output)
			channel = yt_par_input.shape[1]
			refine_output = refine_output.repeat(1, channel, 1, 1)
			final_output = yt_par_input * refine_output + yt_war * (1-refine_output)

			# print(torch.mean(final_output), 'output1')
			final_output_LL = final_output[:, 0:self.C, :, :]
			final_output_HL = final_output[:, self.C:self.C*2, :, :]
			final_output_LH = final_output[:, self.C*2:self.C * 3, :, :]
			final_output_HH = final_output[:, self.C*3:self.C * 4, :, :]

			batch_size = final_output_LL.shape[0]
			channel = final_output_LL.shape[1]
			height = final_output_LL.shape[2]
			width = final_output_LL.shape[3]
			final_output_LL = final_output_LL.contiguous().view(batch_size * channel, 1, height, width)
			final_output_HL = final_output_HL.contiguous().view(batch_size * channel, 1, height, width)
			final_output_LH = final_output_LH.contiguous().view(batch_size * channel, 1, height, width)
			final_output_HH = final_output_HH.contiguous().view(batch_size * channel, 1, height, width)

			final_output1 = F.conv_transpose2d(final_output_LL, torch.mm(self.frequency_transform_L_inverse, self.frequency_transform_L_inverse.permute(1, 0)).unsqueeze(0).view(1,1, 2, 2), stride=2) + \
							F.conv_transpose2d(final_output_HL, torch.mm(self.frequency_transform_H_inverse, self.frequency_transform_L_inverse.permute(1, 0)).unsqueeze(0).view(1,1, 2, 2), stride=2) + \
							F.conv_transpose2d(final_output_LH, torch.mm(self.frequency_transform_L_inverse, self.frequency_transform_H_inverse.permute(1, 0)).unsqueeze(0).view(1,1, 2, 2), stride=2) + \
							F.conv_transpose2d(final_output_HH, torch.mm(self.frequency_transform_H_inverse, self.frequency_transform_H_inverse.permute(1, 0)).unsqueeze(0).view(1,1, 2, 2), stride=2)
			final_output1 = final_output1.view(batch_size, channel, height*2, width*2)
			# final_output1 = final_output1 * 0.25
			final_output2 = self.color_transform_inverse(final_output1)

			batch_size = yt_par_LL.shape[0]
			channel = yt_par_LL.shape[1]
			height = yt_par_LL.shape[2]
			width = yt_par_LL.shape[3]
			yt_par_LL = yt_par_LL.contiguous().view(batch_size * channel, 1, height, width)
			yt_par_HL = yt_par_HL.contiguous().view(batch_size * channel, 1, height, width)
			yt_par_LH = yt_par_LH.contiguous().view(batch_size * channel, 1, height, width)
			yt_par_HH = yt_par_HH.contiguous().view(batch_size * channel, 1, height, width)

			yt_par = F.conv_transpose2d(yt_par_LL, torch.mm(self.frequency_transform_L_inverse, self.frequency_transform_L_inverse.permute(1,0)).unsqueeze(0).view(1,1, 2, 2), stride=2) + \
					 F.conv_transpose2d(yt_par_HL, torch.mm(self.frequency_transform_H_inverse, self.frequency_transform_L_inverse.permute(1,0)).unsqueeze(0).view(1,1, 2, 2), stride=2) + \
					 F.conv_transpose2d(yt_par_LH, torch.mm(self.frequency_transform_L_inverse, self.frequency_transform_H_inverse.permute(1,0)).unsqueeze(0).view(1,1, 2, 2), stride=2) + \
					 F.conv_transpose2d(yt_par_HH, torch.mm(self.frequency_transform_H_inverse, self.frequency_transform_H_inverse.permute(1,0)).unsqueeze(0).view(1,1, 2, 2), stride=2)
			yt_par = yt_par.view(batch_size, channel, height * 2, width * 2)
			# yt_par = yt_par * 0.25
			yt_par = self.color_transform_inverse(yt_par)

			yt_1 = yt_par
			outputs.append(final_output2)
		return torch.stack(outputs, dim = 1)
