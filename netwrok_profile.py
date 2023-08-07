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

from models.networks.pvdd0815_net import pvdd0815
from models.networks.models_rvidenet_srgb_net import RViDeNet
from models.networks.models_no_level_net import FastDVDnet
from models.networks.models_cnn_net import ModifiedDnCNN
from models.networks.EDVR_arch_net import EDVR
import time
import glob
import thop
from models.networks.EMVD_no_level_net import EMVD_network8
#import hiddenlayer as h
# model_temp = shuffleVR(n_chs=16, in_blocks=1, fuse_blocks=1, out_blocks=1)
# vis_graph = h.build_graph(model_temp, torch.zeros([1 ,2, 1, 28, 28]))   # 获取绘制图像的对象
# vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
# vis_graph.save("./demo1.png")



print('Loading models ...')
model_pvdd = pvdd0815(          # input_nc: 3
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
          mlp = '04').cuda()


model_rvi = RViDeNet().cuda()
model_fast = FastDVDnet().cuda()
model_cnn = ModifiedDnCNN().cuda()
model_edvr = EDVR().cuda()
model_emvd = EMVD_network8().cuda()
'''
x = torch.randn(1, 3, 128, 128).cuda()

start_time = time.time()
y = model_pvdd(x)
end_time = time.time()
print('pvdd: {}'.format(end_time - start_time))

start_time = time.time()
y = model_rvi(x)
end_time = time.time()
print('rvi: {}'.format(end_time - start_time))


start_time = time.time()
y = model_cnn(x)
end_time = time.time()
print('cnn: {}'.format(end_time - start_time))



start_time = time.time()
y = model_fast(x)
end_time = time.time()
print('fast: {}'.format(end_time - start_time))


start_time = time.time()
y = model_cnn(x)
end_time = time.time()
print('cnn: {}'.format(end_time - start_time))



start_time = time.time()
y = model_edvr(x)
end_time = time.time()
print('edvr: {}'.format(end_time - start_time))
'''

'''
x = torch.randn(1, 3, 256, 256).cuda()
flops, params = thop.profile(model_emvd, inputs = (x, x,))
flops, params = thop.clever_format([flops, params], "%.3f")
print('pvdd-flops: {}, pvdd-params: {}'.format(flops, params))


x = torch.randn(1, 3, 256, 256).cuda()
flops, params = thop.profile(model_edvr, inputs = (x,))
flops, params = thop.clever_format([flops, params], "%.3f")
print('edvr-flops: {}, edvr-params: {}'.format(flops, params))


x = torch.randn(1, 3, 256, 256).cuda()
flops, params = thop.profile(model_cnn, inputs = (x,))
flops, params = thop.clever_format([flops, params], "%.3f")
print('cnn-flops: {}, cnn-params: {}'.format(flops, params))
'''

x = torch.randn(1, 3, 256, 256).cuda()
flops, params = thop.profile(model_fast, inputs = (x,))
flops, params = thop.clever_format([flops, params], "%.3f")
print('fast-flops: {}, fast-params: {}'.format(flops, params))


x = torch.randn(1, 3, 256, 256).cuda()
flops, params = thop.profile(model_rvi, inputs = (x,))
flops, params = thop.clever_format([flops, params], "%.3f")
print('rvi-flops: {}, rvi-params: {}'.format(flops, params))
