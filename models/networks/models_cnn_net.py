import torch
import torch.nn as nn
from .base import BaseNet
from ..modules import SE, GhostBottleneck, ResnetBlock
from ..modules.pixelshuffle_module import PixelShuffleAlign, SubPixelDown
from ..utils.warp import flow_warp, warp_fn
from utils.registry import NETWORK_REGISTRY
@NETWORK_REGISTRY.register()
class ModifiedDnCNN(BaseNet):
    def __init__(self, input_channels=3*15, output_channels=3, nlconv_features=96, nlconv_layers=4, dnnconv_features=192, dnnconv_layers=15):
        super(ModifiedDnCNN, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.nlconv_features = nlconv_features
        self.nlconv_layers = nlconv_layers
        self.dnnconv_features = dnnconv_features
        self.dnnconv_layers = dnnconv_layers

        layers = []
        layers.append(nn.Conv2d(in_channels=self.input_channels,\
                                out_channels=self.nlconv_features,\
                                kernel_size=1,\
                                padding=0,\
                                bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(self.nlconv_layers-1):
            layers.append(nn.Conv2d(in_channels=self.nlconv_features,\
                                    out_channels=self.nlconv_features,\
                                    kernel_size=1,\
                                    padding=0,\
                                    bias=True))
            layers.append(nn.ReLU(inplace=True))
        # Shorter DnCNN
        layers.append(nn.Conv2d(in_channels=self.nlconv_features,\
                                out_channels=self.dnnconv_features,\
                                kernel_size=3,\
                                padding=1,\
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(self.dnnconv_layers-2):
            layers.append(nn.Conv2d(in_channels=self.dnnconv_features,\
                                    out_channels=self.dnnconv_features,\
                                    kernel_size=3,\
                                    padding=1,\
                                    bias=False))
            layers.append(nn.BatchNorm2d(self.dnnconv_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.dnnconv_features,\
                                out_channels=self.output_channels,\
                                kernel_size=3,\
                                padding=1,\
                                bias=False))
        self.net = nn.Sequential(*layers)
    def load_pretrain_model(self, model_path):
        if model_path is not None and model_path != "":
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))

            state_dict_tmp = {}
            for n, p in state_dict.items():
                if 'module' in n:
                    state_dict_tmp[n[7:]] = p
                else:
                    state_dict_tmp[n] = p
            key_tmp = set(list(state_dict_tmp.keys()))

            #for n, p in state_dict.items():

            #    if n in key_tmp:
            #        key_tmp.remove(n)
            #    else:
            #        print('%s not exist, pass!' % n)

            #   pretrain_weight = p.data
            #    if state_dict_tmp[n].shape != pretrain_weight.shape:
            #        print("%s size mismatch, loading selected kernel!" % n)
            #        pretrain_weight = self.select_pretrain_kernel(pretrain_weight, state_dict_tmp[n].data)

            #    state_dict_tmp[n].copy_(pretrain_weight)

            #if len(key_tmp) != 0:
            #    for k in key_tmp:
            #        print("param %s not found in pretrain model!" % k)

            self.load_state_dict(state_dict_tmp)
            print("Load checkpoint {} successfully!".format(model_path))
    def forward(self, x):
        # x = x.unsqueeze(1).repeat(1, 15, 1, 1, 1)
        b, n, c, h, w = x.size()
        x = x.contiguous()
        x = x.view(b, n*c, h, w)
        out = self.net(x)
        return out

@NETWORK_REGISTRY.register()
class ModifiedDnCNN_level(nn.Module):
    def __init__(self, input_channels=3*15, output_channels=3, nlconv_features=96, nlconv_layers=4, dnnconv_features=192, dnnconv_layers=15):
        super(ModifiedDnCNN_level, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.nlconv_features = nlconv_features
        self.nlconv_layers = nlconv_layers
        self.dnnconv_features = dnnconv_features
        self.dnnconv_layers = dnnconv_layers

        layers = []
        layers.append(nn.Conv2d(in_channels=self.input_channels+self.input_channels//3,\
                                out_channels=self.nlconv_features,\
                                kernel_size=1,\
                                padding=0,\
                                bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(self.nlconv_layers-1):
            layers.append(nn.Conv2d(in_channels=self.nlconv_features,\
                                    out_channels=self.nlconv_features,\
                                    kernel_size=1,\
                                    padding=0,\
                                    bias=True))
            layers.append(nn.ReLU(inplace=True))
        # Shorter DnCNN
        layers.append(nn.Conv2d(in_channels=self.nlconv_features,\
                                out_channels=self.dnnconv_features,\
                                kernel_size=3,\
                                padding=1,\
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(self.dnnconv_layers-2):
            layers.append(nn.Conv2d(in_channels=self.dnnconv_features,\
                                    out_channels=self.dnnconv_features,\
                                    kernel_size=3,\
                                    padding=1,\
                                    bias=False))
            layers.append(nn.BatchNorm2d(self.dnnconv_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.dnnconv_features,\
                                out_channels=self.output_channels,\
                                kernel_size=3,\
                                padding=1,\
                                bias=False))
        self.net = nn.Sequential(*layers)
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

            #for n, p in state_dict.items():

            #    if n in key_tmp:
            #        key_tmp.remove(n)
            #    else:
            #        print('%s not exist, pass!' % n)

            #   pretrain_weight = p.data
            #    if state_dict_tmp[n].shape != pretrain_weight.shape:
            #        print("%s size mismatch, loading selected kernel!" % n)
            #        pretrain_weight = self.select_pretrain_kernel(pretrain_weight, state_dict_tmp[n].data)

            #    state_dict_tmp[n].copy_(pretrain_weight)

            #if len(key_tmp) != 0:
            #    for k in key_tmp:
            #        print("param %s not found in pretrain model!" % k)

            self.load_state_dict(state_dict_tmp)
            print("Load checkpoint {} successfully!".format(model_path))
    def forward(self, x, noise_map):
        # x = x.unsqueeze(1).repeat(1, 15, 1, 1, 1)
        b, n, c, h, w = x.size()
        x = x.view(b, n*c, h, w)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x3 = x[:, 6:9, :, :]
        x4 = x[:, 9:12, :, :]
        x5 = x[:, 12:15, :, :]
        x6 = x[:, 15:18, :, :]
        x7 = x[:, 18:21, :, :]
        x8 = x[:, 21:24, :, :]
        x9 = x[:, 24:27, :, :]
        x10 = x[:, 27:30, :, :]
        x11 = x[:, 30:33, :, :]
        x12 = x[:, 33:36, :, :]
        x13 = x[:, 36:39, :, :]
        x14 = x[:, 39:42, :, :]
        x15 = x[:, 42:45, :, :]

        x0 = torch.cat([x1, noise_map, x2, noise_map, x3, noise_map,
                        x4, noise_map, x5, noise_map, x6, noise_map,
                        x7, noise_map, x8, noise_map, x9, noise_map,
                        x10, noise_map, x11, noise_map, x12, noise_map,
                        x13, noise_map, x14, noise_map, x15, noise_map], dim=1)

        out = self.net(x0)
        return out

