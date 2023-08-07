import torch
import torch.nn as nn

from .base import BaseNet
from utils.registry import NETWORK_REGISTRY

@NETWORK_REGISTRY.register()
class ModifiedDnCNNraw(BaseNet):
    def __init__(self, input_channels=45, output_channels=3, nlconv_features=96, nlconv_layers=4, dnnconv_features=192, dnnconv_layers=15):
        super(ModifiedDnCNNraw, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.nlconv_features = nlconv_features
        self.nlconv_layers = nlconv_layers
        self.dnnconv_features = dnnconv_features
        self.dnnconv_layers = dnnconv_layers

        self.input_channels = 4*15
        self.output_channels = 4

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
    def forward(self, x):
        b, n, c, h, w = x.size()
        x = x.view(b, -1, h, w)
        out = self.net(x)
        return out



class ModifiedDnCNN_levelraw(BaseNet):
    def __init__(self, input_channels=45, output_channels=3, nlconv_features=96, nlconv_layers=4, dnnconv_features=192, dnnconv_layers=15):
        super(ModifiedDnCNN_levelraw, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.nlconv_features = nlconv_features
        self.nlconv_layers = nlconv_layers
        self.dnnconv_features = dnnconv_features
        self.dnnconv_layers = dnnconv_layers

        self.input_channels = 4*15
        self.output_channels = 4

        layers = []
        layers.append(nn.Conv2d(in_channels=self.input_channels+self.input_channels//4,\
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
        b, n, c, h, w = x.size()
        x = x.view(b, -1, h, w)
        x1 = x[:, 0:4, :, :]
        x2 = x[:, 4:8, :, :]
        x3 = x[:, 8:12, :, :]
        x4 = x[:, 12:16, :, :]
        x5 = x[:, 16:20, :, :]
        x6 = x[:, 20:24, :, :]
        x7 = x[:, 24:28, :, :]
        x8 = x[:, 28:32, :, :]
        x9 = x[:, 32:36, :, :]
        x10 = x[:, 36:40, :, :]
        x11 = x[:, 40:44, :, :]
        x12 = x[:, 44:48, :, :]
        x13 = x[:, 48:52, :, :]
        x14 = x[:, 52:56, :, :]
        x15 = x[:, 56:60, :, :]

        x0 = torch.cat([x1, noise_map, x2, noise_map, x3, noise_map,
                        x4, noise_map, x5, noise_map, x6, noise_map,
                        x7, noise_map, x8, noise_map, x9, noise_map,
                        x10, noise_map, x11, noise_map, x12, noise_map,
                        x13, noise_map, x14, noise_map, x15, noise_map], dim=1)

        out = self.net(x0)
        return out

