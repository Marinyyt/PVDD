import torch
import torch.nn as nn
from torchvision import models


class MeanModule(nn.Module):
    
    def __init__(self, use_gray):
        super(MeanModule, self).__init__()
        self.use_gray = use_gray
        if use_gray:
            self.mean = torch.tensor([110.7919485, 110.7919485, 110.7919485]).view(1,3,1,1)
        else:
            self.mean = torch.tensor(
                [129.1863, 104.7624, 93.5940]).view(1, 3, 1, 1)
    def forward(self, x):
        x = x * 255
        mean = self.mean.type_as(x)
        if self.use_gray:
            x = x.repeat(1,3,1,1)
        mean = mean.repeat(x.shape[0],1,x.shape[2],x.shape[3])
     
        y = x - mean
        return y


# class VGGFace(nn.Module):
#
#     def __init__(self, model_path, feature_target, use_gray):
#         '''
#         VGG Face model, only contains layer 0 to layer 6 against VGG_FACE.t7
#         '''
#         super(VGGFace, self).__init__()
#
#         self.preprocess = nn.Sequential(*[MeanModule(use_gray)])
#
#         self.feature = [
#                         nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),
#                         nn.ReLU(),
#                         nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
#                         nn.ReLU(),
#                         nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#                         nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
#                         nn.ReLU(),
#                         ]
#
#         self.feature = nn.Sequential(*self.feature)
#         self.load_model(model_path)
#
#         # self.conv1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))
#         # self.relu1 = nn.ReLU()
#         # self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
#         # self.relu2 = nn.ReLU()
#         # self.maxpool2 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
#         # self.conv3 = nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1))
#         # self.relu3 = nn.ReLU()
#
#     def forward(self, x):
#         # return self.feature(x)
#         # feature = {}
#         # feature['conv1'] = self.conv1(x)
#         # feature['relu1'] = self.relu1(feature['conv1'])
#         # feature['conv2'] = self.conv2(feature['relu1'])
#         # feature['relu2'] = self.relu2(feature['conv2'])
#         # feature['maxpool2'] = self.maxpool2(feature['relu2'])
#         # feature['conv3'] = self.conv3(feature['maxpool2'])
#         # feature['relu3'] = self.relu3(feature['conv3'])
#         return self.feature(self.preprocess(x))
#
#     def load_model(self, model_path):
#         # Load pretrained VGG Face model
#         vgg_dict = self.feature.state_dict()
#         pretrained_dict = torch.load(model_path)
#         # pretrained_dict = {"feature." + k: v for k, v in pretrained_dict.items() if "feature." + k in vgg_dict.keys()}
#         # vgg_dict.update(pretrained_dict)
#         keys = vgg_dict.keys()
#         pretrained_keys = pretrained_dict.keys()
#         for k, pk in zip(keys, pretrained_keys):
#             vgg_dict[k] = pretrained_dict[pk]
#         self.feature.load_state_dict(vgg_dict)


class Subvgg(nn.Module):
    def __init__(self, feature_target):
        super(Subvgg, self).__init__()
        # for reference using low level texture matching loss
        if 'conv2_2' in feature_target:
            self.slice1 = torch.nn.Sequential(
                nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),
                nn.ReLU(),
                nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
                nn.ReLU(),
                # conv1_2
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
                nn.ReLU(),
                nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
                # nn.ReLU(),
                # conv2_2
            )
        if 'conv3_2' in feature_target:
            self.slice2 = torch.nn.Sequential(
                nn.ReLU(),
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
                nn.ReLU(),
                nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
                # nn.ReLU(),
                # conv3_2
            )
        if 'conv4_2' in feature_target:
            self.slice3 = torch.nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
                nn.ReLU(),
                # conv3_3
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
                nn.ReLU(),
                nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
                # nn.ReLU(),
                # conv4_2
            )


class VGGFace(torch.nn.Module):
    def __init__(self, model_path, feature_target, use_gray):
        super(VGGFace, self).__init__()
        # use feature conv3_2 and conv4_2
        self.preprocess = nn.Sequential(*[MeanModule(use_gray)])
        self.features = Subvgg(feature_target)
        self.load_model(model_path)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        x = self.downsample(x)
        # conv2_2
        h_relu1 = self.features.slice1(self.preprocess(x))
        # conv3_2
        h_relu2 = self.features.slice2(h_relu1)
        # conv4_2
        h_relu3 = self.features.slice3(h_relu2)
        out = [h_relu1, h_relu2, h_relu3]
        return out

    def load_model(self, model_path):
        # Load pretrained VGG Face model
        vgg_dict = self.features.state_dict()
        pretrained_dict = torch.load(model_path)
        # pretrained_dict = {"feature." + k: v for k, v in pretrained_dict.items() if "feature." + k in vgg_dict.keys()}
        # vgg_dict.update(pretrained_dict)
        keys1 = vgg_dict.keys()
        pretrained_keys = pretrained_dict.keys()
        for k, pk in zip(keys1, pretrained_keys):
            vgg_dict[k] = pretrained_dict[pk]
        self.features.load_state_dict(vgg_dict)


# add vgg19 model for perceptual loss
class VGG19(torch.nn.Module):
    def __init__(self, model_path, requires_grad=False):
        super(VGG19, self).__init__()
        vgg19 = models.vgg19(pretrained=False)
        vgg19.load_state_dict(torch.load(model_path))
        vgg_pretrained_features = vgg19.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()   
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
           for param in self.parameters():
               param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

