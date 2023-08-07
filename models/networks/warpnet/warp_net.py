import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp

_pretrain_model = osp.join(osp.dirname(osp.abspath(__file__)), 'ckpts', '512', 'improve', 'warpnet-epoch-8.pkl')


class Warpnet_512(nn.Module):

    def __init__(self, image_size=512, out_channels=64, use_gray=False, interp_model=False):
        super(Warpnet_512, self).__init__()
        self.image_size = image_size
        self.use_gray = use_gray
        self.encoder = nn.Sequential(
            # 512 x 512 x 6
            nn.Conv2d(in_channels=6, out_channels=out_channels,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            # 256 x 256 x 64
            nn.Conv2d(out_channels, out_channels * 2,
                      kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels * 2),
            nn.LeakyReLU(0.2, inplace=False),
            # 128 x 128 x 128
            nn.Conv2d(out_channels * 2, out_channels * 2,
                      kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels * 2),
            nn.LeakyReLU(0.2, inplace=False),
            # 64 x 64 x 128
            nn.Conv2d(out_channels * 2, out_channels * 2,
                      kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels * 2),
            nn.LeakyReLU(0.2, inplace=False),
            # 32 x 32 x 128
            nn.Conv2d(out_channels * 2, out_channels * 2,
                      kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels * 2),
            nn.LeakyReLU(0.2, inplace=False),
            # 16 x 16 x 128
            nn.Conv2d(out_channels * 2, out_channels * 2,
                      kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels * 2),
            nn.LeakyReLU(0.2, inplace=False),
            # 8 x 8 x 128
            nn.Conv2d(out_channels * 2, out_channels * 4,
                      kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels * 4),
            nn.LeakyReLU(0.2, inplace=False),
            # 4 x 4 x 256
            nn.Conv2d(out_channels * 4, out_channels * 4,
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False)
        )
        if interp_model:
            self.decoder = nn.Sequential(
                # 2 x 2 x 256
                nn.Upsample(size=(4, 4), mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels=out_channels * 4,
                          out_channels=out_channels * 4, kernel_size=1, stride=1),
                nn.InstanceNorm2d(out_channels * 4),
                nn.ReLU(inplace=False),
                # 4 x 4 x 256

                nn.Upsample(size=(8, 8), mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels=out_channels * 4,
                          out_channels=out_channels * 2, kernel_size=1, stride=1),
                nn.InstanceNorm2d(out_channels * 2),
                nn.ReLU(inplace=False),

                # 8 x 8 x 128
                nn.Upsample(size=(16, 16), mode='bilinear',
                            align_corners=True),
                nn.Conv2d(in_channels=out_channels * 2,
                          out_channels=out_channels * 2, kernel_size=1, stride=1),
                nn.InstanceNorm2d(out_channels * 2),
                nn.ReLU(inplace=False),

                # 16 x 16 x 128
                nn.Upsample(size=(32, 32), mode='bilinear',
                            align_corners=True),
                nn.Conv2d(in_channels=out_channels * 2,
                          out_channels=out_channels * 2, kernel_size=1, stride=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=False),
                # 32 x 32 x 128
                nn.Upsample(size=(128, 128), mode='bilinear',
                            align_corners=True),
                nn.Conv2d(in_channels=out_channels * 2,
                          out_channels=out_channels, kernel_size=1, stride=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=False),
                # 128 x 128 x 64
                nn.Upsample(size=(256, 256), mode='bilinear',
                            align_corners=True),
                nn.Conv2d(in_channels=out_channels,
                          out_channels=out_channels, kernel_size=1, stride=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=False),
                # 256 x 256 x 64
                nn.Upsample(size=(self.image_size, self.image_size),
                            mode='bilinear', align_corners=True),
                # 512 * 512 * 64
                nn.Conv2d(in_channels=out_channels, out_channels=2,
                          kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            )

        else:
            self.decoder = nn.Sequential(
                # 2 x 2 x 256
                nn.ConvTranspose2d(
                    out_channels * 4, out_channels * 4, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels * 4),
                nn.ReLU(inplace=False),
                # 4 x 4 x 256
                nn.ConvTranspose2d(
                    out_channels * 4, out_channels * 2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels * 2),
                nn.ReLU(inplace=False),
                # 8 x 8 x 128
                nn.ConvTranspose2d(
                    out_channels * 2, out_channels * 2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels * 2),
                nn.ReLU(inplace=False),
                # 16 x 16 x 128
                nn.Upsample(size=(32, 32), mode='bilinear'),
                nn.Conv2d(in_channels=out_channels * 2,
                          out_channels=out_channels * 2, kernel_size=1, stride=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=False),
                # 32 x 32 x 128
                nn.Upsample(size=(128, 128), mode='bilinear'),
                nn.Conv2d(in_channels=out_channels * 2,
                          out_channels=out_channels, kernel_size=1, stride=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=False),
                # 128 x 128 x 64
                nn.Upsample(size=(256, 256), mode='bilinear'),
                nn.Conv2d(in_channels=out_channels,
                          out_channels=out_channels, kernel_size=1, stride=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=False),
                # 256 x 256 x 64
                nn.Upsample(size=(self.image_size, self.image_size),
                            mode='bilinear'),
                # 512 * 512 * 64
                nn.Conv2d(in_channels=out_channels, out_channels=2,
                          kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            )

        self.eval()

    def load_pretrain_model(self, pretrain_model=None):
        if pretrain_model is None or pretrain_model == "":
            pretrain_model = _pretrain_model
        self.load_state_dict(torch.load(pretrain_model))

    def forward(self, dImage, gImage, to_warps=None):
        '''
        :param dImage: degraded image
        :param gImage: guidance image
        :return: wrap ref image, flow field
        '''
        ori_height, ori_width = dImage.size()[2], dImage.size()[3]
        # assert ori_height == ori_width

        if ori_height != self.image_size:
            dImage_interp = F.interpolate(dImage, size=(
                self.image_size, self.image_size), mode='bilinear')
            gImage_interp = F.interpolate(gImage, size=(
                self.image_size, self.image_size), mode='bilinear')
            input_layer = torch.cat([dImage_interp, gImage_interp], dim=1)
        else:
            input_layer = torch.cat(
                [dImage, gImage], dim=1)  # 256 x 256 x 6
        input_layer = self.encoder(input_layer)
        # 1 x 1 x 256
        grid = self.decoder(input_layer)

        if ori_height != self.image_size:
            grid = F.interpolate(grid, size=(
                ori_height, ori_width), mode='bilinear')

        # Swap x and y
        # NCHW -> NHWC | grid[:, :, :, 0] -> height | grid[:, :, :, 1] -> width
        grid = grid.permute((0, 2, 3, 1))
        grid_no_grad = grid.clone()

        # grid[:, :, :, 0] -> width | grid[:, :, :, 1] -> height
        grid_no_grad[:, :, :, 0], grid_no_grad[:, :,
                                  :, 1] = grid[:, :, :, 1], grid[:, :, :, 0]

        wImage = F.grid_sample(gImage, grid_no_grad)
        grid = grid.permute((0, 3, 1, 2))

        if self.use_gray:
            wImage = wImage[:, 0, :, :] * 0.299 + wImage[:,
                                                  1, :, :] * 0.587 + wImage[:, 2, :, :] * 0.114
            wImage = wImage.view(wImage.size()[0], 1, ori_height, ori_width)

        if to_warps is None:
            return grid, wImage
        else:
            for i in range(len(to_warps)):
                to_warps[i] = F.grid_sample(to_warps[i], grid_no_grad)

            return grid, wImage, to_warps


