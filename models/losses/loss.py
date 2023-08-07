import torch
import torch.nn as nn
  


from functools import partial
import numpy as np

from .base import LossBase, VGGLossBase
from .utils.utils import conv_4to3, RGB2gray, get_wav
from .utils.cx_modules import *
from .utils.radialProfile import azimuthalAverage
from .utils.pytorch_spl_loss import GPLoss, CPLoss
from utils.registry import LOSS_REGISTRY



__all__ = ['RegLoss', 'CXLoss', 'PerceptualLoss',
        'FeatMatchLoss', 'PatchGramLoss', 'LandmarkLoss', 'L2Loss', 'SpectralLoss', 'SPLLoss',
        'HistogramLoss', 'TVRLoss', 'DefectLoss', 'SSIMLoss', 'L1_CharbonnierLoss', 'EMVDLoss']

@LOSS_REGISTRY.register()
class RegLoss(LossBase):
    def __init__(self) -> None:
        super(RegLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        b, c, h, w = pred.data.size()
        target = target.unsqueeze(-1).unsqueeze(-1)
        target = target.expand((b, c, h, w))
        return self.loss(pred, target)


@LOSS_REGISTRY.register()
class CXLoss(VGGLossBase):

    """
    The modified version of Contextual Loss
    https://arxiv.org/abs/1803.02077
    """

    def __init__(self, 
                 cx_loss_type='cx',
                 norm_input=False,
                 weights_cx_average=False,
                 gt_cx_level=[3, 4],
                 ref_cx_level=[1, 2, 3, 4],
                 avg_pooling=False,
                 cx_h=[0.1, 0.2],
                 content_bias=1.0,
                 scale_size=512,
                 use_texture=False) -> None:

        super(CXLoss, self).__init__()
        
        if weights_cx_average:
            self.weights_cx = [0.1, ] * 5
        else:
            self.weights_cx = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2]
        
        self.cx_loss_type = cx_loss_type
        self.norm_input = norm_input
        self.weights_cx_average = weights_cx_average
        self.gt_cx_level = gt_cx_level
        self.ref_cx_level = ref_cx_level
        self.avg_pooling = avg_pooling
        self.cx_h = cx_h
        self.content_bias = content_bias

        self.scale_size = scale_size
        self.use_texture = use_texture
    
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2.]

        if cx_loss_type == 'cx':
            self.cxloss = contextual_loss(avg_pooling=self.avg_pooling)
        elif cx_loss_type == 'cx_of':
            self.cxloss = ContextualLoss_forward(PONO=True, avg_pooling=self.avg_pooling)
        # contextual bi loss
        elif cx_loss_type == 'cs':
            self.cxloss = partial(CX_loss_helper, slide_size_1d=64, w_spatial=0.1)
        else:
            raise NotImplementedError("loss {} is not supported!".format(cx_loss_type))
        # texture loss
        if use_texture:
            self.HH = get_wav(in_channels=3)
            self.texture_loss = TextureLoss()
    
    def _compute_cx(self, x_vgg, y_vgg, layers: list, h=0.1, w_spatial=0.1):
        # For now w_spatial is obsolete
        loss_cx = 0.
        for n in layers:
            loss_cx += self.weights_cx[n] * self.cxloss(x_vgg[n], y_vgg[n].detach(), h=h)
        return loss_cx
    
    def _compute_texture(self, x_origin, ref_vgg):
        if x_origin.size(1) == 1:
                x_origin = x_origin.repeat(1, 3, 1, 1)
        x_hh = self.HH(F.interpolate(x_origin, size=1024, mode='area'))
        x_hh = self.vgg(x_hh)
        loss_texture = 0.
        for i, (x_, ref_) in enumerate(zip(x_hh, ref_vgg)):
            loss_texture += self.weights[i] * self.texture_loss(x_, ref_)
        return loss_texture

    
    def forward(self, x, y, ref=None, rec=None):
        x_origin = x.clone()
        # when batch = 1 try 1024 resolution
        if x.size(2) > self.scale_size:
            x = F.interpolate(x, size=self.scale_size, mode='area')
            y = F.interpolate(y, size=self.scale_size, mode='area')
            if ref is not None:
                ref = F.interpolate(ref, size=self.scale_size, mode='area')
            if rec is not None:
                rec = F.interpolate(rec, size=self.scale_size, mode='area')
        # input normalization
        if self.norm_input:
            x = self._normalization((x + 1) / 2.)
            y = self._normalization((y + 1) / 2.)
            if ref is not None:
                ref = self._normalization((ref + 1) / 2.)
            if rec is not None:
                rec = self._normalization((rec + 1) / 2.)

        # C channel is 1, then repeat
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.size()[1] == 1:
            y = y.repeat(1, 3, 1, 1)

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        if ref is not None and ref.size()[1] == 1:
            ref = ref.repeat(1, 3, 1, 1)
        if rec is not None and rec.size()[1] == 1:
            rec = rec.repeat(1, 3, 1, 1)

        ref_vgg = self.vgg(ref) if ref is not None else None
        # using gt features while decompose output
        rec_vgg = self.vgg(rec) if rec is not None else None

        # [0, 1, 2, 3, 4]
        # conv1_2, conv2_2, conv3_2, conv4_2, conv5_1
        
        # For now w_spatial is obsolete
        loss_cx = self._compute_cx(x_vgg, y_vgg, self.gt_cx_level, h=self.cx_h[0],
                                    w_spatial=0.1) * self.content_bias
        
        if ref_vgg is not None:
            loss_cx += self._compute_cx(x_vgg, ref_vgg, self.ref_cx_level,
                                        h=self.cx_h[1], w_spatial=0.)
        if self.use_texture:
            loss_cx += self._compute_texture(x_origin, ref_vgg)
        
        return loss_cx      


@LOSS_REGISTRY.register()
class PerceptualLoss(VGGLossBase):
    def __init__(self, norm_input=True, scale_size=256):
        super(PerceptualLoss, self).__init__()

        self.norm_input = norm_input
        self.scale_size = scale_size

        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2.]

    def _compute_vgg(self, x_vgg, y_vgg):
        loss_vgg = 0.
        for i in range(len(x_vgg)):
            loss_vgg += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss_vgg

    def forward(self, x, gt):
        # when batch = 1 try 1024 resolution
        
        if x.size(2) > self.scale_size:
            x = F.interpolate(x, size=self.scale_size, mode='area')
            gt = F.interpolate(gt, size=self.scale_size, mode='area')

        # input normalization
        if self.norm_input:
            x = self._normalization(x)
            gt = self._normalization(gt)

        # C channel is 1, then repeat
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if gt.size()[1] == 1:
            gt = gt.repeat(1, 3, 1, 1)

        x_vgg, gt_vgg = self.vgg(x), self.vgg(gt.detach())

        loss = self._compute_vgg(x_vgg, gt_vgg)

        return loss


@LOSS_REGISTRY.register()
class PatchGramLoss(LossBase):

    def __init__(self, patch_size=3) -> None:
        super(PatchGramLoss, self).__init__()
        self.patch_size = patch_size
        self.loss = nn.MSELoss()

    def forward(self, infer, target):
        assert infer.size() == target.size()
        _, C, H, W = infer.size()

        patch_H, patch_W = min(H, self.patch_size), min(W, self.patch_size)

        def extract_patch(img):
            patches_fold_H = img.unfold(2, patch_H, patch_H)
            if (img.size(2) % patch_H != 0):
                patches_fold_H = torch.cat((patches_fold_H, img[:, :, -patch_H:, ].permute(0, 1, 3, 2).unsqueeze(2)),
                                           dim=2)
            patches_fold_HW = patches_fold_H.unfold(3, patch_W, patch_W)
            if (img.size(3) % patch_W != 0):
                patches_fold_HW = torch.cat(
                    (patches_fold_HW, patches_fold_H[:, :, :, -patch_W:, :].permute(0, 1, 2, 4, 3).unsqueeze(3)), dim=3)
            patches = patches_fold_HW.permute(0, 2, 3, 1, 4, 5).reshape(-1, img.size(1), patch_H, patch_W)
            return patches
        
        patchesOfinfer = extract_patch(infer)
        patchesOftarget = extract_patch(target)

        patchGramOfinfer = torch.einsum("bcji,bkji->bck", [patchesOfinfer, patchesOfinfer])
        patchGramOftarget = torch.einsum("bcji,bkji->bck", [patchesOftarget, patchesOftarget])
        return self.loss(patchGramOfinfer, patchGramOftarget)



@LOSS_REGISTRY.register()
class L2Loss(LossBase):

    def __init__(self) -> None:
        super(L2Loss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        return self.loss(pred, target)


@LOSS_REGISTRY.register()
class LandmarkLoss(LossBase):

    def __init__(self) -> None:
        super(LandmarkLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, x, y):
        return self.loss(x, y)


@LOSS_REGISTRY.register()
class SpectralLoss(LossBase):

    def __init__(self) -> None:
        super(SpectralLoss, self).__init__()
        self.lambda_freq = 1e-5
        self.loss = nn.BCELoss()

    def _get_psd(self, img):
        # N = 360 # 1024
        N = 315
        epsilon = 1e-8
        psd1D_img = np.zeros([img.shape[0], N])

        for t in range(img.shape[0]):
            gen_imgs = img.permute(0, 2, 3, 1)
            img_numpy = gen_imgs[t, :, :, :].cpu().detach().numpy()
            # yuv training
            if img.shape[1] > 1:
                img_gray = RGB2gray(img_numpy)
            else:
                img_gray = img_numpy[:, :, 0]
            fft = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(fft)
            fshift += epsilon
            magnitude_spectrum = 20 * np.log(np.abs(fshift))
            psd1D = azimuthalAverage(magnitude_spectrum)
            psd1D = (psd1D - np.min(psd1D)) / (np.max(psd1D) - np.min(psd1D))
            psd1D_img[t, :] = psd1D
        psd1d_img = torch.from_numpy(psd1D_img).float()

        return psd1d_img.requires_grad_()

    def forward(self, x, y):
        x, y = x[-1], y[-1]

        if x.size(2) > 512:
            x = F.interpolate(x, scale_factor=0.5)
            y = F.interpolate(y, scale_factor=0.5)
        
        psd_pred = self._get_psd(x).to(x.device)
        psd_target = self._get_psd(y).to(y.device)

        loss = self.loss(psd_pred, psd_target.detach())

        return loss



@LOSS_REGISTRY.register()
class TVRLoss(LossBase):

    def __init__(self, weight_tv, weight_reg, lp_reg) -> None:
        super(TVRLoss, self).__init__()
        self.weight_tv = weight_tv
        self.weight_reg = weight_reg
        self.lp_reg = lp_reg

    def _compute_tv_smooth(self, mat):
        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))
    
    def forward(self, x, mask=None):
#        if mask is not None:
#            x = x * mask
        tvr_loss = self.weight_tv * self._compute_tv_smooth(x)
        if self.lp_reg:
            tvr_loss += self.weight_reg * torch.mean(x)
        return tvr_loss



@LOSS_REGISTRY.register()
class SPLLoss(LossBase):

    def __init__(self) -> None:
        super(SPLLoss, self).__init__()
        self.GPL = GPLoss()
        self.CPL = CPLoss(rgb=False, yuv=True)

    def forward(self, pred, target):
        spl_loss = self.GPL(pred, target) + self.CPL(pred, target)

        return spl_loss



@LOSS_REGISTRY.register()
class TextureLoss(LossBase):

    def __init__(self) -> None:
        super(TextureLoss, self).__init__()
        self.loss = nn.MSELoss()
    
    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def forward(self, pred, target):
        loss = self.loss(self.gram_matrix(pred), self.gram_matrix(target))
        return loss


@LOSS_REGISTRY.register()
class DefectLoss(LossBase):

    def __init__(self) -> None:
        super(DefectLoss, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, input, target):
        return self.loss(input, target)



@LOSS_REGISTRY.register()
class HistogramLoss(LossBase):

    def __init__(self, n_bins=254):
        super(HistogramLoss, self).__init__()
        self.n_bins = n_bins
        self.criterion = torch.nn.MSELoss()

    def _get_cdf(self, hist):
        cdf_list = []
        for h in hist:
            cdf = torch.cumsum(h, dim=0)
            cdf_list.append(cdf/cdf[-1])
        return torch.stack(cdf_list, dim=0)

    def _histogram_match(self, x, y):
        """
        x: input
        y: target
        return: matched x
        """
        x_hist = self._get_histogram(x)
        y_hist = self._get_histogram(y)

        x_cdf = self._get_cdf(x_hist)
        y_cdf = self._get_cdf(y_hist)
        # match

        x_norm = x.view(x.size(1), -1)
        x_target = torch.zeros_like(x).view(x.size(1), -1)
        for i in range(x.size(1)):
            x_index = (x_norm[i] - x_norm[i].min()) / (x_norm[i].max() - x_norm[i].min()) * self.n_bins
            #
            x_matching = np.interp(x_cdf[i].data.numpy(), y_cdf[i].data.numpy(), np.arange(self.n_bins+1))
            x_matching = torch.from_numpy(x_matching)
            x_target[i, :] = x_matching[x_index.long()]
        x_target = x_target.clamp(0, 255).view(x.shape)
        return (x_target/255)*2 - 1

    def _get_histogram(self, x):
        assert len(x.size()) == 4
        c = x.size(1)
        h = torch.zeros(c, self.n_bins+1)
        for idx in range(c):
            # normalize
            norm = (x[:, idx, ...] - x[:, idx, ...].min())/(x[:, idx, ...].max() - x[:, idx, ...].min()) * self.n_bins
            bins = torch.bincount(norm.view(-1).int())
            h[idx] = bins
        return h

    def forward(self, x, y):
        """
        :param x:  D predict fake
        :param y:  D predict real
        :return:
        """
        x_target = self._histogram_match(x, y)
        loss = self.criterion(x, x_target)
        return loss



@LOSS_REGISTRY.register()
class FeatMatchLoss(LossBase):

    def __init__(self, n_layers=6, num_D=1, high_order=False, use_hist=False):
        super(FeatMatchLoss, self).__init__()
        self.high_order = high_order
        self.use_hist = use_hist
        self.criterionfeat = torch.nn.L1Loss()
        if self.high_order:
            self.criterion_ho = TextureLoss()
        if self.use_hist:
            self.criterion_hist = HistogramLoss()
        self.feat_weights = 4.0 / (n_layers + 1)
        self.D_weights = 1.0 / num_D
      
    def forward(self, x, y):
        """
        :param x:  D predict fake
        :param y:  D predict real
        :return:
        """
        feat_match_loss = 0.
        for i, (x_feat, y_feat) in enumerate(zip(x, y)):
            feat_match_loss += self.feat_weights * self.D_weights * self.criterionfeat(x_feat, y_feat.detach())
            if self.high_order:
                feat_match_loss += self.feat_weights * self.criterion_ho(x_feat, y_feat)
            if self.use_hist:
                feat_match_loss += self.feat_weights * self.criterion_hist(x_feat, y_feat)
        return feat_match_loss
    
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()
def create_window(window_size, channel = 1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=1.0):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

@LOSS_REGISTRY.register()
class L1_CharbonnierLoss(LossBase):
    """L1 Charbonnierloss."""

    def __init__(self, device = 'cuda'):
        super(L1_CharbonnierLoss, self).__init__()
        self.eps = torch.tensor(1e-6, dtype = torch.float16, device = device)

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

@LOSS_REGISTRY.register()
class SSIMLoss(LossBase):
    def __init__(self, window_size=11, size_average=True, val_range=1.0):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return (1 - ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)) / 2



@LOSS_REGISTRY.register()
class EMVDLoss(LossBase):
    def __init__(self):
        super(EMVDLoss, self).__init__()

    def forward(self, model, device ='cuda'):
        loss_orth = torch.tensor(0., dtype=torch.float32).cuda()
        params = {}
        for name, param in model.named_parameters():
            params[name] = param
      
        ft = params['ft.net1.weight'].squeeze()
        fti = torch.cat([params['fti.net1.weight'], params['fti.net2.weight']], dim=0).squeeze()
        weight_squared = torch.matmul(ft, fti)
        diag = torch.eye(weight_squared.shape[1], dtype=torch.float32).cuda()
        if param.dtype == torch.float16:
          diag = diag.half()
        loss = ((weight_squared - diag) ** 2).sum()
        loss_orth += loss
        return loss_orth
