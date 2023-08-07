"""
GAN相关的loss, 其中有些是辅助loss, 有些仅作用于DNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

import numpy as np

from .base import GANLossBase
from utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class GANLoss(GANLossBase):

    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, fake, real=None, flag='G'):
        """
        fake: NCHW
        real: NCHW
        flag: G / D
        """
        if flag == 'G':
            global_real_label = torch.FloatTensor(fake.size()).fill_(1.).to(fake.device)
            loss = self.loss(fake, global_real_label)
        elif flag == 'D':
            global_real_label = torch.FloatTensor(real.size()).fill_(1.).to(fake.device)
            global_fake_label = torch.FloatTensor(fake.size()).fill_(0.).to(fake.device)
            loss = self.loss(real, global_real_label) + self.loss(fake, global_fake_label)
        else:
            raise ValueError('flag should be G or D, or %s' % flag)
        
        return loss



@LOSS_REGISTRY.register()
class LSGANLoss(GANLossBase):

    def __init__(self):
        super(LSGANLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, fake, real=None, flag='G'):
        """
        fake: NCHW
        real: NCHW
        flag: G / D
        """
        if flag == 'G':
            global_real_label = torch.FloatTensor(fake.size()).fill_(1.).to(fake.device)
            loss = self.loss(fake, global_real_label)
        elif flag == 'D':
            global_real_label = torch.FloatTensor(real.size()).fill_(1.).to(real.device)
            global_fake_label = torch.FloatTensor(fake.size()).fill_(0.).to(fake.device)
            loss = self.loss(real, global_real_label) + self.loss(fake, global_fake_label)
        else:
            raise ValueError('flag should be G or D, or %s' % flag)

        return loss



@LOSS_REGISTRY.register()
class HingeGANLoss(GANLossBase):

    def __init__(self):
        super(HingeGANLoss, self).__init__()
        self.weight_loss = lambda x, weight: torch.mean(x * weight)
        self.loss = lambda x: torch.mean(x)

    def forward(self, fake, real=None, weight=None, flag='G'):
        """
        fake: NCHW
        real: NCHW
        flag: G / D
        """
        if flag == 'G':
            if weight is not None:
                _weight = weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                _weight = _weight.repeat(1, fake.shape[1], fake.shape[2], fake.shape[3])
                loss = self.weight_loss(fake, _weight)
            else:
                loss = self.loss(fake)
        elif flag == 'D':
            loss = self.loss(F.relu(1 - real)) + self.loss(F.relu(1. + fake))
        else:
            raise ValueError('flag should be G or D, or %s' % flag)

        return loss



class CategoricalLoss(nn.Module):
    """https://github.com/kam1107/RealnessGAN
        Realness GAN 
        output a distribution not a single scalar
    """
    def __init__(self, atoms=51, v_max=1.0, v_min=-1.0):
        super(CategoricalLoss, self).__init__()

        self.atoms = atoms
        self.v_max = v_max
        self.v_min = v_min
        supports = torch.linspace(v_min, v_max, atoms).view(1, 1, atoms) # RL: [bs, #action, #quantiles]
        self.delta = (v_max - v_min) / (atoms - 1)

        self.register_buffer('supports', supports)

    def forward(self, anchor, feature, skewness=0.0):
        batch_size = feature.shape[0]
        skew = torch.zeros((batch_size, self.atoms)).to(feature.device).fill_(skewness)

        # experiment to adjust KL divergence between positive/negative anchors
        Tz = skew + self.supports.view(1, -1).to(feature.device) * torch.ones((batch_size, 1)).to(torch.float).view(-1, 1).to(feature.device)
        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / self.delta
        l = b.floor().to(torch.int64)
        u = b.ceil().to(torch.int64)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms - 1)) * (l == u)] += 1
        offset = torch.linspace(0, (batch_size - 1) * self.atoms, batch_size).to(torch.int64).unsqueeze(dim=1).expand(batch_size, self.atoms).to(feature.device)
        skewed_anchor = torch.zeros(batch_size, self.atoms).to(feature.device)
        skewed_anchor.view(-1).index_add_(0, (l + offset).view(-1), (anchor * (u.float() - b)).view(-1))  
        skewed_anchor.view(-1).index_add_(0, (u + offset).view(-1), (anchor * (b - l.float())).view(-1))  
        loss = -(skewed_anchor * (feature + 1e-16).log()).sum(-1).mean()
        return loss

@LOSS_REGISTRY.register()
class RealnessGANLoss(GANLossBase):
    """https://github.com/kam1107/RealnessGAN
        Realness GAN 
        output a distribution, not a single scalar
    """

    def __init__(self, atoms=51) -> None:
        super(RealnessGANLoss, self).__init__()
        self.atoms = atoms
        self.criterion = CategoricalLoss(atoms=self.atoms, v_max=1.0, v_min=-1.0)
        self.anchor_init(self.atoms)

    def anchor_init(self, atoms=51):
        gauss = np.random.normal(0, 0.1, 1000)
        count, bins = np.histogram(gauss, atoms)
        self.anchor0 = count / sum(count)

        unif = np.random.uniform(-1, 1, 1000)
        count, bins = np.histogram(unif, atoms)
        self.anchor1 = count / sum(count)

    def forward(self, fake, real, flag='G'):
        '''
        fake: NCHW
        real: NCHW
        '''
        b_size = fake.shape[0]

        anchor_real = torch.zeros((b_size, self.atoms), dtype=torch.float).to(fake.device) + \
            torch.tensor(self.anchor1, dtype=torch.float).to(fake.device)
        
        anchor_fake = torch.zeros((b_size, self.atoms), dtype=torch.float).to(fake.device) + \
            torch.tensor(self.anchor1, dtype=torch.float).to(fake.device)
        
        fake = fake.log_softmax(1).exp()
        real = real.log_softmax(1).exp()

        if flag == 'G':
            loss = self.criterion(anchor_real, fake, skewness=1.0) \
                - self.criterion(anchor_fake, fake, skewness=-1.0)
        elif flag == 'D':
            loss = self.criterion(anchor_fake, fake, skewness=-1.0) \
                + self.criterion(anchor_real, real, skewness=1.0)
        else:
            raise ValueError('flag should be G or D, or %s' % flag)

        return loss 



@LOSS_REGISTRY.register()
class RaGANLoss(GANLossBase):

    def __init__(self):
        super(RaGANLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, fake, real, flag='G'):
        """
        fake: NCHW
        real: NCHW
        flag: not used
        """
        y = torch.ones_like(fake)
        loss = self.loss(fake - real, y)
        return loss



@LOSS_REGISTRY.register()
class NonsaturateGANLoss(GANLossBase):

    def __init__(self, freq=8):
        super(NonsaturateGANLoss, self).__init__()
        self.freq = freq


    def _d_logistic_loss(self, real, fake):
        real_loss = F.softplus(-real)
        fake_loss = F.softplus(fake)

        return real_loss.mean() + fake_loss.mean()

    def _g_nonsaturating_loss(self, fake):
        loss = F.softplus(-fake).mean()
        return loss

    def forward(self, fake, real=None, flag='G'):
        """
        fake: NCHW
        real: NCHW
        flag: G / D
        """

        if flag == 'G':
            loss = self._g_nonsaturating_loss(fake) 
        elif flag == 'D':
            loss = self._d_logistic_loss(real, fake)
        else:
            raise ValueError('flag should be G or D, or %s' % flag)
        
        return loss



@LOSS_REGISTRY.register()
class R1Loss(GANLossBase):

    def __init__(self, freq=4, r1=10) -> None:
        super(R1Loss, self).__init__()
        self.freq = freq
        self.r1 = r1

    def _d_r1_loss(self, real_pred, real_img):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum().mean()

        return grad_penalty

    def forward(self, real_pred, real_img):
        """
        Args:
            real_pred: pred of d_net
            real_img: real_img data, need require gradient
        Returns:

        """
        r1_loss = self._d_r1_loss(real_pred, real_img)
        return self.r1 / 2 * r1_loss * self.freq + 0 * torch.mean(real_pred[0]), r1_loss



@LOSS_REGISTRY.register()
class RelativeHingeGANLoss(GANLossBase):

    def __init__(self):
        super(RelativeHingeGANLoss, self).__init__()

    def dis_loss(self, real, fake):
        r_f_diff = real - torch.mean(fake)
        f_r_diff = fake - torch.mean(real)

        loss = torch.mean(F.relu(1 - r_f_diff)) + torch.mean(F.relu(1 + f_r_diff))

        return loss

    def gen_loss(self, real, fake):
        r_f_diff = real - torch.mean(fake)
        f_r_diff = fake - torch.mean(real)

        loss = torch.mean(F.relu(1 + r_f_diff)) + torch.mean(F.relu(1 - f_r_diff))

        return loss

    def forward(self, fake, real, flag='G'):
        """
        fake: NCHW
        real: NCHW
        flag: G / D
        """
        if flag == 'G':
            loss = self.gen_loss(real, fake)
        elif flag == 'D':
            loss = self.dis_loss(real, fake)
        else:
            raise ValueError('flag should be G or D, or %s' % flag)

        return loss
