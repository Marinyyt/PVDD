import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import sys
from . import cx_distance as CX



def postprocessing(x):
    return x / x.norm(dim=1, p=2, keepdim=True)


def identity_loss(x, y):
    x_vector = postprocessing(x)
    y_vector = postprocessing(y)
    return (1 - (x_vector * y_vector).sum(dim=1)).mean()


# New mask pooling function, random pooling features according to binary mask locations
def mask_pooling(features, mask_parts, s=48, alpha=1.):
    """
        The Binary mask will yiled many indices, non-zero parts are where
            we are interested into

        :features: the [x, y] (rec, the other)
        :mask_parts: the parts including eyes, eye brow and mouse
        :alpha: mask weight
    """
    assert type(features) == list, 'features must be a list of tensors'
    x_list, y_list = [], []
    s_size = list(features[0].shape[2:])
    new_mask_parts = F.interpolate(mask_parts, s_size, mode="nearest")

    cand_len = 1024 ** 2
    # cand_len = 512 ** 2
    for n, _ in enumerate(new_mask_parts):
        H, W = s_size
        mask_i_ = new_mask_parts[n].view(H * W)
        cand_inds = mask_i_.nonzero().flatten()
        cand_len_ = len(cand_inds)
        if cand_len_ < cand_len:
            mask_i = mask_i_

    for i, (x, y) in enumerate(zip(*features)):
        C, H, W = x.size()
        # mask_i = new_mask_parts[i].view(H * W)
        cand_inds = mask_i.nonzero().flatten()
        cand_inds_o = (1 - mask_i).nonzero().flatten()
        cand_len = len(cand_inds)
        cand_len_o = len(cand_inds_o)

        if cand_len < s * s:
            s_i = math.floor(math.sqrt(cand_len))
        else:
            s_i = s

        index = cand_inds[torch.randperm(cand_len)[:round(s_i * s_i * alpha)]]
        index_o = cand_inds_o[torch.randperm(cand_len_o)[:round(s_i * s_i * (1 - alpha))]]
        select_indices = torch.cat((index, index_o))

        x, y = x.view(C, -1), y.view(C, -1)
        x_sliced = x[:, select_indices].contiguous().view(1, C, s_i, s_i)
        y_sliced = y[:, select_indices].contiguous().view(1, C, s_i, s_i)
        x_list.append(x_sliced)
        y_list.append(y_sliced)

    return [torch.cat(x_list, dim=0), torch.cat(y_list, dim=0)]


# random sampling operation for large size features
# random sample need same indices for target and synthesis
def rand_sampling(features, s=64, d_indices=None):
    """
    :param features: features for random sampling operation NCHW
    :param s: sampled feature size
    :param d_indices: sampled index
    :return:
    """
    N, C, H, W = features.size()
    features = features.view(N, C, -1)
    # features = features.data.unfold(0, C, C).unfold(1, s, s).unfold(2, s, s)
    all_indices = torch.randperm(H * W)
    select_indices = torch.arange(0, s ** 2, dtype=torch.long)
    d_indices = torch.gather(all_indices, dim=0, index=select_indices) if d_indices is None else d_indices
    # features = torch.gather(features, dim=2, index=d_indices)
    features = features[:, :, d_indices]
    re = features.contiguous().view(N, C, s, s)
    return re, d_indices



# random pool samples from features
def rand_pooling(features, s=64):
    assert type(features) == list, 'features must be a list of tensors'
    s_features = []
    sample_feature_0, d_indices = rand_sampling(features[0], s)
    s_features.append(sample_feature_0)
    for i in range(1, len(features)):
        sample_feature, _ = rand_sampling(features[i], s=s, d_indices=d_indices)
        s_features.append(sample_feature)

    return s_features



def uniform_pooling(features):
    device = features.device
    kernel_dict = {512: [8, 8, 4], 256: [4, 4, 2], 128: [8, 4, 4]}
    # get feature shape
    _, c, h, w = features.size()
    # generate pooling operator
    k, s, p = kernel_dict[h]
    conv_pool = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=k, stride=s, padding=p, groups=c, bias=False)
    # set not require gradient
    conv_pool.weight.requires_grad = False
    # random sample kernel
    local_filter = torch.zeros((k, k))
    local_index = random.randint(0, k - 1), random.randint(0, k - 1)
    local_filter[local_index] = 1
    conv_pool.weight.data = local_filter.unsqueeze(dim=0).expand(c, -1, -1, -1)
    conv_pool = conv_pool.to(device)
    return conv_pool(features)




# uniformed local pooling
def local_pooling(features):
    device = features.device
    kernel_dict = {512: [8, 8, 4], 256: [4, 4, 2], 128: [3, 2, 1]}
    # get feature shape
    _, c, h, w = features.size()
    # generate pooling operator
    k, s, p = kernel_dict[h]
    conv_pool = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=k, stride=s, padding=p, groups=c, bias=False)
    # set not require gradient
    conv_pool.weight.requires_grad = False
    # random sample kernel
    local_filter = torch.zeros((k, k))
    local_index = random.randint(0, k - 1), random.randint(0, k - 1)
    local_filter[local_index] = 1
    conv_pool.weight.data = local_filter.unsqueeze(dim=0).expand(c, -1, -1, -1)
    conv_pool = conv_pool.to(device)
    return conv_pool(features)



def CX_loss_helper(vgg_A, vgg_B, slide_size_1d=48, w_spatial=0.1, mask_parts=None, h=0.1):
    """

    :vgg_A: first vgg output features
    :vgg_B: second vgg output features
    :CX_config: dictionary of CX loss configuration
    """

    N, fC, fH, fW = vgg_A.shape
    if fH * fW <= slide_size_1d ** 2:
        pass
    else:
        if mask_parts is not None:
            vgg_A, vgg_B = mask_pooling([vgg_A, vgg_B], mask_parts, s=slide_size_1d, alpha=1.)
        else:
            vgg_A, vgg_B = rand_pooling([vgg_A, vgg_B], s=slide_size_1d)
    cx_loss = CX.CX_loss(vgg_A, vgg_B, h, w_sp=w_spatial)
    return cx_loss



def feature_normalize(feature_in):
    feature_in_norm = torch.norm(feature_in, 2, 1, keepdim=True) + sys.float_info.epsilon
    feature_in_norm = torch.div(feature_in, feature_in_norm)
    return feature_in_norm




class ContextualLoss_forward(nn.Module):
    '''
        input is Al, Bl, channel = 1, range ~ [0, 255]
    '''

    def __init__(self, PONO, avg_pooling=False):
        super(ContextualLoss_forward, self).__init__()
        self.PONO = PONO
        self.avg_pooling = avg_pooling

    def forward(self, X_features, Y_features, h=0.1, feature_centering=True):
        '''
        X_features&Y_features are are feature vectors or feature 2d array
        h: bandwidth
        return the per-sample loss
        '''
        if X_features.size(2) > 64:
            if not self.avg_pooling:
                X_features, Y_features = rand_pooling([X_features, Y_features], s=64)
                # replace rand pooling as not aligned local pooling
                # X_features, Y_features = local_pooling(X_features), local_pooling(Y_features)
            else:
                X_features, Y_features = F.adaptive_avg_pool2d(X_features, 64), F.adaptive_avg_pool2d(Y_features, 64)

        batch_size = X_features.shape[0]
        feature_depth = X_features.shape[1]
        feature_size = X_features.shape[2]

        # to normalized feature vectors
        if feature_centering:
            if self.PONO:
                X_features = X_features - Y_features.mean(dim=1).unsqueeze(dim=1)
                Y_features = Y_features - Y_features.mean(dim=1).unsqueeze(dim=1)
            else:
                X_features = X_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(
                    dim=-1).unsqueeze(dim=-1)
                Y_features = Y_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(
                    dim=-1).unsqueeze(dim=-1)
        X_features = feature_normalize(X_features).view(batch_size, feature_depth,
                                                        -1)  # batch_size * feature_depth * feature_size * feature_size
        Y_features = feature_normalize(Y_features).view(batch_size, feature_depth,
                                                        -1)  # batch_size * feature_depth * feature_size * feature_size

        # X_features = F.unfold(
        #     X_features, kernel_size=self.opt.match_kernel, stride=1, padding=int(self.opt.match_kernel // 2))  # batch_size * feature_depth_new * feature_size^2
        # Y_features = F.unfold(
        #     Y_features, kernel_size=self.opt.match_kernel, stride=1, padding=int(self.opt.match_kernel // 2))  # batch_size * feature_depth_new * feature_size^2

        # conine distance = 1 - similarity
        X_features_permute = X_features.permute(0, 2, 1)  # batch_size * feature_size^2 * feature_depth
        d = 1 - torch.matmul(X_features_permute, Y_features)  # batch_size * feature_size^2 * feature_size^2

        # normalized distance: dij_bar
        # d_norm = d
        d_norm = d / (torch.min(d, dim=-1, keepdim=True)[0] + 1e-3)  # batch_size * feature_size^2 * feature_size^2

        # pairwise affinity
        w = torch.exp((1 - d_norm) / h)
        A_ij = w / torch.sum(w, dim=-1, keepdim=True)

        # contextual loss per sample
        CX = torch.mean(torch.max(A_ij, dim=-1)[0], dim=1)
        loss = -torch.log(CX)

        # contextual loss per batch
        loss = torch.mean(loss)
        return loss



# contextual loss for multi GPUs
class contextual_loss(nn.Module):
    def __init__(self,
                 avg_pooling=False):
        super(contextual_loss, self).__init__()
        self.avg_pooling = avg_pooling

    def forward(self, x, y, h=0.5, mask_parts=None):
        """Computes contextual loss between x and y.

        Args:
          x: features of shape (N, C, H, W).
          y: features of shape (N, C, H, W).
          h: h>0 is a band-width parameter; semantic style transfer(h=0.1for c, h=0.2for s)

        Returns:
          cx_loss = contextual loss between x and y (Eq (1) in the paper)
        """
        if x.size(2) > 64:
            if mask_parts is not None:  # mask parts constraint
                x, y = mask_pooling([x, y], mask_parts, s=64)
            else:
                if not self.avg_pooling:
                    # x, y = rand_pooling([x, y], s=64)
                    # replace rand pooling as not aligned local pooling
                    x, y = local_pooling(x), local_pooling(y)
                else:
                    x, y = F.adaptive_avg_pool2d(x, 64), F.adaptive_avg_pool2d(y, 64)

        assert x.size() == y.size()
        N, C, H, W = x.size()  # e.g., 10 x 512 x 14 x 14. In this case, the number of points is 196 (14x14).

        y_mu = y.mean(3).mean(2).mean(0).reshape(1, -1, 1, 1)

        x_centered = x - y_mu
        y_centered = y - y_mu
        x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
        y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1, keepdim=True)

        # The equation at the bottom of page 6 in the paper
        # Vectorized computation of cosine similarity for each pair of x_i and y_j
        x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
        y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)
        cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H*W, H*W)

        d = 1 - cosine_sim  # (N, H*W, H*W)  d[n, i, j] means d_ij for n-th data
        d_min, _ = torch.min(d, dim=2, keepdim=True)  # (N, H*W, 1)

        # Eq (2)
        d_tilde = d / (d_min + 1e-5)

        # Eq(3)
        w = torch.exp((1 - d_tilde) / h)

        # Eq(4)
        cx_ij = w / torch.sum(w, dim=2, keepdim=True)  # (N, H*W, H*W)

        # Eq (1)
        cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)  # (N, )
        cx_loss = torch.mean(-torch.log(cx + 1e-5))

        return cx_loss