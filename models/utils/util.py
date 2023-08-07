import torch
import torch.nn as nn
import torch.nn.functional as F
from random import choices

def normalize_augment(datain1, datain2):
        def transform(sample):
                do_nothing = lambda x: x
                do_nothing.__name__ = 'do_nothing'
                flipud = lambda x: torch.flip(x, dims=[2])
                flipud.__name__ = 'flipup'
                rot90 = lambda x: torch.rot90(x, k=1, dims=[2, 3])
                rot90.__name__ = 'rot90'
                rot90_flipud = lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2])
                rot90_flipud.__name__ = 'rot90_flipud'
                rot180 = lambda x: torch.rot90(x, k=2, dims=[2, 3])
                rot180.__name__ = 'rot180'
                rot180_flipud = lambda x: torch.flip(torch.rot90(x, k=2, dims=[2, 3]), dims=[2])
                rot180_flipud.__name__ = 'rot180_flipud'
                rot270 = lambda x: torch.rot90(x, k=3, dims=[2, 3])
                rot270.__name__ = 'rot270'
                rot270_flipud = lambda x: torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2])
                rot270_flipud.__name__ = 'rot270_flipud'
                add_csnt = lambda x: x + torch.normal(mean=torch.zeros(x.size()[0], 1, 1, 1), std=(5/255.)).expand_as(x).to(x.device)
                add_csnt.__name__ = 'add_csnt'
                aug_list = [do_nothing, flipud, rot90, rot90_flipud, rot180, rot180_flipud, rot270, rot270_flipud]
                w_aug = [32, 12, 12, 12, 12, 12, 12, 12]  # one fourth chances to do_nothing
                transf = choices(aug_list, w_aug)
                # transform all images in array
                return transf[0](sample)
        b, n, c, h, w = datain1.size()
        channel = datain1.shape[2]
        img_train = torch.cat([datain1, datain2], dim=2)
        # convert to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 255.]
        img_train = img_train.view(b, n*c*2, h, w)
        #augment
        img_train = transform(img_train)
        img_train = img_train.view(b, n, c*2, h, w)

        input_train = img_train[:, :, 0:channel, :, :]
        input_train = input_train.view(b, n, c, h, w)
        # extract ground truth (central frame)
        # gt_train = img_train[:, 3*ctrl_fr_idx:3*ctrl_fr_idx+3, :, :]
        gt_train = img_train[:, :, channel:channel * 2, :, :]
        gt_train = gt_train.view(b, n, c, h, w)
        # gt_train = gt_train[:, 3 * ctrl_fr_idx:3 * ctrl_fr_idx + 3, :, :]
        return input_train, gt_train

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)




def is_module_instance(module, classtype):
    """
    Judge network or loss type
    """
    assert isinstance(module, nn.Module)

    return isinstance(module, classtype) or hasattr(module, 'module') and isinstance(module.module, classtype)


def get_activation(activation_name):
    """ get activation from string to obj
    """
    active_name = activation_name.lower()

    if active_name == "relu":
        return nn.ReLU(True)
    elif active_name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif active_name == 'gelu':
        return nn.GELU()
    else:
        raise NotImplementedError("Unidentified activation name {}.".format(active_name))
