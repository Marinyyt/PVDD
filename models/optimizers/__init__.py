import importlib
from os import path as osp
import os
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim


from utils.registry import OPTIM_REGISTRY



# register pytorch internal optimizer
optimizers = ['Adadelta', 'Adagrad', 'Adam', 'SparseAdam', 'Adamax',
              'ASGD', 'SGD', 'Rprop', 'RMSprop', 'LBFGS']
for o in optimizers:
    OPTIM_REGISTRY.register(getattr(optim, o))



# automatically scan and import optimizer modules for registry
# scan all the files under the 'optimizers' folder and collect files ending with
# 'optim.py'
optim_folder = osp.dirname(osp.abspath(__file__))
optim_filenames = [osp.splitext(osp.basename(v))[0] for v in os.listdir(optim_folder) if v.endswith('optim.py')]
# import all the optimizer modules
_optim_modules = [importlib.import_module(f'models.optimizers.{file_name}') for file_name in optim_filenames]


def build_optimizer(params, opt):
    """Build optimizer from config.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Optimizer type.
    """
    opt = deepcopy(opt)

    if hasattr(opt, 'args'):
        optim = OPTIM_REGISTRY.get(opt['type'])(params=params, **opt.args)
    else:
        optim = OPTIM_REGISTRY.get(opt['type'])()
    
    print(f'Optimizer [{optim.__class__.__name__}] is created.')
    return optim
