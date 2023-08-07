import importlib
from copy import deepcopy
from os import path as osp
import os

from utils.registry import MODEL_REGISTRY

__all__ = ['build_model']

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with
# 'model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in os.listdir(model_folder) if v.endswith('model.py')]
# import all the model modules
_model_modules = [importlib.import_module(f'models.restorers.{file_name}') for file_name in model_filenames]


def build_model(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt['type'])(opt)
    print(f'Model [{model.__class__.__name__}] is created.')
    return model
