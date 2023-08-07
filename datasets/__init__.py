import importlib
from copy import deepcopy
from os import path as osp
import os

from .collate import collate
from utils.registry import DATASET_REGISTRY



__all__ = ['build_dataset', 'collate']

# automatically scan and import dataset modules for registry
# scan all the files under the 'datasets' folder and collect files ending with
# 'dataset.py'
dataset_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in os.listdir(dataset_folder) if v.endswith('dataset.py')]
# import all the dataset modules
_dataset_modules = [importlib.import_module(f'datasets.{file_name}') for file_name in dataset_filenames]


def build_dataset(opt):
    """Build dataset from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): dataset type.
    """
    opt = deepcopy(opt)
    dataset = DATASET_REGISTRY.get(opt['type'])(opt)
    print(f'dataset [{dataset.__class__.__name__}] is created.')
    return dataset
