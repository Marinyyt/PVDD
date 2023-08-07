import importlib
from copy import deepcopy
from os import path as osp
import os

from utils.registry import RUNNER_REGISTRY

__all__ = ['build_runner']

# automatically scan and import runner modules for registry
# scan all the files under the 'runners' folder and collect files ending with
# 'runner.py'
runner_folder = osp.dirname(osp.abspath(__file__))
runner_filenames = [osp.splitext(osp.basename(v))[0] for v in os.listdir(runner_folder) if v.endswith('runner.py')]
# import all the model modules
_model_modules = [importlib.import_module(f'runners.{file_name}') for file_name in runner_filenames]


def build_runner(opt):
    """Build runner from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Runner type.
    """
    opt = deepcopy(opt)
    runner = RUNNER_REGISTRY.get(opt['type'])(opt)
    print(f'Runner [{runner.__class__.__name__}] is created.')
    return runner
