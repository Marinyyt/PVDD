import importlib
from os import path as osp
import os
from copy import deepcopy

from utils.registry import NETWORK_REGISTRY

from .discriminators import *
from .warpnet import Warpnet_512
from .base import BaseNet

# automatically scan and import network modules for registry
# scan all the files under the 'networks' folder and collect files ending with
# 'net.py'
network_folder = osp.dirname(osp.abspath(__file__))
network_filenames = [osp.splitext(osp.basename(v))[0] for v in os.listdir(network_folder) if v.endswith('net.py')]
# import all the network modules
_network_modules = [importlib.import_module(f'models.networks.{file_name}') for file_name in network_filenames]


def build_network(opt):
    """Build network from config.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Net type.
    """
    opt = deepcopy(opt)

    net = NETWORK_REGISTRY.get(opt['type'])(**opt.args)

    print(f'Net [{net.__class__.__name__}] is created.')
    return net