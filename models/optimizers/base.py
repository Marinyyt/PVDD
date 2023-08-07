import torch
import torch.nn as nn
from torch.optim import Optimizer


class OptimBase(Optimizer):

    def __init__(self, params, default: dict) -> None:
        super(OptimBase, self).__init__(params, default)
