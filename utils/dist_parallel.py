import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import functools
import os

from .singleton import Singleton


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper



def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= dist.get_world_size()
    return rt


def get_ip(node_name: str) -> str:
    node_name_split = node_name.split('-')
    ip_ = ''
    for idx, part in enumerate(node_name_split[2:]):
        if ',' in part:
            part = part.split(',')[0]
        ip_ += part
        if idx != 3:
            ip_ += '.'
        else:
            ip_ = ip_.replace('[','')
            break
    return ip_


class Parallel(metaclass=Singleton):

    def __init__(self, num_gpus, distributed_training=False, port='10086'):
        """
        args:
            num_gpus: GPU number on a single machine, used in DataParallel
            distributed_training: enable distributed training or not
            port: port for distributed training
        """

        self.distributed_training = distributed_training
        self.rank = None
        self.local_rank = None
        self.world_size = None

        # 单机单/多卡
        if not self.distributed_training:
            # 设置设备
            if num_gpus <= 0:
                device = torch.device("cpu")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.device = device
            self.num_gpus = num_gpus

            self.device_ids = range(num_gpus) if num_gpus > 0 else []

            print("There are %d GPUs used" % num_gpus)
        # 多机多卡，一个Parallel代表一个节点实体(即一个机器)
        else:
            self.rank = int(os.environ['SLURM_PROCID'])
            self.local_rank = int(os.environ['SLURM_LOCALID'])
            self.world_size = int(os.environ['SLURM_NTASKS'])
            self.ip = get_ip(os.environ['SLURM_STEP_NODELIST'])
            print('IP is {} world_size is {}, rank is {}, local_rank is {}'.format(self.ip, self.world_size, self.rank, self.local_rank))

            # These are the parameters used to initialize the process group
            host_addr_full = 'tcp://' + self.ip + ':' + port
            dist.init_process_group("nccl", init_method=host_addr_full, rank=self.rank, world_size=self.world_size)
            torch.cuda.set_device(self.local_rank)

            print(
                f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
                + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
            )
            # n = num_gpus // self.world_size
            # n = num_gpus
            # self.device_ids = list(range(self.local_rank * n, (self.local_rank + 1) * n))
            self.device_ids = [self.local_rank]

            print(
                f"[{os.getpid()}] rank = {dist.get_rank()}, "
                + f"world_size = {dist.get_world_size()}, device_ids = {self.device_ids}"
            )

            # 主卡
            self.device = self.device_ids[0]

    def wrapper(self, entry, find_unused_parameters=True):
        """
        args:
            entry: nn.Module
        """
        if self.distributed_training:
            entry = entry.to(self.device)
            return DDP(entry, device_ids=self.device_ids, find_unused_parameters=find_unused_parameters)
        else:
            if len(self.device_ids) <= 1:
                entry = entry.to(self.device)
            else:
                entry = entry.to(self.device)
                entry = nn.DataParallel(entry, self.device_ids)
            return entry

    def close(self):
        if self.distributed_training:
            dist.destroy_process_group()