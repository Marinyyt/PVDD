import argparse
import os
import torch
import torchvision.ops
import yaml
from easydict import EasyDict as edict

from runners import build_runner


def attatch_args_to_config(args, config):
    config.RunnerConfig.num_gpus = args.num_gpus
    config.RunnerConfig.save_path = args.save_path
    config.RunnerConfig.distributed_training = args.distributed_training
    config.RunnerConfig.port = args.port
    config.RunnerConfig.use_pavi = args.use_pavi


def setup_env():
    torch.backends.cudnn.benchmark = True


def main(config):
    setup_env()
    runner_config = config.RunnerConfig
    runner = build_runner(runner_config)
    runner.run()
    runner.finish()



if __name__ == "__main__":



    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config YAML file.')
    parser.add_argument('--num_gpus', type=int, default=1, help='GPU number, 0 means cpu is used.')
    parser.add_argument('--save_path', type=str, help='Path to save model and log.')
    parser.add_argument('--distributed_training', action='store_true')
    parser.add_argument('--port', type=str, default='10086')
    parser.add_argument('--use_pavi', action='store_true')
    args = parser.parse_args()

    with open(args.config, mode='r') as f:
        config = edict(yaml.load(f, Loader=yaml.SafeLoader))
    
    attatch_args_to_config(args, config)

    main(config)

