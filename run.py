import argparse
import random
import numpy as np
import torch
import os

import src.utils
from src.system import EC_SLAM

def setup_seed(seed):
    # print("set seed ",seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    # setup_seed(43)
    parser = argparse.ArgumentParser(description='Arguments for running the EC_SLAM.')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    args = parser.parse_args()
    cfg = src.utils.load_config(args.config, 'configs/EC_SLAM.yaml')  # dataset&&slam yaml concat
    slam = EC_SLAM(cfg, args)

    slam.run()


if __name__ == '__main__':
    main()
