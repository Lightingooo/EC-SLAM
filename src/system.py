import os
import time
import torch
import numpy as np
import torch.multiprocessing
from threading import Thread
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.Logger import Logger
from src.Datasets import get_dataset


class EC_SLAM():
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.verbose = cfg['verbose']
        self.dataset = cfg['dataset']
        if args.output is None:
            self.output = cfg['data']['output']
        else:
            self.output = args.output
        self.ckptsdir = os.path.join(self.output, 'ckpts')
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(f'{self.output}/mesh', exist_ok=True)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.update_cam()
        self.scale = cfg['scale']
        self.load_bound(cfg)
        self.max_n_models = cfg['mapping']['max_n_models']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.obj_id = -1
        self.frame_reader = get_dataset(cfg)
        self.n_img = len(self.frame_reader)
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.estimate_relative_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.idx = torch.zeros((1)).int()
        self.mapping_first_frame = torch.zeros((1)).int()
        self.mapping_idx = torch.zeros((1)).int()
        self.mapping_cnt = torch.zeros((1)).int()
        self.vis_dict = {}
        self.logger = Logger(self)
        self.mapper = Mapper(cfg, args, self)
        self.tracker = Tracker(cfg, args, self)
        self.print_output_desc()

    def print_output_desc(self):
        self.logger.logPrinter.info(f"INFO: The output folder is {self.output}")

    def update_cam(self):
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx * self.fx
            self.fy = sy * self.fy
            self.cx = sx * self.cx
            self.cy = sy * self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]
        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge'] * 2
            self.W -= self.cfg['cam']['crop_edge'] * 2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']

    def load_bound(self, cfg):
        self.bound = torch.from_numpy(
            np.array(cfg['mapping']['bound']) * self.scale)
        self.marching_cube_bound = torch.from_numpy(np.array(self.cfg['mapping']['marching_cubes_bound']))

    def tracking(self, rank):
        while True:
            if self.mapping_first_frame[0] == 1:
                break
            time.sleep(1)

        self.tracker.run()

    def mapping(self, rank):
        self.mapper.run()

    def run(self):
        processes = []
        for rank in range(2):
            if rank == 0 and not self.gt_camera:
                p = Thread(target=self.tracking, args=(rank,), name="1")
                p.start()
                processes.append(p)
                # p = mp.Process(target=self.tracking, args=(rank,))
            elif rank == 1:
                p = Thread(target=self.mapping, args=(rank,), name="2")
                p.start()
                processes.append(p)
                # p = mp.Process(target=self.mapping, args=(rank,))
        for p in processes:
            p.join()


if __name__ == '__main__':
    pass
