import os
import logging
import torch


class Logger(object):

    def __init__(self, slam):
        log_level = logging.INFO
        logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')
        self.logPrinter = logging.getLogger()

        self.verbose = slam.verbose
        self.ckptsdir = slam.ckptsdir
        self.vis_dict = slam.vis_dict

    def log(self, idx, es_c2s, gt_c2w, name=""):
        path = os.path.join(self.ckptsdir, '{:05d}{}.tar'.format(idx, name))
        torch.save({
            'gt_c2w_list': gt_c2w[:idx + 1],
            'estimate_c2w_list': es_c2s[:idx + 1],
            'idx': idx,
        }, path, _use_new_zipfile_serialization=False)
        if self.verbose:
            print('Saved checkpoints at', path)
