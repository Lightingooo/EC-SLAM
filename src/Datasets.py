import os
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler


class SeqSampler(Sampler):

    def __init__(self, n_samples, step, include_last=True):
        self.n_samples = n_samples
        self.step = step
        self.include_last = include_last

    def __iter__(self):
        if self.include_last:
            if self.n_samples % self.step != 1:
                return iter(list(range(0, self.n_samples, self.step)) + [self.n_samples - 1])
            else:
                return iter(list(range(0, self.n_samples, self.step)))
        else:
            return iter(range(0, self.n_samples, self.step))

    def __len__(self) -> int:
        return self.n_samples


class DepthScale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, depth):
        depth = depth.astype(np.float32)
        return depth * self.scale


class DepthFilter(object):
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def __call__(self, depth):
        far_mask = depth > self.max_depth
        depth[far_mask] = 0.
        return depth


def get_dataset(cfg):
    dataset = Replica(cfg)
    return dataset


def get_datasetloader(frame_reader, mapperFrame=1, multi_worker=True, num_works=4, pre_fetch=2):
    if multi_worker:
        # multi worker loader
        if mapperFrame != 1:
            dataloader = DataLoader(frame_reader, batch_size=None, shuffle=False,
                                    batch_sampler=None, num_workers=num_works, collate_fn=None,
                                    pin_memory=True, drop_last=False, timeout=0,
                                    worker_init_fn=None, generator=None, prefetch_factor=pre_fetch,
                                    persistent_workers=True, sampler=SeqSampler(len(frame_reader), mapperFrame))
        else:
            dataloader = DataLoader(frame_reader, batch_size=None, shuffle=False, sampler=None,
                                    batch_sampler=None, num_workers=num_works, collate_fn=None,
                                    pin_memory=True, drop_last=False, timeout=0,
                                    worker_init_fn=None, generator=None, prefetch_factor=pre_fetch,
                                    persistent_workers=True)
    else:
        # single worker loader
        dataloader = DataLoader(frame_reader, batch_size=None, shuffle=False, sampler=None,
                                batch_sampler=None, num_workers=0)
    return dataloader


class Replica(Dataset):
    def __init__(self, cfg):
        self.imap_mode = False
        self.root_dir = cfg['data']['input_folder']
        traj_file = os.path.join(self.root_dir, "traj_w_c.txt")
        self.Twc = np.loadtxt(traj_file, delimiter=" ").reshape([-1, 4, 4])
        self.depth_scale = 1 / cfg["cam"]['png_depth_scale']
        self.depth_transform = transforms.Compose(
            [DepthScale(self.depth_scale)])

    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir, "depth")))

    def __getitem__(self, idx):
        rgb_file = os.path.join(self.root_dir, "rgb", "rgb_" + str(idx) + ".png")
        depth_file = os.path.join(self.root_dir, "depth", "depth_" + str(idx) + ".png")
        depth = cv2.imread(depth_file, -1).astype(np.float32).transpose(1, 0)
        image = cv2.imread(rgb_file).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.03, minDistance=5)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(1, 0, 2)
        if self.depth_transform:
            depth = self.depth_transform(depth)
        T = self.Twc[idx]
        sample = {"image": image / 255., "depth": depth, "T": T, "frame_id": idx, "features": features}

        return sample
