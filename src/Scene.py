import random
import torch
from src.Model import Model


class Scene:
    def __init__(self, cfg, obj_id, n_img, dirs, rgb, depth, frame_id) -> None:
        self.cfg = cfg
        self.dirs = dirs
        self.device = cfg["device"]
        self.n_img = n_img

        self.obj_id = obj_id
        self.frames_width = rgb.shape[0]
        self.frames_height = rgb.shape[1]
        self.keyframe_buffer_size = n_img // cfg['mapping']['every_frame'] + 1
        self.total_pixels = self.frames_width * self.frames_height
        self.kf_pixels = int(cfg["keyframe"]["kf_number_bg"] * self.total_pixels)

        self.vMapModel = Model(cfg).to(self.device)
        self.kf_ids = torch.empty(self.keyframe_buffer_size, 1, dtype=torch.long, device=self.device)
        self.kf_data = torch.empty(self.keyframe_buffer_size, self.kf_pixels, 7, device=self.device)

        idxs = random.sample(range(0, self.total_pixels), self.kf_pixels)
        data = torch.cat([rgb, self.dirs.rays_dir_cache, depth[..., None]], dim=-1).reshape(-1, 7)
        self.kf_ids[0, 0] = frame_id
        self.kf_data[0, :, :] = data[idxs]
        self.n_keyframes = 1

    def append_keyframe(self, rgb, depth, features, frame_id):
        featured = False
        Grided = False
        if not hasattr(self, 'keyframe_step'):
            self.keyframe_step = self.cfg['keyframe']['keyframe_step_bg']
        if frame_id % self.keyframe_step == 0:
            if featured:
                w_indices = features[:,:, 0].squeeze()
                h_indices = features[:, :,1].squeeze()
                indices = w_indices * self.frames_height + h_indices
                idxs = torch.tensor(random.sample(range(0, self.total_pixels), self.kf_pixels - indices.shape[0]))
                idxs = torch.cat((idxs, indices)).long()
            else:
                idxs = random.sample(range(0, self.total_pixels), self.kf_pixels)
            # if Grided:
            #
            data = torch.cat([rgb, self.dirs.rays_dir_cache, depth[..., None]], dim=-1).reshape(-1, 7)
            self.kf_ids[self.n_keyframes, 0] = frame_id
            self.kf_data[self.n_keyframes, :, :] = data[idxs]
            self.n_keyframes += 1

    def get_training_samples(self, n_frames, n_samples, optimized_ids):
        n_optimize_frame = len(optimized_ids)
        keyframe_ids = torch.tensor(optimized_ids * (n_frames // n_optimize_frame), dtype=torch.long,
                                    device=self.device)
        indice = torch.randint(0, self.kf_pixels, (n_frames, n_samples), device=self.device)
        sampled_data = torch.gather(self.kf_data[keyframe_ids], 1, indice.unsqueeze(-1).expand(-1, -1, 7))
        sampled_id = self.kf_ids[keyframe_ids]
        sampled_mask = None
        if self.cfg["do_obj"]:
            sampled_mask = torch.gather(self.kf_mask[keyframe_ids], 1, indice.unsqueeze(-1).expand(-1, -1, 1))
        return sampled_data, sampled_mask, sampled_id

    def get_tracking_samples(self, n_samples, gt_color, gt_depth, features):
        if not hasattr(self, 'iW'):
            self.iW = self.cfg["tracking"]["ignore_edge_W"]
            self.iH = self.cfg["tracking"]["ignore_edge_H"]
            self.tracking_pixels = (self.frames_width - 2 * self.iW) * (self.frames_height - 2 * self.iH)

        global w_indices, h_indices
        featured = False
        Grided = False
        featureNum = 0
        if featured:
            features = features[features[:, :, 0] >= self.iW]
            features = features[features[:,  0] < self.frames_width - self.iW]
            features = features[features[:,  1] >= self.iH]
            features = features[features[:,  1] < self.frames_height - self.iH]
            w_indices = features[:,  0].squeeze()-self.iW
            h_indices = features[:,  1].squeeze()-self.iH
            featureNum = features.shape[0]
        # if Grided:
        #
        indice = random.sample(range(self.tracking_pixels), int(n_samples - featureNum))
        indice = torch.tensor(indice)
        indice_w, indice_h = indice % (self.frames_width - self.iW * 2), torch.div(indice,
                                                                                   self.frames_width - self.iW * 2,
                                                                                   rounding_mode='trunc')
        if featured:
            indice_w = torch.cat((w_indices, indice_w)).long()
            indice_h = torch.cat((h_indices, indice_h)).long()

        sampled_rgb = gt_color[self.iW:-self.iW, self.iH:-self.iH, :][indice_w, indice_h]
        sampled_depth = gt_depth[self.iW:-self.iW, self.iH:-self.iH][indice_w, indice_h]
        sampled_ray_dirs = self.dirs.rays_dir_cache[self.iW:-self.iW, self.iH:-self.iH][indice_w, indice_h]

        return sampled_rgb, sampled_ray_dirs, sampled_depth


class cameraInfo:

    def __init__(self, device, H, W, fx, fy, cx, cy) -> None:
        self.device = device
        self.width = W
        self.height = H
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.rays_dir_cache = self.get_rays_dirs()

    def get_rays_dirs(self):
        idx_w = torch.arange(end=self.width, device=self.device)  # x1200
        idx_h = torch.arange(end=self.height, device=self.device)  # y680
        dirs = torch.ones((self.width, self.height, 3), device=self.device)
        dirs[:, :, 0] = ((idx_w - self.cx) / self.fx)[:, None]
        dirs[:, :, 1] = ((idx_h - self.cy) / self.fy)
        return dirs
