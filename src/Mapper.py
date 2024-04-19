import os
import time
import numpy as np
import torch
from colorama import Fore, Style
from torch import nn
from src.utils import get_tensor_from_camera
from src.tools.eval_ate import convert_poses, evaluate
from src.Datasets import get_datasetloader
from src.Scene import Scene, cameraInfo
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

class Mapper(object):
    def __init__(self, cfg, args, slam):
        self.cfg = cfg
        self.args = args
        self.output = slam.output
        self.verbose = slam.verbose
        self.ckptsdir = slam.ckptsdir
        self.sleepTime = cfg['sleepTime']
        self.scale = cfg['scale']  # scale of pose
        self.device = cfg["device"]
        self.do_obj = self.cfg["do_obj"]
        self.marching_cube_bound = slam.marching_cube_bound.to(self.device)
        self.bound = slam.bound.to(self.device)
        self.logger = slam.logger
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy
        self.dirs = cameraInfo(self.device, self.H, self.W, self.fx, self.fy, self.cx, self.cy)
        self.debug = cfg["debug"]["flag"]
        self.checkpoint = cfg["debug"]["checkpoint"]
        self.showLoss = cfg["debug"]["showMapperLoss"]
        self.gt_camera = cfg['tracking']['gt_camera']
        self.max_n_models = cfg["mapping"]["max_n_models"]
        self.every_frame = cfg["mapping"]["every_frame"]
        self.pixels_per_obj = cfg['mapping']['pixels']
        self.pixels_bg = cfg['mapping']['pixels_bg']
        self.optimize_window_size = cfg["mapping"]['optimize_window_size']
        self.optimize_window_size_bg = cfg["mapping"]['optimize_window_size_bg']
        self.idx = slam.idx
        self.mapping_idx = slam.mapping_idx
        self.mapping_first_frame = slam.mapping_first_frame
        self.estimate_c2w_list = slam.estimate_c2w_list
        self.estimate_relative_c2w_list = slam.estimate_relative_c2w_list
        self.gt_c2w_list = slam.gt_c2w_list
        self.vis_dict = slam.vis_dict
        self.mapping_cnt = slam.mapping_cnt
        self.lost = False
        self.no_inst = False
        self.pe_buffer = None
        self.pe_param = None
        self.pe_model = None
        self.fc_buffer = None
        self.fc_param = None
        self.fc_model = None
        self.optimiser = None
        self.optimize_inst_ids = []
        self.fc_models = []
        self.pe_models = []
        self.frame_reader = slam.frame_reader
        self.frame_loader = get_datasetloader(self.frame_reader, mapperFrame=self.every_frame)
        self.frameloader_iterator = iter(self.frame_loader)
        self.currentNumberOfModels = 0
        self.n_img = len(self.frame_loader)

    def pose_difference(self, P1, P2):

        diff = torch.inverse(P1) @ P2
        diff_quad = get_tensor_from_camera(diff)
        t = diff_quad[-3:]
        quad = diff_quad[:4]

        translation_diff = torch.norm(t).item()
        angle_diff = 2 * np.arccos(abs(torch.dot(quad, torch.tensor([1, 0, 0, 0], dtype=torch.float32)))) * (
                180 / np.pi)

        return translation_diff, angle_diff

    def optimize_map(self, idx):
        currentC2w = self.estimate_c2w_list[idx]
        framesNumber = self.vis_dict[0].n_keyframes
        optimized_ids = []
        for i in range(framesNumber):
            estimatedC2w = self.estimate_c2w_list[i * self.every_frame].clone()
            loss_t, loss_r = self.pose_difference(currentC2w, estimatedC2w)
            optimized_ids.append([i, loss_t, loss_r])
        sorted_data = sorted(optimized_ids, key=lambda x: x[2])
        top = sorted_data[:self.optimize_window_size_bg]
        sorted_top = sorted(top, key=lambda x: x[1], reverse=True)
        optimized_ids = [x[0] for x in sorted_top]

        smooth = True
        if self.first_map:
            self.first_map = False
            smooth = False
            num_joint_iters = self.cfg['mapping']['iters_first']
        else:
            num_joint_iters = self.cfg['mapping']['iters']

        frameNumber_bg = len(optimized_ids) * num_joint_iters
        self.pixels_bg_per_frame = self.pixels_bg // len(optimized_ids)
        scene_obj = self.vis_dict[0]
        bg_data, bg_mask, bg_id = scene_obj.get_training_samples(frameNumber_bg, self.pixels_bg_per_frame,
                                                                 optimized_ids)

        uniqueIDs = sorted(torch.unique(bg_id).cpu().tolist())
        frameNum = len(uniqueIDs)
        optIDX = 1
        BAFalg = frameNum > 1 and self.cfg['mapping']['BA']
        if not self.cfg["mapping"]["optim_cur"] and frameNum == 2:
            BAFalg = False
        poseOptimize = None
        if BAFalg:
            if self.cfg["mapping"]["optim_cur"]:
                optIDs = uniqueIDs[optIDX:]
            else:
                optIDs = uniqueIDs[optIDX:-1]
            BAPose = self.estimate_c2w_list.clone()[optIDs].to(self.device)
            BAPose_quad = nn.Parameter(torch.cat([matrix_to_quaternion(BAPose[:, :3, :3]), BAPose[:, :3, 3]], dim=-1))
            poseOptimize = torch.optim.Adam([{'params': BAPose_quad, 'lr': 0.001}])
            poseOptimize.zero_grad()
            if self.verbose:
                initLoss = []
                for i in optIDs:
                    quad_loss = torch.abs(
                        get_tensor_from_camera(self.gt_c2w_list[i].to(self.device)) - get_tensor_from_camera(
                            self.estimate_c2w_list[i].to(self.device))).mean().item()
                    initLoss.append(quad_loss)
        else:
            currentEstimation = self.estimate_c2w_list.clone().to(self.device)


        self.train_optimiser.zero_grad()
        if hasattr(self, "obj_optimiser"):
            self.obj_optimiser.zero_grad()
        for joint_iter in range(num_joint_iters):
            if BAFalg:
                c2w = torch.eye(4, device=BAPose_quad.device).unsqueeze(0).repeat(BAPose_quad.shape[0], 1, 1)
                c2w[:, :3, :3] = quaternion_to_matrix(BAPose_quad[:, :4])
                c2w[:, :3, 3] = BAPose_quad[:, 4:]
                currentEstimation = self.estimate_c2w_list.clone().to(self.device)
                currentEstimation[optIDs] = c2w

            bg_frame_idx = slice(joint_iter * len(optimized_ids),
                                 (joint_iter + 1) * len(optimized_ids))
            scene_bg = self.vis_dict[0]

            bg_id_idx = bg_id[bg_frame_idx].squeeze(1)
            bg_rgb_idx = bg_data[bg_frame_idx, :, 0:3].reshape(-1, 3)
            bg_dir_idx = bg_data[bg_frame_idx, :, 3:6]
            bg_depth_idx = bg_data[bg_frame_idx, :, 6].reshape(-1, 1)

            bg_c2w_idx = currentEstimation[bg_id_idx, :3, :]. \
                unsqueeze(1). \
                expand(-1, self.pixels_bg_per_frame, -1, -1)
            bg_origin_idx = bg_c2w_idx[:, :, :, 3].reshape(-1, 3)
            bg_dirW_idx = (bg_c2w_idx[:, :, :, :3] @ bg_dir_idx[..., None]). \
                squeeze(-1). \
                reshape([self.pixels_bg_per_frame * len(optimized_ids), 3])


            bg_loss = scene_bg.vMapModel.forward(bg_origin_idx, bg_dirW_idx, bg_rgb_idx, bg_depth_idx,
                                                 smooth=smooth, tracker=False)
            bg_loss.backward()
            self.train_optimiser.step()
            self.train_optimiser.zero_grad()
            if hasattr(self, "obj_optimiser"):
                self.obj_optimiser.step()
                self.obj_optimiser.zero_grad()

            if poseOptimize is not None:
                poseOptimize.step()
                poseOptimize.zero_grad()
            debugShow = joint_iter % 5 == 0 and self.debug and self.showLoss
            if (debugShow or joint_iter == num_joint_iters - 1) and self.verbose:
                print("frame_id: ", self.idx[0].item(), ", iter: ", joint_iter, ', loss: ',
                      bg_loss.item())

        if BAFalg:
            BAPose_quad = BAPose_quad.detach()
            c2w = torch.eye(4, device=BAPose_quad.device).unsqueeze(0).repeat(BAPose_quad.shape[0], 1, 1)
            c2w[:, :3, :3] = quaternion_to_matrix(BAPose_quad[:, :4])
            c2w[:, :3, 3] = BAPose_quad[:, 4:]
            self.estimate_c2w_list[optIDs] = c2w.cpu().clone()
            self.estimate_relative_c2w_list[optIDs] = c2w.cpu().clone()

            if self.verbose:
                index = 0
                for i in optIDs:
                    quad_loss = torch.abs(
                        get_tensor_from_camera(self.gt_c2w_list[i].to(self.device)) - get_tensor_from_camera(
                            self.estimate_c2w_list[i].to(self.device))).mean().item()
                    if self.verbose:
                        print(f'frame {i},camera tensor error: {initLoss[index]:.4f}->{quad_loss:.4f}')
                    index += 1
        if self.do_obj:
            with torch.no_grad():
                model_id = 0
                for obj_id in self.optimize_inst_ids:
                    if obj_id != 0:
                        scene_obj = self.vis_dict[obj_id]
                        for i, param in enumerate(scene_obj.vMapModel.fc_occ_map.parameters()):
                            param.copy_(self.fc_param[i][model_id])
                        for i, param in enumerate(scene_obj.vMapModel.pe.parameters()):
                            param.copy_(self.pe_param[i][model_id])
                        model_id += 1

    def run(self):
        pre_mapping_index = -1
        self.first_map = True
        while True:
            while True:
                idx = self.idx[0].clone().item()
                if idx == self.n_img - 1 or self.gt_camera:
                    break
                if idx % self.every_frame == 0 and idx != pre_mapping_index:
                    break
                time.sleep(self.sleepTime)
            if pre_mapping_index != idx:
                sample = next(self.frameloader_iterator)
                gt_color = sample["image"].to(self.device)
                gt_depth = sample["depth"].to(self.device)
                gt_c2w = sample["T"].to(self.device)
                features = sample["features"]
                if not hasattr(self, "optimize_lr"):
                    self.optimize_lr = self.cfg["objNetwork"]["optimize_lr"]
                    self.weight_decay = self.cfg["mapping"]["weight_decay"]
                if self.gt_camera or idx == 0:
                    self.estimate_c2w_list[idx] = gt_c2w.clone().cpu()
                if 0 in self.vis_dict.keys():
                    scene_obj = self.vis_dict[0]
                    scene_obj.append_keyframe(gt_color, gt_depth, features,idx)
                else:
                    scene_bg = Scene(self.cfg, 0, self.n_img, self.dirs, gt_color, gt_depth, idx)
                    self.vis_dict.update({0: scene_bg})
                if not hasattr(self, "train_optimiser"):
                    self.train_optimiser = torch.optim.Adam(self.vis_dict[0].vMapModel.decoder.get_optParameter(),
                                                            betas=(0.9, 0.99))
            self.optimize_map(idx)
            if self.verbose:
                print(Fore.GREEN)
                print("Mapping Frame {}".format(idx))
                print(Style.RESET_ALL)

            if (idx > 0 and idx % self.cfg["debug"]["c2wSave"] == 0) or idx == self.n_img - 1:
                if os.path.exists(self.ckptsdir):
                    if not os.path.exists(self.ckptsdir + "/targetPlot"):
                        os.makedirs(self.ckptsdir + "/targetPlot")
                    # compute absolute
                    estimate_c2w_list = self.estimate_c2w_list[0:idx + 1]
                    gt_c2w_list = self.gt_c2w_list[0:idx + 1]
                    self.logger.log(idx, estimate_c2w_list, gt_c2w_list)
                    poses_gt, mask = convert_poses(gt_c2w_list, idx, self.scale)
                    poses_est, _ = convert_poses(estimate_c2w_list, idx, self.scale)
                    poses_est = poses_est[mask]
                    results = evaluate(poses_gt, poses_est,
                                       plot='{}/targetPlot/plotTar_{:05d}.png'.format(self.ckptsdir, idx),
                                       showResults=False)
                    with open('{}/RMSE.txt'.format(self.ckptsdir), 'a') as file:
                        file.writelines(str(results) + "\n")

                    # compute relative
                    estimate_c2w_list = torch.zeros((idx + 1, 4, 4))
                    for i in range(idx + 1):
                        if i % self.every_frame == 0:
                            estimate_c2w_list[i] = self.estimate_relative_c2w_list[i]
                        else:
                            kf_id = int(i // self.every_frame) * self.every_frame
                            estimate_c2w_list[i] = self.estimate_relative_c2w_list[kf_id] @ \
                                                   self.estimate_relative_c2w_list[i]
                    self.logger.log(idx, estimate_c2w_list, gt_c2w_list, name="_rel")
                    poses_gt, mask = convert_poses(gt_c2w_list, idx, self.scale)
                    poses_est, _ = convert_poses(estimate_c2w_list, idx, self.scale)
                    poses_est = poses_est[mask]
                    results = evaluate(poses_gt, poses_est,
                                       plot='{}/targetPlot/plotTarRel_{:05d}.png'.format(self.ckptsdir, idx),
                                       showResults=False)
                    with open('{}/RMSE_rel.txt'.format(self.ckptsdir), 'a') as file:
                        file.writelines(str(results) + "\n")
            pre_mapping_index = idx
            self.mapping_first_frame[0] = 1
            self.mapping_idx[0] = idx
            self.mapping_cnt[0] += 1
            if self.gt_camera:
                self.gt_c2w_list[pre_mapping_index] = gt_c2w.clone().cpu()
                self.idx[0] = pre_mapping_index + 1
            if idx == self.n_img - 1:
                break


