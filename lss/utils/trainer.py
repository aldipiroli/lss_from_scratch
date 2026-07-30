import torch
from tqdm import tqdm

from lss.utils.trainer_base import TrainerBase
from lss.utils.camera_utils import (
    batch_data,
    unbatch_data,
    scale_camera_intrinsic,
    pixel_to_camera_rays,
    add_depth_along_ray,
    get_depths,
    camera_to_ego,
)
from lss.utils.bev_utils import project_pcl_to_pillar, scatter_feautres_to_bev
from lss.dataset.nuscenes_dataset import CAMERAS
from lss.model.model import ShootHead
import pickle
import numpy as np

CAMERAS = CAMERAS[:1]


class Trainer(TrainerBase):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.shoot_head = ShootHead(config)
        self.shoot_head = self.shoot_head.cuda()
        self.load_trajectory_library()

    def load_trajectory_library(self):
        with open(self.config["DATA"]["trajectory_library_path"], "rb") as f:
            self.trajectory_lib = pickle.load(f)
        self.trajectory_lib = torch.tensor(np.array(self.trajectory_lib))
        self.logger.info("Loaded trajectory_library_path!")

    def train(self):
        self.logger.info("Started training..")
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.config["OPTIM"]["num_epochs"]):
            self.epoch = epoch
            self.train_one_epoch()
            self.evaluate_model()
            self.save_checkpoint(epoch)

    def train_one_epoch(self):
        self.model.train()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for n_iter, (all_images, intrinsics, extrinsics, gt_trajectory) in pbar:
            images = batch_data(all_images, CAMERAS)
            images = images.to(self.device)
            feats, dist = self.model(images)
            feats, dist = self.post_process_model_output(feats, dist)
            all_rays, all_feats = self.lyft_features(
                images, feats, dist, intrinsics, extrinsics
            )
            all_bevs = self.splat_features(all_rays, all_feats)
            all_cost_map = self.shoot_head(all_bevs)
            all_trajectory_costs = self.get_cost_per_trajectory(all_cost_map)
            loss, loss_dict = self.loss_fn(
                all_trajectory_costs, gt_trajectory, self.trajectory_lib
            )
            self.write_dict_to_tb(loss_dict, self.total_iters_train, prefix="train")
            loss.backward()
            self.accumulate_gradients()
            self.total_iters_train += 1
            pbar.set_postfix(
                {
                    "mode": "train",
                    "epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}",
                    "loss": loss.item(),
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )
            self.write_float_to_tb(
                self.optimizer.param_groups[0]["lr"], "train/lr", self.total_iters_train
            )

    @torch.no_grad()
    def evaluate_model(self):
        self.model.eval()
        pass

    def post_process_model_output(self, feats, dist):
        feats = unbatch_data(
            feats,
            batch_size=self.config["OPTIM"]["batch_size"],
            n_cameras=len(CAMERAS),
        )
        dist = unbatch_data(
            dist,
            batch_size=self.config["OPTIM"]["batch_size"],
            n_cameras=len(CAMERAS),
        )
        return feats, dist

    def lyft_features(self, images, feats, dist, intrinsics, extrinsics):
        input_size = (images.shape[-2], images.shape[-1])
        feats_size = (feats.shape[-2], feats.shape[-1])
        B = feats.shape[0]
        all_rays = []
        all_feats = []
        for i, cam in enumerate(CAMERAS):
            curr_feats = feats[:, i]
            extrinsic = extrinsics[cam].to(self.device)
            K = intrinsics[cam].to(self.device)
            K = scale_camera_intrinsic(input_size, feats_size, K)
            depths = get_depths(self.config["LSS"]["depth_config"])
            rays, cam_feats = pixel_to_camera_rays(curr_feats, K)

            rays = add_depth_along_ray(rays, depths)
            cam_feats = cam_feats[:, :, None, :].repeat(1, 1, len(depths), 1)
            rays = rays.reshape(B, -1, rays.shape[-1])
            cam_feats = cam_feats.reshape(B, -1, cam_feats.shape[-1])
            rays = camera_to_ego(rays, extrinsic)

            all_rays.append(rays)
            all_feats.append(cam_feats)

        all_rays = torch.stack(all_rays, 1)  # B, n_cam, n_points * n_depths, 3
        all_feats = torch.stack(all_feats, 1)  # B, n_cam, n_points * n_depths, C
        return all_rays, all_feats

    def splat_features(self, all_rays, all_feats):
        bev_cfg = self.config["LSS"]["bev_config"]
        all_rays_quant, valid = project_pcl_to_pillar(all_rays, bev_cfg)
        B, n_cam, n_points, n_feats = all_feats.shape

        all_feats = all_feats.reshape(B, n_cam * n_points, n_feats)
        all_rays_quant = all_rays_quant.reshape(B, -1, 3)
        valid = valid.reshape(B, -1)

        all_bevs = []
        for batch_idx in range(B):
            curr_feats = all_feats[batch_idx]
            curr_bev_idx = all_rays_quant[batch_idx]
            curr_valid = valid[batch_idx]
            bev = scatter_feautres_to_bev(
                curr_feats[curr_valid], curr_bev_idx[curr_valid], bev_cfg
            )
            all_bevs.append(bev)
        all_bevs = torch.stack(all_bevs, 0)
        all_bevs = all_bevs.permute(0, 3, 1, 2)  # B, C, H, W
        return all_bevs

    def get_cost_per_trajectory(self, all_cost_map):
        self.trajectory_lib = self.trajectory_lib
        all_trajectory_costs = []
        for curr_traj in self.trajectory_lib:
            curr_traj_padded = torch.cat(
                [curr_traj, torch.zeros(curr_traj.shape[0], 1)], dim=1
            )
            curr_traj_quant, valid = project_pcl_to_pillar(
                curr_traj_padded, self.config["LSS"]["bev_config"]
            )
            x = curr_traj_quant[valid][:, 0].long()
            y = curr_traj_quant[valid][:, 1].long()
            curr_trajectory_cost = torch.sum(all_cost_map[:, 0, y, x], -1)
            all_trajectory_costs.append(curr_trajectory_cost)
        return torch.stack(all_trajectory_costs, dim=1)
