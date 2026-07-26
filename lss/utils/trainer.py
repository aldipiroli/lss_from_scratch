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
from lss.dataset.nuscenes_dataset import CAMERAS

CAMERAS = CAMERAS[:1]


class Trainer(TrainerBase):
    def __init__(self, config, logger):
        super().__init__(config, logger)

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
        for n_iter, (all_images, intrinsics, extrinsics) in pbar:
            images = batch_data(all_images, CAMERAS)
            images = images.cuda()
            feats, dist = self.model(images)
            feats, dist = self.post_process_model_output(feats, dist)
            all_rays, all_feats = self.lyft_features(
                images, feats, dist, intrinsics, extrinsics
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
            img = images[:, i]
            extrinsic = extrinsics[cam].cuda()
            K = intrinsics[cam].cuda()
            K = scale_camera_intrinsic(input_size, feats_size, K)
            depths = get_depths(self.config["LSS"]["depth_config"])
            rays, feats = pixel_to_camera_rays(img, K)
            rays = add_depth_along_ray(rays, depths)
            rays = rays.reshape(B, -1, rays.shape[-1])
            rays = camera_to_ego(rays, extrinsic)

            all_rays.append(rays)
            all_feats.append(feats)

        all_rays = torch.stack(all_rays, 1)
        all_feats = torch.stack(all_feats, 1)
        return all_rays, all_feats
