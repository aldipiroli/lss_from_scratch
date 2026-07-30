import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import numpy as np
from lss.utils.camera_utils import scale_camera_intrinsic
import torch.nn.functional as F

CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


class NuScenesDataset(Dataset):
    def __init__(self, cfg, mode="train", logger=None, version="v1.0-mini"):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.logger = logger
        self.root_dir = cfg["DATA"]["root_dir"]

        self.logger.info("Loading NuScenes dataset..")
        self.nusc = NuScenes(
            version=version,
            dataroot=self.root_dir,
            verbose=True,
        )
        target_scenes = cfg["DATA"]["splits"][mode]
        scene_tokens = {
            scene["token"]
            for scene in self.nusc.scene
            if scene["name"] in target_scenes
        }

        self.samples = []
        for sample in self.nusc.sample:
            scene = self.nusc.get("scene", sample["scene_token"])
            if scene["token"] in scene_tokens:
                self.samples.append(sample)

        self.samples = sorted(
            self.samples, key=lambda x: (x["scene_token"], x["timestamp"])
        )
        if logger is not None:
            logger.info(f"{mode}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def get_camera_data(self, sample):
        images = {}
        intrinsics = {}
        extrinsics = {}

        for cam in CAMERAS:
            sample_data = self.nusc.get("sample_data", sample["data"][cam])
            img_path = os.path.join(self.root_dir, sample_data["filename"])
            images[cam] = read_image(img_path).float() / 255.0

            calib = self.nusc.get(
                "calibrated_sensor", sample_data["calibrated_sensor_token"]
            )

            # Intrinsic matrix (3x3)
            K = torch.tensor(calib["camera_intrinsic"], dtype=torch.float32)

            # Extrinsic matrix Camera -> Ego (4x4)
            R = Quaternion(calib["rotation"]).rotation_matrix
            t = np.array(calib["translation"])

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            intrinsics[cam] = K
            extrinsics[cam] = torch.tensor(T, dtype=torch.float32)

            if self.cfg["DATA"]["scale_down_img"]:
                scale_factor = self.cfg["DATA"]["scale_imgs_factor"]
                old_size = images[cam].shape
                img = images[cam].unsqueeze(0)
                new_size = (
                    img.shape[-2] // scale_factor,
                    img.shape[-1] // scale_factor,
                )
                img_resized = F.interpolate(
                    img,
                    size=new_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                images[cam] = img_resized
                intrinsics[cam] = scale_camera_intrinsic(
                    old_size, new_size, intrinsics[cam]
                )

        return images, intrinsics, extrinsics

    def get_ego_pose(self, sample):
        sd = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        ego_pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
        return ego_pose

    def global_to_ego(self, point, ego_pose):
        translation = np.array(ego_pose["translation"])
        rotation = Quaternion(ego_pose["rotation"])
        point = point - translation
        point = rotation.inverse.rotate(point)
        return point

    def get_driven_trajectory(self, sample):
        horizon = self.cfg["LSS"].get("trajectory_horizon", 30)
        current_pose = self.get_ego_pose(sample)
        trajectory = []
        current = sample
        for _ in range(horizon):
            pose = self.get_ego_pose(current)
            global_xyz = np.array(pose["translation"])
            local_xyz = self.global_to_ego(global_xyz, current_pose)

            trajectory.append(local_xyz[:2])
            if current["next"] == "":
                break
            current = self.nusc.get("sample", current["next"])
        while len(trajectory) < horizon:  # pad trajectory
            trajectory.append(trajectory[-1].copy())

        return np.asarray(trajectory, dtype=np.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        images, intrinsics, extrinsics = self.get_camera_data(sample)
        trajectory = self.get_driven_trajectory(sample)
        return images, intrinsics, extrinsics, trajectory
