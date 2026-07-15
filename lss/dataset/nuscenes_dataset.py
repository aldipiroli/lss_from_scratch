import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import numpy as np

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
                "calibrated_sensor",
                sample_data["calibrated_sensor_token"]
            )

            # Intrinsic matrix (3x3)
            K = torch.tensor(
                calib["camera_intrinsic"],
                dtype=torch.float32
            )

            # Extrinsic matrix Camera -> Ego (4x4)
            R = Quaternion(calib["rotation"]).rotation_matrix
            t = np.array(calib["translation"])

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            intrinsics[cam] = K
            extrinsics[cam] = torch.tensor(T, dtype=torch.float32)
        return images, intrinsics, extrinsics

    def __getitem__(self, idx):
        sample = self.samples[idx]

        images, intrinsics, extrinsics = self.get_camera_data(sample)

        return {
            "images": images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
        }