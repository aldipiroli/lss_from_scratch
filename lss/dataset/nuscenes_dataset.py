import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from nuscenes.nuscenes import NuScenes


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

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Front camera
        cam_token = sample["data"]["CAM_FRONT"]
        cam_data = self.nusc.get("sample_data", cam_token)

        img_path = os.path.join(self.root_dir, cam_data["filename"])

        image = read_image(img_path).float() / 255.0

        return {
            "image": image,
            "sample_token": sample["token"],
            "timestamp": sample["timestamp"],
        }
