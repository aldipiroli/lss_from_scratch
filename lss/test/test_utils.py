import os
import sys
import torch

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from lss.utils.misc import load_config
from lss.utils.camera_utils import pixel_to_camera_rays, camera_to_ego
import numpy as np

from pathlib import Path

CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def load_data(npz_path="lss/test/data/sample.npz"):
    npz_path = Path(npz_path)
    images = {}
    intrinsics = {}
    extrinsics = {}
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        images_np = data["images"].item()
        intrinsics_np = data["intrinsics"].item()
        extrinsics_np = data["extrinsics"].item()
        for cam in CAMERAS:
            images[cam] = torch.from_numpy(images_np[cam]).float()
            intrinsics[cam] = torch.from_numpy(intrinsics_np[cam]).float()
            extrinsics[cam] = torch.from_numpy(extrinsics_np[cam]).float()
    else:
        for cam in CAMERAS:
            images[cam] = torch.rand(1, 3, 900, 1600)
            intrinsics[cam] = torch.eye(3).unsqueeze(0)
            extrinsics[cam] = torch.eye(4).unsqueeze(0)
    return images, intrinsics, extrinsics


def test_load_config():
    config = load_config("lss/config/nuscenes_mini_config.yaml")
    assert config is not None


def test_pixel_to_camera_rays():
    images, intrinsics, extrinsics = load_data()
    img = images[CAMERAS[0]][0]
    K = intrinsics[CAMERAS[0]][0]

    r, feats = pixel_to_camera_rays(img, K)
    assert r.shape == (img.shape[1] * img.shape[2], 3)
    assert feats.shape == (img.shape[1] * img.shape[2], img.shape[0])


def test_camera_to_ego():
    images, intrinsics, extrinsics = load_data()
    img = images[CAMERAS[0]][0]
    K = intrinsics[CAMERAS[0]][0]
    extrinsic = extrinsics[CAMERAS[0]][0]

    r, feats = pixel_to_camera_rays(img, K)
    r = camera_to_ego(r, extrinsic)
    assert r.shape == (img.shape[1] * img.shape[2], 3)


def test_all_pixel_to_camera_rays():
    images, intrinsics, extrinsics = load_data()
    all_r = []
    all_colors = []
    for i, cam in enumerate(CAMERAS):
        img = images[cam][0]
        K = intrinsics[cam][0]
        extrinsic = extrinsics[cam][0]

        r, feats = pixel_to_camera_rays(img, K)
        r = camera_to_ego(r, extrinsic)
        all_r.append(r)
        all_colors.append(feats)

    all_r = torch.cat(all_r, 0)
    all_colors = torch.cat(all_colors, 0)
