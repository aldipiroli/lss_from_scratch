import os
import sys
import torch

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from lss.utils.misc import load_config
from lss.test.test_camera_utils import load_data, CAMERAS
from lss.model.model import ResNet18


def test_model_forward_pass():
    config = load_config("lss/config/nuscenes_mini_config.yaml")
    images, intrinsics, extrinsics = load_data(npz_path="lss/test/data/none.npz")
    all_imgs = []
    for i, cam in enumerate(CAMERAS):
        img = images[cam]
        all_imgs.append(img)
    all_imgs = torch.stack(all_imgs, 1)
    B, n_cam, ch, h, w = (
        all_imgs.shape[0],
        all_imgs.shape[1],
        all_imgs.shape[2],
        all_imgs.shape[3],
        all_imgs.shape[4],
    )
    all_imgs = all_imgs.reshape(B * n_cam, ch, h, w)
    model = ResNet18(config)
    feats = model(all_imgs)
    assert feats is not None
