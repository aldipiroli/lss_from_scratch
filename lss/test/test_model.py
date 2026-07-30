import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from lss.utils.misc import load_config
from lss.test.test_camera_utils import load_data, CAMERAS
from lss.model.model import ResNet18, ShootHead
from lss.utils.camera_utils import batch_data
import torch


def test_model_forward_pass():
    config = load_config("lss/config/nuscenes_mini_config.yaml")
    images, intrinsics, extrinsics = load_data(npz_path="lss/test/data/none.npz")
    images = batch_data(images, CAMERAS)
    model = ResNet18(config)
    feats, dist = model(images)
    assert feats.shape[0] == dist.shape[0]


def test_shoot_head():
    config = load_config("lss/config/nuscenes_mini_config.yaml")
    bevs = torch.rand(2, 512, 100, 100)
    model = ShootHead(config)
    cost = model(bevs)
    assert cost.shape[2] == bevs.shape[2]
    assert cost.shape[3] == bevs.shape[3]
    assert cost.shape[1] == 1
