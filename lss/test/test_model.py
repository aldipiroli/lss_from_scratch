import os
import sys
import torch

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from lss.utils.misc import load_config
from lss.test.test_camera_utils import load_data, CAMERAS
from lss.model.model import ResNet18
from lss.utils.bev_utils import get_depths
from lss.utils.camera_utils import batch_data


def test_model_forward_pass():
    config = load_config("lss/config/nuscenes_mini_config.yaml")
    images, intrinsics, extrinsics = load_data(npz_path="lss/test/data/none.npz")
    images = batch_data(images, CAMERAS)
    model = ResNet18(config)
    feats, dist = model(images)
    n_depths = len(get_depths(config["LSS"]["depth_config"]))
    assert feats.shape[0] == dist.shape[0]
