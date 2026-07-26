import os
import sys
import torch

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from lss.utils.misc import load_config
from lss.test.test_camera_utils import load_data, CAMERAS
from lss.utils.bev_utils import project_pcl_to_pillar, scatter_feautres_to_bev
from lss.utils.camera_utils import (
    pixel_to_camera_rays,
    add_depth_along_ray,
    camera_to_ego,
    scale_image,
    scale_camera_intrinsic,
    get_depths
)


def test_project_pcl_to_pillar():
    config = load_config("lss/config/nuscenes_mini_config.yaml")
    B, N, C = 1, 1024, 3
    pcl = torch.rand(B, N, C)
    bev_cfg = config["LSS"]["bev_config"]
    pcl_quant, valid = project_pcl_to_pillar(pcl, bev_cfg)
    assert pcl_quant.shape == pcl.shape


def test_scatter_feautres():
    config = load_config("lss/config/nuscenes_mini_config.yaml")
    N, C, C_feat = 1024, 3, 128
    pcl = torch.rand(N, C)
    feats = torch.rand(N, C_feat)
    bev_cfg = config["LSS"]["bev_config"]
    cell_size = bev_cfg["cell_size"]
    W = int((bev_cfg["x_max"] - bev_cfg["x_min"]) / cell_size)
    H = int((bev_cfg["y_max"] - bev_cfg["y_min"]) / cell_size)
    pcl_quant, valid = project_pcl_to_pillar(pcl, bev_cfg)
    bev = scatter_feautres_to_bev(feats, pcl_quant, bev_cfg)
    assert bev.shape == (H, W, C_feat)


def test_example_project_pcl_to_pillar():
    config = load_config("lss/config/nuscenes_mini_config.yaml")
    images, intrinsics, extrinsics = load_data(npz_path="lss/test/data/sample.npz")
    all_r = []
    all_feats = []
    for i, cam in enumerate(CAMERAS):
        img_full = images[cam]
        img = scale_image(img_full, 0.5)
        K = intrinsics[cam]
        K = scale_camera_intrinsic(img_full.shape, img.shape, K)
        extrinsic = extrinsics[cam]
        B = img.shape[0]

        r, feats = pixel_to_camera_rays(img, K)
        depths = get_depths(config["LSS"]["depth_config"])
        r = add_depth_along_ray(r, depths)
        feats = feats[:, :, None, :].repeat(1, 1, len(depths), 1)
        # flatten depths
        r = r.reshape(B, -1, r.shape[-1])
        feats = feats.reshape(B, -1, feats.shape[-1])

        r = camera_to_ego(r, extrinsic)
        all_r.append(r)
        all_feats.append(feats)

    all_r = torch.stack(all_r, 0).transpose(1, 0)
    all_feats = torch.stack(all_feats, 0).transpose(1, 0)

    bev_cfg = config["LSS"]["bev_config"]
    all_r_quant, valid = project_pcl_to_pillar(all_r, bev_cfg)

    B, n_cam, n_points, n_feats = all_feats.shape

    all_feats = all_feats.reshape(B, n_cam * n_points, n_feats)
    all_r_quant = all_r_quant.reshape(B, n_cam * n_points, n_feats)
    valid = valid.reshape(B, n_cam * n_points)
    all_bevs = []
    for batch_idx in range(B):
        curr_feats = all_feats[batch_idx]
        curr_bev_idx = all_r_quant[batch_idx]
        curr_valid = valid[batch_idx]
        bev = scatter_feautres_to_bev(
            curr_feats[curr_valid], curr_bev_idx[curr_valid], bev_cfg
        )
        all_bevs.append(bev)
    all_bevs = torch.stack(all_bevs, 0)

    cell_size = bev_cfg["cell_size"]
    W = int((bev_cfg["x_max"] - bev_cfg["x_min"]) / cell_size)
    H = int((bev_cfg["y_max"] - bev_cfg["y_min"]) / cell_size)

    assert all_bevs.shape == (B, H, W, all_feats.shape[-1])