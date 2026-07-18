import torch
from lss.utils.plotters import visualize_pcl


def get_f(K):
    # get focal length
    assert K.shape == (3, 3)
    f_x = K[0, 0]
    f_y = K[1, 1]
    return f_x, f_y


def get_p(K):
    # get principal point
    assert K.shape == (3, 3)
    p_x = K[0, 2]
    p_y = K[1, 2]
    return p_x, p_y


def pixel_to_camera_rays(img, K):
    # useful: https://hedivision.github.io/Pinhole.html
    device = img.device
    H, W = img.shape[-2:]
    grid_v, grid_u = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    u = grid_u.reshape(-1)
    v = grid_v.reshape(-1)

    p_x, p_y = get_p(K)
    f_x, f_y = get_f(K)

    r_x = (u - p_x) / f_x
    r_y = (v - p_y) / f_y
    r = torch.stack([r_x, r_y, torch.ones_like(r_x)], dim=-1)
    feats = img[..., v, u].transpose(1, 0)  # N,C
    return r, feats


def add_depth_along_ray(r, depth_config):
    d_min = depth_config["d_min"]
    d_max = depth_config["d_max"]
    d_step = depth_config["d_step"]
    depths = torch.arange(d_min, d_max, d_step)
    r = r[:, None, :] * depths[None, :, None]
    return r, depths


def camera_to_ego(r, extrinsic):
    R = extrinsic[:3, :3]
    t = extrinsic[:-1, -1]
    r = r @ R.T + t
    return r


def plot_camera_rays(r, img, u, v):
    color = img[..., u, v].transpose(1, 0)
    visualize_pcl(r, color)
