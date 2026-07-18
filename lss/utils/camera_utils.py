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
    grid_u, grid_v = torch.meshgrid(
        torch.arange(img.shape[-2]), torch.arange(img.shape[-1]), indexing="ij"
    )
    u = grid_u.reshape(-1)
    v = grid_v.reshape(-1)

    p_x, p_y = get_p(K)
    f_x, f_y = get_f(K)

    r_x = (u - p_x) / f_x
    r_y = (v - p_y) / f_y
    r = torch.stack([r_x, r_y, torch.ones_like(r_x)], dim=-1)
    feats = img[..., u, v].transpose(1, 0)  # N,C
    return r, feats


def plot_camera_rays(r, img, u, v):
    color = img[..., u, v].transpose(1, 0)
    visualize_pcl(r, color)
