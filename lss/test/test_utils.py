import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch

from lss.utils.utils import (
    compute_alpha_bar_t,
    denoise_image_step,
    get_noise_scheduler,
    get_noisy_image,
    precompute_alpha_t,
    drop_class_condition,
)
from lss.utils.misc import load_config


def test_get_noise_schedule():
    config = load_config("lss/config/mnist_config.yaml")
    scheadule = get_noise_scheduler(config)
    T = config["MODEL"]["T"]
    beta_1 = config["MODEL"]["beta_1"]
    beta_T = config["MODEL"]["beta_T"]
    assert len(scheadule) == T
    assert scheadule.min() == beta_1
    assert scheadule.max() == beta_T


def test_compute_alpha_bar_t():
    config = load_config("lss/config/mnist_config.yaml")
    scheadule = get_noise_scheduler(config)
    t = 10
    alpha_bar = compute_alpha_bar_t(scheadule, t)
    assert alpha_bar > 0


def test_precompute_alpha_t():
    config = load_config("lss/config/mnist_config.yaml")
    all_alpha_bar = precompute_alpha_t(config)
    assert len(all_alpha_bar) == config["MODEL"]["T"]

    for i in range(len(all_alpha_bar) - 1):
        assert all_alpha_bar[i + 1] < all_alpha_bar[i]


def test_get_noisy_image():
    config = load_config("lss/config/mnist_config.yaml")
    B, C, H, W = 2, 3, 128, 128
    img = torch.randn(B, C, H, W)
    all_alpha_bar = precompute_alpha_t(config)
    img_noisy, eps, t = get_noisy_image(img, all_alpha_bar)
    assert img_noisy.shape == img.shape
    assert eps.shape == img.shape


def test_denoise_image_step():
    config = load_config("lss/config/mnist_config.yaml")
    B, C, H, W = 2, 3, 128, 128
    x_t = torch.randn(B, C, H, W)
    pred_eps = torch.randn(B, C, H, W)
    pred_eps_cond = torch.randn(B, C, H, W)
    t = 10
    weight = 0.1
    scheaduler = get_noise_scheduler(config)
    all_alpha_bar = precompute_alpha_t(config)
    xt_m1 = denoise_image_step(
        x_t, pred_eps, pred_eps_cond, t, scheaduler, all_alpha_bar, weight
    )
    assert xt_m1.shape == x_t.shape


def test_drop_class_condition():
    config = load_config("lss/config/mnist_config.yaml")
    B = 32
    labels = torch.randn(B)
    labels_dropped = drop_class_condition(labels, config)
    assert (labels_dropped == (config["MODEL"]["num_classes"])).sum() > 0
