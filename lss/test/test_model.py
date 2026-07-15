import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import pytest
import torch

from lss.model.model import (
    BaseModel,
    DownsampleBlock,
    ResBlock,
    SelfAttention2D,
    TimeClassEncoder,
    UNet,
    UNetDecoder,
    UNetEncoder,
    UpsampleBlock,
)
from lss.utils.misc import load_config


def test_base_model():
    config = load_config("lss/config/mnist_config.yaml")
    model = BaseModel(config)
    img_size = config["DATA"]["img_size"]
    B, C, H, W = 2, 3, img_size[0], img_size[1]
    img = torch.randn(B, C, H, W)
    out = model(img)
    assert out.shape == img.shape


@pytest.mark.parametrize(
    "B, C, H, W, out_channels",
    [
        (2, 32, 128, 128, 64),
        (2, 32, 128, 128, 32),
    ],
)
def test_res_block(B, C, H, W, out_channels):
    t_dim = 128
    t_embed = torch.rand(B, t_dim)
    x = torch.randn(B, C, H, W)
    model = ResBlock(C, out_channels, C // 2, t_dim)
    out = model(x, t_embed)
    assert out.shape == (B, out_channels, H, W)


def test_downsample_block():
    B, C, H, W = 2, 32, 128, 128
    x = torch.randn(B, C, H, W)
    model = DownsampleBlock(C)
    out = model(x)
    assert out.shape == (B, C, H // 2, W // 2)


def test_upsample_block():
    B, C, H, W = 2, 32, 128, 128
    x = torch.randn(B, C, H, W)
    model = UpsampleBlock(C)
    out = model(x)
    assert out.shape == (B, C, H * 2, W * 2)


def test_self_attention():
    B, C, H, W = 2, 32, 16, 16
    x = torch.randn(B, C, H, W)
    model = SelfAttention2D(C)
    out = model(x)
    assert out.shape == (B, C, H, W)


def test_unet_encoder():
    config = load_config("lss/config/mnist_config.yaml")
    model = UNetEncoder(config)
    B, C, H, W = 2, 1, 32, 32
    x = torch.randn(B, C, H, W)
    t_dim = config["MODEL"]["t_dim"]
    t_embed = torch.rand(B, t_dim)
    out, skips = model(x, t_embed)
    ch_scalings = config["MODEL"]["ch_scalings"]
    base_ch = config["MODEL"]["base_ch"]
    n_scalings = len(ch_scalings)
    n_down = 2 ** (n_scalings - 1)
    assert out.shape == (B, base_ch * ch_scalings[-1], H // n_down, W // n_down)
    assert len(skips) == n_scalings


def test_unet_decoder():
    config = load_config("lss/config/mnist_config.yaml")
    enc = UNetEncoder(config)
    dec = UNetDecoder(config, enc_ch=enc.ch)
    B, C, H, W = 2, 1, 32, 32
    x = torch.randn(B, C, H, W)
    t_dim = config["MODEL"]["t_dim"]
    t_embed = torch.rand(B, t_dim)
    out, skips = enc(x, t_embed)
    out = dec(out, skips, t_embed)
    assert out.shape == (B, C, H, W)


def test_unet():
    config = load_config("lss/config/mnist_config.yaml")
    model = UNet(config)
    B, C, H, W = 2, 1, 256, 256
    x = torch.randn(B, C, H, W)
    t = torch.tensor(10)
    c = torch.tensor(1)
    out = model(x, t, c)
    assert out.shape == (B, C, H, W)


def test_time_class_encoder():
    config = load_config("lss/config/mnist_config.yaml")
    time_encoder = TimeClassEncoder(config)
    t_dim = config["MODEL"]["t_dim"]
    T = config["MODEL"]["T"]
    B = 2
    time = torch.randint(0, T, (B, 1))
    enc_time = time_encoder(time)
    assert enc_time.shape == (B, t_dim)


if __name__ == "__main__":
    print("All tests passed!")
