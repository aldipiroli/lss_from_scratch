import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config

    def forward(self, x):
        return x


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.time_proj = nn.Linear(t_dim, out_ch)

    def forward(self, x, time_embed):
        h = self.conv1(F.relu(self.norm1(x)))
        time_embed = self.time_proj(time_embed).unsqueeze(-1).unsqueeze(-1)
        h = h + time_embed
        h = self.conv2(F.relu(self.norm2(h)))
        out = h + self.skip(x)
        return out


class DownsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class SelfAttention2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.Q = nn.Linear(dim, dim)
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        z = x.view(B, C, H * W).permute(0, 2, 1)
        q = self.Q(z)
        k = self.K(z)
        v = self.V(z)

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / (self.dim**0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = self.proj(out)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        out = x + out
        return out


class UNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_ch = config["MODEL"]["in_ch"]
        base_ch = config["MODEL"]["base_ch"]
        ch_scalings = config["MODEL"]["ch_scalings"]
        group_norm_ch = config["MODEL"]["group_norm_ch"]
        t_dim = config["MODEL"]["t_dim"]

        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        ch = base_ch
        for i, scaling in enumerate(ch_scalings):
            out_ch = base_ch * scaling
            block1 = ResBlock(ch, out_ch, group_norm_ch, t_dim)
            block2 = ResBlock(out_ch, out_ch, group_norm_ch, t_dim)
            attn = SelfAttention2D(out_ch) if scaling == 2 else nn.Identity()
            down = (
                DownsampleBlock(out_ch) if i != len(ch_scalings) - 1 else nn.Identity()
            )
            self.down_blocks.append(nn.ModuleList([block1, block2, attn, down]))
            ch = out_ch

        self.bottleneck = nn.ModuleList(
            [
                ResBlock(ch, ch, group_norm_ch, t_dim),
                SelfAttention2D(ch),
                ResBlock(ch, ch, group_norm_ch, t_dim),
            ]
        )
        self.ch = ch

    def forward(self, x, t_embed):
        x = self.init_conv(x)
        skips = []
        for block1, block2, attn, down in self.down_blocks:
            x = block1(x, t_embed)
            x = block2(x, t_embed)
            x = attn(x)
            skips.append(x)
            x = down(x)

        for layer in self.bottleneck:
            if layer._get_name() == "ResBlock":
                x = layer(x, t_embed)
            else:
                x = layer(x)
        return x, skips


class UNetDecoder(nn.Module):
    def __init__(self, config, enc_ch):
        super().__init__()
        in_ch = config["MODEL"]["in_ch"]
        base_ch = config["MODEL"]["base_ch"]
        ch_scalings = config["MODEL"]["ch_scalings"]
        group_norm_ch = config["MODEL"]["group_norm_ch"]
        t_dim = config["MODEL"]["t_dim"]

        self.ups = nn.ModuleList()
        ch = enc_ch
        for i, mult in reversed(list(enumerate(ch_scalings))):
            out_ch = base_ch * mult
            block1 = ResBlock(ch + out_ch, out_ch, group_norm_ch, t_dim)
            block2 = ResBlock(out_ch, out_ch, group_norm_ch, t_dim)
            attn = SelfAttention2D(out_ch) if mult == 2 else nn.Identity()
            up = UpsampleBlock(out_ch) if i != 0 else nn.Identity()
            self.ups.append(nn.ModuleList([block1, block2, attn, up]))
            ch = out_ch
        self.final = nn.Sequential(
            nn.GroupNorm(group_norm_ch, out_ch), nn.Conv2d(ch, in_ch, 1)
        )

    def forward(self, x, skips, t_embed):
        for block1, block2, attn, up in self.ups:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block1(x, t_embed)
            x = block2(x, t_embed)
            x = attn(x)
            x = up(x)
        x = self.final(x)
        return x


class TimeClassEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        t_dim = config["MODEL"]["t_dim"]
        self.register_buffer(
            "frequencies",
            torch.tensor([1 / (10_000 ** (2 * i / t_dim)) for i in range(t_dim // 2)]),
        )
        self.mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4), nn.ReLU(), nn.Linear(t_dim * 4, t_dim)
        )

    def forward(self, time):
        freq = self.frequencies
        sin = torch.sin(time * freq)
        cos = torch.cos(time * freq)
        time_enc = torch.cat([sin, cos], -1)
        time_enc = self.mlp(time_enc)
        return time_enc


class UNet(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = UNetEncoder(config)
        self.decoder = UNetDecoder(config, self.encoder.ch)
        self.time_encoder = TimeClassEncoder(config)
        self.class_emb = nn.Embedding(
            config["MODEL"]["num_classes"] + 1, config["MODEL"]["t_dim"]
        )
        self.config = config

    def forward(self, x, t, c):
        assert c.all() >= 0 and c.all() <= self.config["MODEL"]["num_classes"] + 1
        t_embed = self.time_encoder(t)
        c_embed = self.class_emb(c)
        tc_embed = t_embed + c_embed
        x, skips = self.encoder(x, tc_embed)
        x = self.decoder(x, skips, tc_embed)
        return x
