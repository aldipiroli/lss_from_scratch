import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from lss.utils.bev_utils import get_depths


class BaseModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config


class ResNet18(BaseModel):
    def __init__(self, config=None):
        super().__init__(config)
        weights = (
            ResNet18_Weights.DEFAULT if self.config["MODEL"]["pretrained"] else None
        )
        self.model = resnet18(weights=weights)

        depths = get_depths(config["LSS"]["depth_config"])
        self.dist_conv = nn.Conv2d(
            in_channels=128,
            out_channels=len(depths),
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        feats = self.model.layer2(x)  # (B, 128, 12, 20)
        dist = self.dist_conv(feats)  # (B, n_depths, 12, 20)
        return feats, dist
