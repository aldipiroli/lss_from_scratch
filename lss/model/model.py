import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class BaseModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config

class ResNet18(BaseModel):
    def __init__(self, config=None):
        super().__init__(config)
        weights = ResNet18_Weights.DEFAULT if self.config["MODEL"]["pretrained"] else None
        self.model = resnet18(weights=weights)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x