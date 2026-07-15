import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self, config, logger):
        super(BaseLoss, self).__init__()
        self.config = config
        self.logger = logger

    def forward(self, preds, labels):
        pass


class CFGLoss(BaseLoss):
    def __init__(self, config, logger):
        super(CFGLoss, self).__init__(config, logger)

    def forward(self, preds, labels):
        loss = torch.nn.functional.mse_loss(preds, labels)
        loss_dict = {"lss_loss": loss}
        return loss, loss_dict
