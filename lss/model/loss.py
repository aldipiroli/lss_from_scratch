import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self, config, logger):
        super(BaseLoss, self).__init__()
        self.config = config
        self.logger = logger

    def forward(self, preds, labels):
        pass


class LSSLoss(BaseLoss):
    def __init__(self, config, logger):
        super(LSSLoss, self).__init__(config, logger)

    def get_matching_trajectory_idx(self, gt_trajectory, trajectory_lib):
        distances = torch.norm(
            trajectory_lib.unsqueeze(0) - gt_trajectory.unsqueeze(1), dim=-1
        ).mean(dim=-1)
        best_idx = distances.argmin(dim=1)
        return best_idx

    def forward(self, trajectory_costs, gt_trajectory, trajectory_lib):
        best_idx = self.get_matching_trajectory_idx(gt_trajectory, trajectory_lib)
        best_idx = best_idx.to(trajectory_costs.device)
        loss = torch.nn.functional.cross_entropy(-trajectory_costs, best_idx)
        loss_dict = {"lss_loss": loss}
        return loss, loss_dict
