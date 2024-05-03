import torch

from torch import nn
from torch import Tensor


class LocalizationLoss(nn.Module):
    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        rel_x_loss = self.mse_loss().sum(dim=2).mean(dim=(0,1))
        rel_y_loss = self.mse_loss().sum(dim=2).mean(dim=(0,1))
        rel_theta_loss = self.mse_loss().sum(dim=2).mean(dim=(0,1))

        rel_x_loss = self.mse_loss().sum(dim=2).mean(dim=(0,1))
        rel_y_loss = self.mse_loss().sum(dim=2).mean(dim=(0,1))
        rel_theta_loss = self.mse_loss().sum(dim=2).mean(dim=(0,1))
        R_diffs = input @ target.permute(0, 2, 1)
        # See: https://github.com/pytorch/pytorch/issues/7500#issuecomment-502122839.
        traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
        dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + self.eps, 1 - self.eps))
        if self.reduction == "none":
            return dists
        elif self.reduction == "mean":
            return dists.mean()
        elif self.reduction == "sum":
            return dists.sum()