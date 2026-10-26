from __future__ import annotations
import math
import torch
import torch.nn as nn
from leap.records import NEU, SUP

class InternalScorer(nn.Module):

    def __init__(self, in_dim: int=5, hidden: int=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(z), dim=-1)

def triplet_stats(tau: torch.Tensor, eps: float=1e-12):
    c = tau[..., SUP] + 0.5 * tau[..., NEU]
    u = -(tau * torch.log(tau.clamp(min=eps))).sum(dim=-1) / math.log(3.0)
    return (c, u.clamp(0.0, 1.0))
