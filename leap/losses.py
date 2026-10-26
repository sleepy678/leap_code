from __future__ import annotations
import torch
import torch.nn.functional as F

def classification_loss(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[mask], y[mask])

def confidence_loss(c_in: torch.Tensor, c_out: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    v = valid.float()
    per_edge = (1.0 - beta) * (c_in - c_out) ** 2 + beta * alpha ** 2
    return (v * per_edge).sum() / (v.sum() + 1.0)

def prior_loss(alpha: torch.Tensor, is_observed: torch.Tensor) -> torch.Tensor:
    return ((alpha - is_observed.float()) ** 2).mean()

def stability_loss(h_final: torch.Tensor) -> torch.Tensor:
    return h_final.pow(2).sum(dim=1).mean()
