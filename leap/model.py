from __future__ import annotations
import torch
import torch.nn as nn
from leap.config import LEAPConfig

def _directed(edges: torch.Tensor, w: torch.Tensor):
    src = torch.cat([edges[:, 0], edges[:, 1]])
    dst = torch.cat([edges[:, 1], edges[:, 0]])
    return (src, dst, torch.cat([w, w]))

def sym_normalized_coefficients(edges: torch.Tensor, w: torch.Tensor, n: int, eps: float=1e-08):
    src, dst, w2 = _directed(edges, w)
    deg = torch.zeros(n, dtype=w.dtype, device=w.device).index_add_(0, dst, w2)
    inv_sqrt = deg.clamp(min=eps).pow(-0.5) * (deg > eps).float()
    return (src, dst, w2 * inv_sqrt[src] * inv_sqrt[dst])

def weighted_aggregate(h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, coef: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros(n, h.shape[1], dtype=h.dtype, device=h.device)
    return out.index_add(0, dst, coef.unsqueeze(1) * h[src])

class DualPathGCN(nn.Module):

    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_layers: int=2, config: LEAPConfig | None=None):
        super().__init__()
        self.config = config or LEAPConfig()
        self.n_layers = n_layers
        dims = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
        self.w_rel = nn.ModuleList((nn.Linear(dims[l], dims[l + 1], bias=False) for l in range(n_layers)))
        self.w_con = nn.ModuleList((nn.Linear(dims[l], dims[l + 1], bias=False) for l in range(n_layers)))
        self.w_self = nn.ModuleList((nn.Linear(dims[l], dims[l + 1], bias=True) for l in range(n_layers)))

    def forward(self, x: torch.Tensor, edges: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        src_r, dst_r, coef_rel = sym_normalized_coefficients(edges, alpha * (1.0 - beta), n)
        src_c, dst_c, coef_con = sym_normalized_coefficients(edges, alpha * beta, n)
        h = x
        for layer in range(self.n_layers):
            m_rel = weighted_aggregate(self.w_rel[layer](h), src_r, dst_r, coef_rel, n)
            m_con = weighted_aggregate(self.w_con[layer](h), src_c, dst_c, coef_con, n)
            h = m_rel + self.config.rho_at(layer) * m_con + self.w_self[layer](h)
            if layer < self.n_layers - 1:
                h = torch.relu(h)
        return h
