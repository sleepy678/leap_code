from __future__ import annotations
from dataclasses import dataclass
import torch
SUP, NEU, RISK = (0, 1, 2)
CATEGORY_NAMES = ('supportive', 'neutral', 'risky')

@dataclass
class EdgeRecords:
    edges: torch.Tensor
    is_observed: torch.Tensor
    z_in: torch.Tensor
    tau_in: torch.Tensor
    c_in: torch.Tensor
    u_in: torch.Tensor
    m: torch.Tensor
    tau_out: torch.Tensor
    c_out: torch.Tensor
    u_out: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor

    @classmethod
    def empty(cls, edges: torch.Tensor, is_observed: torch.Tensor) -> 'EdgeRecords':
        m = edges.shape[0]
        dev = edges.device
        tau_out = torch.zeros(m, 3, device=dev)
        tau_out[:, NEU] = 1.0
        return cls(edges=edges, is_observed=is_observed, z_in=torch.zeros(m, 5, device=dev), tau_in=torch.full((m, 3), 1.0 / 3.0, device=dev), c_in=torch.full((m,), 0.5, device=dev), u_in=torch.ones(m, device=dev), m=torch.zeros(m, dtype=torch.bool, device=dev), tau_out=tau_out, c_out=torch.full((m,), 0.5, device=dev), u_out=torch.zeros(m, device=dev), alpha=torch.full((m,), 0.5, device=dev), beta=torch.zeros(m, device=dev))

    @property
    def num_edges(self) -> int:
        return self.edges.shape[0]

    def as_dict(self, idx: int) -> dict:
        return {'edge': tuple((int(v) for v in self.edges[idx])), 'source': 'observed' if bool(self.is_observed[idx]) else 'retrieved', 'z_in': [round(float(v), 4) for v in self.z_in[idx]], 'tau_in': [round(float(v), 4) for v in self.tau_in[idx]], 'c_in': round(float(self.c_in[idx]), 4), 'u_in': round(float(self.u_in[idx]), 4), 'm': bool(self.m[idx]), 'tau_out': [round(float(v), 4) for v in self.tau_out[idx]], 'c_out': round(float(self.c_out[idx]), 4), 'u_out': round(float(self.u_out[idx]), 4), 'alpha': round(float(self.alpha[idx]), 4), 'beta': round(float(self.beta[idx]), 4)}
