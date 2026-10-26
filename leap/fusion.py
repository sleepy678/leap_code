from __future__ import annotations
import torch
STRATEGIES = ('conservative', 'weighted_average', 'max_confidence', 'internal_only')

def fuse(c_in: torch.Tensor, u_in: torch.Tensor, tau_in: torch.Tensor, c_out: torch.Tensor, u_out: torch.Tensor, tau_out: torch.Tensor, m: torch.Tensor, gamma_u: float=0.5, strategy: str='conservative', eps: float=1e-08):
    if strategy not in STRATEGIES:
        raise ValueError(f'unknown fusion strategy {strategy!r}; pick one of {STRATEGIES}')
    m_f = m.float()
    alpha0 = c_in
    beta0 = gamma_u * u_in
    if strategy == 'internal_only':
        return (alpha0, beta0)
    w_in, w_out = (1.0 - u_in, 1.0 - u_out)
    denom = w_in + w_out
    c_bar = torch.where(denom > eps, (w_in * c_in + w_out * c_out) / denom.clamp(min=eps), 0.5 * (c_in + c_out))
    if strategy == 'conservative':
        a_c = 1.0 - (c_in - c_out).abs()
        a_tau = (tau_in * tau_out).sum(dim=-1)
        beta1 = 1.0 - 0.5 * (a_c + a_tau)
        alpha1 = (1.0 - beta1) * c_bar + beta1 * torch.minimum(c_in, c_out)
    elif strategy == 'weighted_average':
        alpha1, beta1 = (c_bar, torch.zeros_like(c_bar))
    else:
        alpha1, beta1 = (torch.maximum(c_in, c_out), torch.zeros_like(c_bar))
    alpha = m_f * alpha1 + (1.0 - m_f) * alpha0
    beta = m_f * beta1 + (1.0 - m_f) * beta0
    return (alpha, beta)

def transmission_weight(alpha: torch.Tensor, beta: torch.Tensor, rho: float) -> torch.Tensor:
    return alpha * (1.0 - beta * (1.0 - rho))
