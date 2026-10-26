from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class LEAPConfig:
    k: int = 10
    lambda_x: float = 0.5
    eps: float = 1e-08
    r_audit: float = 0.089
    lambda_q: float = 0.3
    lambda_v: float = 0.2
    eta_plus: float = 0.1
    gamma_u: float = 0.5
    rho: float = 0.2
    rho_schedule: tuple[float, ...] | None = None
    fusion: str = 'conservative'
    hidden: int = 256
    n_layers: int = 2
    mlp_hidden: int = 64
    lambda1: float = 0.5
    lambda2: float = 0.001
    lambda3: float = 0.001
    T0: int = 50
    T: int = 200
    lr: float = 0.005
    weight_decay: float = 0.0005
    patience: int = 50
    seed: int = 0
    device: str = 'cpu'
    high_beta_threshold: float = 0.9

    def rho_at(self, layer: int) -> float:
        if self.rho_schedule is not None:
            return float(self.rho_schedule[min(layer, len(self.rho_schedule) - 1)])
        return float(self.rho)
