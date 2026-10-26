import numpy as np
import torch
from leap.config import LEAPConfig
from leap.data import canonical_edges
from leap.model import DualPathGCN, sym_normalized_coefficients, weighted_aggregate

def _toy():
    edges = torch.tensor(canonical_edges(np.array([[0, 1], [1, 2], [2, 3], [0, 3]])))
    x = torch.randn(4, 6)
    return (x, edges)

def test_sym_normalization_matches_dense_formula():
    x, edges = _toy()
    w = torch.tensor([0.5, 1.0, 0.25, 0.75])
    src, dst, coef = sym_normalized_coefficients(edges, w, 4)
    dense = torch.zeros(4, 4)
    dense[edges[:, 0], edges[:, 1]] = w
    dense = dense + dense.T
    deg = dense.sum(dim=1)
    expected = torch.diag(deg.pow(-0.5)) @ dense @ torch.diag(deg.pow(-0.5))
    got = torch.zeros(4, 4)
    got[dst, src] = coef
    assert torch.allclose(got, expected, atol=1e-06)
    assert torch.allclose(got, got.T, atol=1e-06), 'symmetrized'

def test_zero_weight_edges_transmit_nothing():
    x, edges = _toy()
    w = torch.ones(4)
    row_12 = [r for r, (i, j) in enumerate(edges.tolist()) if (i, j) == (1, 2)][0]
    w[row_12] = 0.0
    src, dst, coef = sym_normalized_coefficients(edges, w, 4)
    h = torch.eye(4)
    agg = weighted_aggregate(h, src, dst, coef, 4)
    assert float(agg[1, 2]) == 0.0 and float(agg[2, 1]) == 0.0

def test_isolated_node_is_degree_safe():
    edges = torch.tensor([[0, 1]])
    w = torch.ones(1)
    src, dst, coef = sym_normalized_coefficients(edges, w, 3)
    assert torch.isfinite(coef).all()

def test_forward_shapes_and_grads():
    x, edges = _toy()
    model = DualPathGCN(6, 8, 3, n_layers=2)
    alpha = torch.rand(4, requires_grad=True)
    beta = torch.rand(4, requires_grad=True)
    out = model(x, edges, alpha, beta)
    assert out.shape == (4, 3)
    out.sum().backward()
    assert alpha.grad is not None and torch.isfinite(alpha.grad).all()
    assert beta.grad is not None, 'propagation is differentiable in the edge states'

def test_beta_zero_disables_conservative_path():
    x, edges = _toy()
    cfg_a = LEAPConfig(rho=0.9)
    cfg_b = LEAPConfig(rho=0.0)
    torch.manual_seed(3)
    model_a = DualPathGCN(6, 8, 3, config=cfg_a)
    torch.manual_seed(3)
    model_b = DualPathGCN(6, 8, 3, config=cfg_b)
    alpha = torch.rand(4)
    beta = torch.zeros(4)
    assert torch.allclose(model_a(x, edges, alpha, beta), model_b(x, edges, alpha, beta), atol=1e-06)

def test_rho_zero_blocks_contested_edges():
    x, edges = _toy()
    cfg = LEAPConfig(rho=0.0)
    model = DualPathGCN(6, 8, 3, config=cfg)
    alpha = torch.ones(4)
    out_contested = model(x, edges, alpha, torch.ones(4))
    out_selfonly = model(x, edges, torch.zeros(4), torch.zeros(4))
    assert torch.allclose(out_contested, out_selfonly, atol=1e-05)

def test_uniform_alpha_scale_cancels_under_normalization():
    x, edges = _toy()
    w = torch.rand(4) + 0.1
    _, _, coef = sym_normalized_coefficients(edges, w, 4)
    _, _, coef_scaled = sym_normalized_coefficients(edges, 0.01 * w, 4)
    assert torch.allclose(coef, coef_scaled, atol=1e-06)

def test_conflict_on_one_edge_shrinks_its_own_coefficient():
    x, edges = _toy()
    alpha = torch.full((4,), 0.8)
    coefs = []
    for bv in (0.0, 0.5, 0.99):
        beta = torch.zeros(4)
        beta[0] = bv
        _, _, coef = sym_normalized_coefficients(edges, alpha * (1.0 - beta), 4)
        coefs.append(float(coef[0]))
    assert coefs[0] > coefs[1] > coefs[2]

def test_rho_schedule_is_per_layer():
    x, edges = _toy()
    cfg = LEAPConfig(rho_schedule=(1.0, 0.0))
    model = DualPathGCN(6, 8, 3, config=cfg)
    assert model.config.rho_at(0) == 1.0
    assert model.config.rho_at(1) == 0.0
    assert model.config.rho_at(5) == 0.0
    out = model(x, edges, torch.rand(4), torch.rand(4))
    assert torch.isfinite(out).all()
