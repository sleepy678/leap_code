import pytest
import torch
from leap.fusion import STRATEGIES, fuse, transmission_weight

def _simplex(n):
    return torch.softmax(torch.randn(n, 3) * 2, dim=1)

def _stats(tau):
    from leap.internal import triplet_stats
    return triplet_stats(tau)

def test_agreement_keeps_beta_small_conflict_raises_it():
    tau_agree = torch.tensor([[0.9, 0.08, 0.02]])
    tau_conflict = torch.tensor([[0.02, 0.08, 0.9]])
    c_a, u_a = _stats(tau_agree)
    c_c, u_c = _stats(tau_conflict)
    m = torch.tensor([True])
    _, beta_agree = fuse(c_a, u_a, tau_agree, c_a, u_a, tau_agree, m)
    _, beta_conflict = fuse(c_a, u_a, tau_agree, c_c, u_c, tau_conflict, m)
    assert float(beta_agree) < 0.2
    assert float(beta_conflict) > 0.8
    assert float(beta_conflict) > float(beta_agree)

def test_unaudited_branch():
    tau = _simplex(10)
    c, u = _stats(tau)
    m = torch.zeros(10, dtype=torch.bool)
    alpha, beta = fuse(c, u, tau, c * 0, u * 0, tau, m, gamma_u=0.5)
    assert torch.allclose(alpha, c)
    assert torch.allclose(beta, 0.5 * u)

def test_prop1_i_alpha_below_weighted_mean_and_monotone_in_beta():
    torch.manual_seed(1)
    tau_in, tau_out = (_simplex(500), _simplex(500))
    c_in, u_in = _stats(tau_in)
    c_out, u_out = _stats(tau_out)
    m = torch.ones(500, dtype=torch.bool)
    alpha, beta = fuse(c_in, u_in, tau_in, c_out, u_out, tau_out, m)
    w_in, w_out = (1 - u_in, 1 - u_out)
    c_bar = (w_in * c_in + w_out * c_out) / (w_in + w_out).clamp(min=1e-08)
    assert (alpha <= c_bar + 1e-06).all()
    assert (alpha >= torch.minimum(c_in, c_out) - 1e-06).all()
    for b_lo, b_hi in [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]:
        mn = torch.minimum(c_in, c_out)
        a_lo = (1 - b_lo) * c_bar + b_lo * mn
        a_hi = (1 - b_hi) * c_bar + b_hi * mn
        assert (a_hi <= a_lo + 1e-06).all()

def test_prop1_ii_transmission_bounded_and_monotone():
    alpha = torch.rand(200) * 0.98 + 0.01
    beta = torch.rand(200) * 0.98 + 0.01
    for rho in (0.0, 0.2, 0.9):
        w = transmission_weight(alpha, beta, rho)
        assert (w <= alpha + 1e-07).all()
        assert (w < alpha).all(), 'strict at nonzero alpha, beta and rho<1'
        w_more_conflict = transmission_weight(alpha, (beta + 0.2).clamp(max=1.0), rho)
        assert (w_more_conflict <= w + 1e-07).all()
    assert torch.allclose(transmission_weight(alpha, torch.zeros_like(beta), 0.3), alpha)
    assert torch.allclose(transmission_weight(alpha, beta, 1.0), alpha)
    zero = torch.zeros(5)
    assert torch.allclose(transmission_weight(zero, beta[:5], 0.3), zero)

def test_prop1_iii_nonincreasing_rho_schedule():
    alpha, beta = (torch.tensor([0.8]), torch.tensor([0.6]))
    schedule = [1.0, 0.5, 0.2, 0.2]
    ws = [transmission_weight(alpha, beta, r) for r in schedule]
    for earlier, later in zip(ws, ws[1:]):
        assert float(later) <= float(earlier) + 1e-09, 'contested edges never regain influence'

def test_alternative_strategies_match_appendix_i():
    tau_in, tau_out = (_simplex(50), _simplex(50))
    c_in, u_in = _stats(tau_in)
    c_out, u_out = _stats(tau_out)
    m = torch.ones(50, dtype=torch.bool)
    a_wavg, b_wavg = fuse(c_in, u_in, tau_in, c_out, u_out, tau_out, m, strategy='weighted_average')
    assert torch.allclose(b_wavg, torch.zeros(50))
    a_max, b_max = fuse(c_in, u_in, tau_in, c_out, u_out, tau_out, m, strategy='max_confidence')
    assert torch.allclose(a_max, torch.maximum(c_in, c_out))
    a_int, b_int = fuse(c_in, u_in, tau_in, c_out, u_out, tau_out, m, strategy='internal_only', gamma_u=0.5)
    assert torch.allclose(a_int, c_in) and torch.allclose(b_int, 0.5 * u_in)
    with pytest.raises(ValueError):
        fuse(c_in, u_in, tau_in, c_out, u_out, tau_out, m, strategy='nonsense')
    assert set(STRATEGIES) == {'conservative', 'weighted_average', 'max_confidence', 'internal_only'}

def test_audit_never_hard_edits():
    tau_in, tau_out = (_simplex(300), _simplex(300))
    c_in, u_in = _stats(tau_in)
    c_out, u_out = _stats(tau_out)
    m = torch.rand(300) > 0.5
    alpha, beta = fuse(c_in, u_in, tau_in, c_out, u_out, tau_out, m)
    assert float(alpha.min()) >= 0.0 and float(alpha.max()) <= 1.0
    assert float(beta.min()) >= 0.0 and float(beta.max()) <= 1.0
