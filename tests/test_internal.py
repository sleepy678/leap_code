import math
import torch
from leap.internal import InternalScorer, triplet_stats
from leap.records import NEU, RISK, SUP

def test_scorer_outputs_simplex():
    scorer = InternalScorer()
    with torch.no_grad():
        tau = scorer(torch.randn(50, 5))
    assert tau.shape == (50, 3)
    assert torch.allclose(tau.sum(dim=1), torch.ones(50), atol=1e-05)
    assert float(tau.min()) >= 0.0

def test_triplet_stats_uniform():
    tau = torch.full((1, 3), 1.0 / 3.0)
    c, u = triplet_stats(tau)
    assert abs(float(c) - 0.5) < 1e-06
    assert abs(float(u) - 1.0) < 1e-06

def test_triplet_stats_pure_categories():
    eye = torch.eye(3)
    c, u = triplet_stats(eye)
    assert abs(float(c[SUP]) - 1.0) < 1e-06
    assert abs(float(c[NEU]) - 0.5) < 1e-06
    assert abs(float(c[RISK]) - 0.0) < 1e-06
    assert torch.allclose(u, torch.zeros(3), atol=1e-05)

def test_uncertainty_bounds():
    tau = torch.softmax(torch.randn(200, 3) * 3, dim=1)
    _, u = triplet_stats(tau)
    assert float(u.min()) >= 0.0 and float(u.max()) <= 1.0
    assert float(u.max()) <= 1.0 + 1e-06
    assert math.isclose(float(triplet_stats(torch.full((1, 3), 1 / 3))[1]), 1.0, abs_tol=1e-06)
