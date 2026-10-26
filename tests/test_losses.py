import torch
from leap.losses import classification_loss, confidence_loss, prior_loss, stability_loss

def test_confidence_loss_defined_at_zero_audits():
    m = torch.zeros(10, dtype=torch.bool)
    out = confidence_loss(torch.rand(10), torch.rand(10), torch.rand(10), torch.rand(10), m)
    assert torch.isfinite(out) and float(out) == 0.0, 'unit offset keeps the term defined'

def test_confidence_loss_manual_value():
    c_in, c_out = (torch.tensor([0.9]), torch.tensor([0.1]))
    alpha, beta = (torch.tensor([0.5]), torch.tensor([0.25]))
    m = torch.tensor([True])
    expected = (0.75 * 0.8 ** 2 + 0.25 * 0.5 ** 2) / 2.0
    assert abs(float(confidence_loss(c_in, c_out, alpha, beta, m)) - expected) < 1e-06

def test_confidence_loss_penalizes_confident_conflict():
    m = torch.tensor([True])
    c_in, c_out = (torch.tensor([0.9]), torch.tensor([0.1]))
    beta = torch.tensor([0.9])
    high = confidence_loss(c_in, c_out, torch.tensor([0.9]), beta, m)
    low = confidence_loss(c_in, c_out, torch.tensor([0.05]), beta, m)
    assert float(high) > float(low), 'high confidence under conflict costs more'

def test_prior_loss_anchors_sources():
    alpha = torch.tensor([1.0, 0.0, 0.2, 0.9])
    is_obs = torch.tensor([True, False, True, False])
    val = float(prior_loss(alpha, is_obs))
    expected = ((1 - 1) ** 2 + 0 ** 2 + (0.2 - 1) ** 2 + 0.9 ** 2) / 4
    assert abs(val - expected) < 1e-06
    assert float(prior_loss(torch.tensor([0.0]), torch.tensor([True]))) == 1.0

def test_stability_loss_is_mean_squared_norm():
    h = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    assert abs(float(stability_loss(h)) - 12.5) < 1e-06

def test_classification_loss_masks_nodes():
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0]])
    y = torch.tensor([0, 1, 1])
    mask = torch.tensor([True, True, False])
    assert float(classification_loss(logits, y, mask)) < 0.001
