import math
import torch
from leap.audit import AuditCache, MockTopicAuditor
from leap.config import LEAPConfig
from leap.data import perturb_random_add, perturb_random_delete, synthetic_tag
from leap.train import train_gcn_baseline, train_leap
FAST = LEAPConfig(k=3, hidden=32, T0=5, T=40, patience=40, r_audit=0.089, seed=0)

def _perturbed(seed=0):
    data = synthetic_tag(n=200, num_classes=4, d=16, avg_degree=6.0, seed=seed)
    return perturb_random_add(data, rate=0.3, seed=seed)

def test_perturbations_change_edge_counts():
    data = synthetic_tag(n=100, num_classes=4, d=8, seed=1)
    m = data.num_edges
    assert perturb_random_add(data, 0.2, seed=1).num_edges == m + math.ceil(0.2 * m)
    assert perturb_random_delete(data, 0.5, seed=1).num_edges == m - math.ceil(0.5 * m)

def test_leap_full_pipeline_learns(tmp_path):
    data = _perturbed()
    auditor = MockTopicAuditor()
    cache_path = str(tmp_path / 'cache.json')
    result = train_leap(data, FAST, auditor=auditor, cache=cache_path)
    assert result.test_acc > 0.6, f'well above 4-class chance, got {result.test_acc}'
    assert result.audit_stats['queried'] > 0
    assert result.audit_stats['failed'] == 0
    rec = result.records
    assert rec.num_edges >= data.num_edges
    assert bool(rec.m.any()), 'some edges carry valid audits'
    assert float(rec.alpha.min()) >= 0.0 and float(rec.alpha.max()) <= 1.0
    assert float(rec.beta.min()) >= 0.0 and float(rec.beta.max()) <= 1.0
    assert rec.z_in.shape == (rec.num_edges, 5)
    diag = result.diagnostics
    assert 0.0 <= diag['coverage_observed'] <= 1.0
    assert 'internal_llm_agreement' in diag
    assert abs(sum(diag['verdict_shares'].values()) - 1.0) < 1e-05

def test_leap_int_runs_without_auditor():
    result = train_leap(_perturbed(), FAST, auditor=None)
    assert result.test_acc > 0.5
    assert result.audit_stats['queried'] == 0
    assert not bool(result.records.m.any())

def test_cache_reused_across_runs(tmp_path):
    data = _perturbed()
    cache_path = str(tmp_path / 'cache.json')
    auditor = MockTopicAuditor()
    train_leap(data, FAST, auditor=auditor, cache=cache_path)
    first_calls = auditor.call_count
    assert first_calls > 0
    r2 = train_leap(data, FAST, auditor=auditor, cache=cache_path)
    assert auditor.call_count == first_calls, 'second run fully served from cache'
    assert r2.audit_stats['cache_hits'] == r2.audit_stats['queried']

def test_gcn_baseline_runs():
    result = train_gcn_baseline(_perturbed(), FAST)
    assert result.test_acc > 0.4

def test_record_dump_is_inspectable():
    result = train_leap(_perturbed(), FAST, auditor=MockTopicAuditor())
    idx = int(result.records.m.float().argmax())
    d = result.records.as_dict(idx)
    assert set(d) == {'edge', 'source', 'z_in', 'tau_in', 'c_in', 'u_in', 'm', 'tau_out', 'c_out', 'u_out', 'alpha', 'beta'}
    assert d['m'] is True

def test_deterministic_given_seed():
    r1 = train_leap(_perturbed(), FAST, auditor=MockTopicAuditor())
    r2 = train_leap(_perturbed(), FAST, auditor=MockTopicAuditor())
    assert r1.test_acc == r2.test_acc
    assert torch.allclose(r1.records.alpha, r2.records.alpha)
