import json
import math
import torch
from leap.audit import ABSTRACT_LIMIT, AuditCache, CosineAuditor, MockTopicAuditor, TITLE_LIMIT, audit_value, build_prompt, format_endpoint, parse_auditor_response, query_priority, run_external_audit, select_queries
from leap.records import EdgeRecords, NEU

def _records(edges, observed_mask):
    return EdgeRecords.empty(torch.tensor(edges), torch.tensor(observed_mask))

def test_budget_size_is_ceil():
    pi = torch.rand(100)
    assert select_queries(pi, 0.089).numel() == math.ceil(0.089 * 100)
    assert select_queries(pi, 0.0).numel() == 0
    assert select_queries(pi, 1.0).numel() == 100

def test_priority_orders_uncertain_edges_first():
    nu = torch.zeros(3)
    c = torch.tensor([0.95, 0.5, 0.5])
    u = torch.tensor([0.05, 0.9, 0.3])
    pi = query_priority(c, u, nu, lambda_q=0.3, lambda_v=0.2)
    order = torch.argsort(pi, descending=True).tolist()
    assert order[0] == 1
    assert order[-1] == 0

def test_audit_value_favors_low_degree_and_retrieved():
    edges = torch.tensor([[0, 1], [0, 2], [0, 3], [0, 4], [3, 4]])
    is_obs = torch.tensor([True, True, True, True, False])
    nu = audit_value(edges, is_obs, n=5, eta_plus=0.1)
    assert int(nu.argmax()) == 4
    hub_edges = nu[:4]
    assert float(nu[4]) > float(hub_edges.max())

def test_truncation_limits():
    text = 'T' * 1000 + '\n' + 'A' * 5000
    formatted = format_endpoint(text)
    title, _, abstract = formatted.partition(' | ')
    assert len(title) == TITLE_LIMIT
    assert len(abstract) == ABSTRACT_LIMIT
    prompt = build_prompt(text, text, 'retrieved')
    assert 'retrieved' in prompt and 'supportive' in prompt

def test_parser_valid_response():
    raw = json.dumps({'verdict': 'risky', 'scores': {'supportive': 0.1, 'neutral': 0.2, 'risky': 0.7}, 'rationale': 'off-topic'})
    out = parse_auditor_response(raw)
    assert out['ok'] and out['parser_status'] == 'ok'
    assert abs(out['tau_out'].sum() - 1.0) < 1e-09
    assert out['tau_out'][2] == 0.7

def test_parser_json_embedded_in_prose():
    raw = 'Sure! Here is my verdict:\n{"scores": {"supportive": 2, "neutral": 1, "risky": 1}}\nHope this helps.'
    out = parse_auditor_response(raw)
    assert out['ok']
    assert abs(out['tau_out'][0] - 0.5) < 1e-09

def test_parser_failures_fall_back_to_neutral():
    for raw in ['', 'no json here', '{"scores": {"supportive": "high"}}', '{"scores": {"supportive": -1, "neutral": -1, "risky": -1}}']:
        out = parse_auditor_response(raw)
        assert not out['ok']
        assert out['tau_out'][NEU] == 1.0

def test_mock_auditor_topic_logic():
    aud = MockTopicAuditor()
    same = parse_auditor_response(aud.audit_pair('A study of graph neural networks\nfoo', 'deep graph neural networks\nbar', 'observed'))
    diff = parse_auditor_response(aud.audit_pair('A study of graph neural networks\nfoo', 'notes on convex optimization\nbar', 'retrieved'))
    assert same['tau_out'][0] > 0.8 and diff['tau_out'][2] > 0.7

def test_cosine_auditor_returns_valid_simplex():
    aud = CosineAuditor()
    out = parse_auditor_response(aud.audit_pair('alpha beta gamma\nx', 'alpha beta gamma\ny', 'observed'))
    assert out['ok'] and out['tau_out'][0] == out['tau_out'].max()

def test_run_audit_fills_records_and_counts_failures():

    class Flaky(MockTopicAuditor):

        def audit_pair(self, ti, tj, source):
            self.call_count += 1
            if self.call_count == 1:
                return 'garbage ###'
            return super().audit_pair(ti, tj, source)
    texts = ['a about graph neural networks\nz', 'b about graph neural networks\nz', 'c about convex optimization\nz']
    rec = _records([[0, 1], [0, 2]], [True, False])
    stats = run_external_audit(rec, texts, torch.tensor([0, 1]), Flaky())
    assert stats['queried'] == 2 and stats['failed'] == 1
    assert not bool(rec.m[0]) and bool(rec.m[1])
    assert rec.tau_out[0, NEU] == 1.0
    assert float(rec.c_out[1]) < 0.5

def test_cache_prevents_requeries(tmp_path):
    texts = ['p graph neural networks\nz', 'q graph neural networks\nz']
    path = str(tmp_path / 'audit_cache.json')
    aud = MockTopicAuditor()
    rec1 = _records([[0, 1]], [True])
    s1 = run_external_audit(rec1, texts, torch.tensor([0]), aud, AuditCache(path))
    rec2 = _records([[0, 1]], [True])
    s2 = run_external_audit(rec2, texts, torch.tensor([0]), aud, AuditCache(path))
    assert aud.call_count == 1, 'second run answered from the cache'
    assert s1['cache_hits'] == 0 and s2['cache_hits'] == 1
    assert torch.allclose(rec1.tau_out, rec2.tau_out)
