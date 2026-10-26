import numpy as np
import torch
from leap.candidate import adjacency_csr, build_candidate_set, retrieve_candidates
from leap.data import canonical_edges

def test_retrieved_edges_disjoint_from_observed(tiny):
    e_plus = retrieve_candidates(tiny.x, tiny.edges, k=3)
    observed = {(int(i), int(j)) for i, j in tiny.edges}
    assert len(e_plus) > 0
    assert all(((int(i), int(j)) not in observed for i, j in e_plus))
    assert (e_plus[:, 0] < e_plus[:, 1]).all(), 'canonical i<j form'
    assert len(np.unique(e_plus, axis=0)) == len(e_plus)

def test_topk_budget_bound(tiny):
    k = 3
    e_plus = retrieve_candidates(tiny.x, tiny.edges, k=k)
    assert 0 < len(e_plus) <= k * tiny.n

def test_k_zero_gives_observed_only(tiny):
    e_plus = retrieve_candidates(tiny.x, tiny.edges, k=0)
    cand, is_obs = build_candidate_set(tiny.edges, e_plus)
    assert cand.shape[0] == tiny.num_edges
    assert bool(is_obs.all())

def test_hybrid_score_prefers_similar_and_overlapping():
    edges = canonical_edges(np.array([[0, 1], [0, 2], [0, 3], [1, 4], [2, 4]]))
    x = torch.eye(6)
    x[5] = x[3]
    e_plus = retrieve_candidates(x, torch.tensor(edges), k=1, lambda_x=0.5)
    pairs = {(int(i), int(j)) for i, j in e_plus}
    assert (0, 4) in pairs
    assert (3, 5) in pairs

def test_source_flags(tiny, tiny_candidates):
    cand, is_obs = tiny_candidates
    m_obs = tiny.num_edges
    assert bool(is_obs[:m_obs].all()) and (not bool(is_obs[m_obs:].any()))
    assert torch.equal(cand[:m_obs], tiny.edges)

def test_adjacency_symmetric(tiny):
    a = adjacency_csr(tiny.edges.numpy(), tiny.n)
    assert (a != a.T).nnz == 0
    assert a.diagonal().sum() == 0
