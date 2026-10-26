import numpy as np
import torch
from leap.candidate import adjacency_csr
from leap.data import canonical_edges
from leap.evidence import anomaly_scores, evidence_matrix, role_vectors, static_signals

def test_static_signal_ranges(tiny, tiny_candidates):
    cand, _ = tiny_candidates
    z4 = static_signals(tiny.x, tiny.edges, cand, tiny.y, tiny.train_mask)
    assert z4.shape == (cand.shape[0], 4)
    sim_x, ov_n, agr_y, role = z4.unbind(dim=1)
    assert float(ov_n.min()) >= 0.0 and float(ov_n.max()) <= 1.0
    assert set(np.unique(agr_y.numpy())).issubset({0.0, 0.5, 1.0})
    assert torch.isfinite(z4).all()

def test_jaccard_manual():
    edges = torch.tensor(canonical_edges(np.array([[0, 1], [1, 2], [0, 2]])))
    x = torch.eye(3)
    y = torch.zeros(3, dtype=torch.long)
    mask = torch.zeros(3, dtype=torch.bool)
    z4 = static_signals(x, edges, edges, y, mask)
    ov = z4[:, 1]
    idx = {(int(i), int(j)): r for r, (i, j) in enumerate(edges.tolist())}
    assert abs(float(ov[idx[0, 1]]) - 1.0 / 3.0) < 1e-06

def test_label_agreement_masking(tiny, tiny_candidates):
    cand, _ = tiny_candidates
    z4 = static_signals(tiny.x, tiny.edges, cand, tiny.y, tiny.train_mask)
    agr = z4[:, 2].numpy()
    tr = tiny.train_mask.numpy()
    pairs = cand.numpy()
    both = tr[pairs[:, 0]] & tr[pairs[:, 1]]
    assert (agr[~both] == 0.5).all(), 'masked entries carry the neutral midpoint'
    same = tiny.y.numpy()[pairs[:, 0]] == tiny.y.numpy()[pairs[:, 1]]
    assert (agr[both] == same[both].astype(float)).all()

def test_role_vectors_triangle():
    edges = canonical_edges(np.array([[0, 1], [1, 2], [0, 2], [2, 3]]))
    a = adjacency_csr(edges, 4)
    r = role_vectors(a)
    assert r.shape == (4, 3)
    assert abs(r[0, 1] - 1.0) < 1e-06
    assert abs(r[2, 1] - 1.0 / 3.0) < 1e-06
    assert abs(r[3, 1]) < 1e-06
    assert abs(r[:, 2].sum() - 1.0) < 0.001

def test_anomaly_flags_outlier_edge():
    torch.manual_seed(0)
    h = torch.cat([torch.randn(10, 8) * 0.05, torch.randn(10, 8) * 0.05 + 10.0])
    intra = [[i, i + 1] for i in range(9)] + [[10 + i, 11 + i] for i in range(9)]
    bridge = [[0, 19]]
    cand = torch.tensor(canonical_edges(np.array(intra + bridge)))
    anom = anomaly_scores(h, cand)
    assert anom.shape == (cand.shape[0],)
    assert float(anom.min()) >= 0.0 and float(anom.max()) <= 1.0
    bridge_row = [r for r, (i, j) in enumerate(cand.tolist()) if (i, j) == (0, 19)][0]
    others = torch.tensor([v for r, v in enumerate(anom.tolist()) if r != bridge_row])
    assert float(anom[bridge_row]) > float(others.max()), 'bridge edge is the anomaly'

def test_evidence_matrix_shape(tiny, tiny_candidates):
    cand, _ = tiny_candidates
    z4 = static_signals(tiny.x, tiny.edges, cand, tiny.y, tiny.train_mask)
    anom = anomaly_scores(torch.randn(tiny.n, 4), cand)
    z = evidence_matrix(z4, anom)
    assert z.shape == (cand.shape[0], 5)
    assert torch.isfinite(z).all()
