from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import torch
from leap.candidate import adjacency_csr

def role_vectors(a: sp.csr_matrix, damping: float=0.85, iters: int=50) -> np.ndarray:
    n = a.shape[0]
    deg = np.asarray(a.sum(axis=1)).ravel()
    tri = np.asarray((a @ a).multiply(a).sum(axis=1)).ravel() / 2.0
    possible = deg * (deg - 1) / 2.0
    clust = np.divide(tri, possible, out=np.zeros(n), where=possible > 0)
    out_deg = np.maximum(deg, 1.0)
    p = a.multiply(1.0 / out_deg[:, None]).tocsr()
    pr = np.full(n, 1.0 / n)
    dangling = deg == 0
    for _ in range(iters):
        pr = (1 - damping) / n + damping * (p.T @ pr + pr[dangling].sum() / n)
    return np.stack([np.log1p(deg), clust, pr], axis=1)

def _pair_cos(mat: np.ndarray, pairs: np.ndarray, eps: float, chunk: int=16384) -> np.ndarray:
    mat = mat.astype(np.float32, copy=False)
    norms = np.linalg.norm(mat, axis=1)
    out = np.empty(len(pairs), dtype=np.float64)
    for s in range(0, len(pairs), chunk):
        p = pairs[s:s + chunk]
        num = np.einsum('ij,ij->i', mat[p[:, 0]], mat[p[:, 1]])
        out[s:s + chunk] = num / np.maximum(norms[p[:, 0]] * norms[p[:, 1]], eps)
    return out

def static_signals(x: torch.Tensor, edges: torch.Tensor, cand_edges: torch.Tensor, y: torch.Tensor, train_mask: torch.Tensor, eps: float=1e-08) -> torch.Tensor:
    n = x.shape[0]
    pairs = cand_edges.numpy()
    a = adjacency_csr(edges.numpy(), n)
    deg = np.asarray(a.sum(axis=1)).ravel()
    sim_x = _pair_cos(x.detach().cpu().numpy(), pairs, eps)
    rows_i, rows_j = (a[pairs[:, 0]], a[pairs[:, 1]])
    inter = np.asarray(rows_i.multiply(rows_j).sum(axis=1)).ravel()
    union = deg[pairs[:, 0]] + deg[pairs[:, 1]] - inter
    ov_n = inter / (union + eps)
    tr = train_mask.numpy()
    yy = y.numpy()
    both_tr = tr[pairs[:, 0]] & tr[pairs[:, 1]]
    agr_y = np.where(both_tr, (yy[pairs[:, 0]] == yy[pairs[:, 1]]).astype(np.float64), 0.5)
    role = _pair_cos(role_vectors(a), pairs, eps)
    out = np.stack([sim_x, ov_n, agr_y, role], axis=1)
    return torch.tensor(out, dtype=torch.float32)

def anomaly_scores(h: torch.Tensor, cand_edges: torch.Tensor, eps: float=1e-08) -> torch.Tensor:
    h = h.detach()
    n, m = (h.shape[0], cand_edges.shape[0])
    if m == 0:
        return torch.zeros(0, device=h.device)
    src = torch.cat([cand_edges[:, 0], cand_edges[:, 1]])
    dst = torch.cat([cand_edges[:, 1], cand_edges[:, 0]])
    sums = torch.zeros(n, h.shape[1], device=h.device).index_add_(0, src, h[dst])
    cnt = torch.zeros(n, device=h.device).index_add_(0, src, torch.ones_like(src, dtype=h.dtype))
    cnt_safe = cnt.clamp(min=1.0)
    mean_incl = sums / cnt_safe.unsqueeze(1)
    dist_incl = (h - mean_incl).norm(dim=1)
    i_idx, j_idx = (cand_edges[:, 0], cand_edges[:, 1])

    def delta(anchor, other):
        c = cnt[anchor]
        excl = torch.where((c > 1).unsqueeze(1), (sums[anchor] - h[other]) / (c - 1).clamp(min=1.0).unsqueeze(1), h[anchor])
        return dist_incl[anchor] - (h[anchor] - excl).norm(dim=1)
    raw = 0.5 * (delta(i_idx, j_idx) + delta(j_idx, i_idx))
    lo, hi = (raw.min(), raw.max())
    return (raw - lo) / (hi - lo + eps)

def evidence_matrix(static: torch.Tensor, anom: torch.Tensor, standardize: bool=True, eps: float=1e-06) -> torch.Tensor:
    z = torch.cat([static, anom.unsqueeze(1)], dim=1)
    if standardize:
        z = (z - z.mean(dim=0)) / z.std(dim=0).clamp(min=eps)
    return z
