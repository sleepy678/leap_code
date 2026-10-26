from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import torch
from leap.data import canonical_edges

def adjacency_csr(edges: np.ndarray, n: int) -> sp.csr_matrix:
    if len(edges) == 0:
        return sp.csr_matrix((n, n), dtype=np.float32)
    i, j = (edges[:, 0], edges[:, 1])
    data = np.ones(2 * len(edges), dtype=np.float32)
    a = sp.coo_matrix((data, (np.concatenate([i, j]), np.concatenate([j, i]))), shape=(n, n))
    a = a.tocsr()
    a.data[:] = 1.0
    return a

def retrieve_candidates(x: torch.Tensor, edges: torch.Tensor, k: int, lambda_x: float=0.5, eps: float=1e-08, chunk: int=512) -> np.ndarray:
    n = x.shape[0]
    if k <= 0 or n < 3:
        return np.zeros((0, 2), dtype=np.int64)
    edges_np = edges.numpy() if isinstance(edges, torch.Tensor) else np.asarray(edges)
    a = adjacency_csr(edges_np, n)
    deg = np.asarray(a.sum(axis=1)).ravel()
    feats = x.detach().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.maximum(norms, eps)
    pairs = []
    for start in range(0, n, chunk):
        rows = np.arange(start, min(start + chunk, n))
        cos = feats[rows] @ feats.T
        inter = np.asarray((a[rows] @ a.T).todense())
        union = deg[rows][:, None] + deg[None, :] - inter
        jac = inter / (union + eps)
        q = lambda_x * cos + (1.0 - lambda_x) * jac
        q[np.asarray(a[rows].todense()) > 0] = -np.inf
        q[np.arange(len(rows)), rows] = -np.inf
        kk = min(k, n - 1)
        top = np.argpartition(-q, kth=kk - 1, axis=1)[:, :kk]
        for r, i in enumerate(rows):
            for j in top[r]:
                if np.isfinite(q[r, j]):
                    pairs.append((i, int(j)))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    e_plus = canonical_edges(np.array(pairs))
    observed = {(int(i), int(j)) for i, j in edges_np}
    mask = np.array([(int(i), int(j)) not in observed for i, j in e_plus])
    return e_plus[mask]

def build_candidate_set(edges: torch.Tensor, e_plus: np.ndarray):
    edges_np = edges.numpy() if isinstance(edges, torch.Tensor) else np.asarray(edges)
    cand = np.vstack([edges_np, e_plus.reshape(-1, 2)]).astype(np.int64)
    is_observed = np.zeros(len(cand), dtype=bool)
    is_observed[:len(edges_np)] = True
    return (torch.tensor(cand, dtype=torch.long), torch.tensor(is_observed, dtype=torch.bool))
