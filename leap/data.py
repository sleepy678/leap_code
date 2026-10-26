from __future__ import annotations
import math
from dataclasses import dataclass, replace
import numpy as np
import torch
TOPIC_NAMES = ['graph neural networks', 'convex optimization', 'reinforcement learning', 'computational biology', 'database systems', 'quantum computing', 'natural language processing', 'computer vision']

@dataclass
class TAGData:
    x: torch.Tensor
    y: torch.Tensor
    edges: torch.Tensor
    texts: list | None
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor

    @property
    def n(self) -> int:
        return self.x.shape[0]

    @property
    def d(self) -> int:
        return self.x.shape[1]

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item()) + 1

    @property
    def num_edges(self) -> int:
        return self.edges.shape[0]

def canonical_edges(pairs: np.ndarray) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    lo = np.minimum(pairs[:, 0], pairs[:, 1])
    hi = np.maximum(pairs[:, 0], pairs[:, 1])
    return np.unique(np.stack([lo, hi], axis=1), axis=0)

def split_nodes(n: int, seed: int=0, train_frac: float=0.1, val_frac: float=0.1):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = max(1, int(round(train_frac * n)))
    n_va = max(1, int(round(val_frac * n)))
    masks = []
    for idx in (perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]):
        m = torch.zeros(n, dtype=torch.bool)
        m[torch.from_numpy(np.ascontiguousarray(idx))] = True
        masks.append(m)
    return masks

def _node_text(topic: str, idx: int, rng: np.random.Generator) -> str:
    verbs = ['studies', 'revisits', 'analyzes', 'proposes a method for', 'surveys']
    aspects = ['scalability', 'robustness', 'generalization', 'sample efficiency', 'interpretability']
    title = f'A study of {topic} ({verbs[int(rng.integers(len(verbs)))]} paper {idx})'
    abstract = f'This paper {verbs[int(rng.integers(len(verbs)))]} {topic}, with emphasis on {aspects[int(rng.integers(len(aspects)))]}. We report experiments and discuss implications for {topic}.'
    return f'{title}\n{abstract}'

def synthetic_tag(n: int=300, num_classes: int=4, d: int=32, avg_degree: float=6.0, inter_ratio: float=0.1, feature_noise: float=0.8, seed: int=0) -> TAGData:
    assert num_classes <= len(TOPIC_NAMES), 'not enough topic phrases'
    rng = np.random.default_rng(seed)
    y = rng.integers(0, num_classes, size=n)
    prototypes = rng.normal(size=(num_classes, d))
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    x = prototypes[y] + feature_noise * rng.normal(size=(n, d)) / math.sqrt(d)
    target_edges = int(round(avg_degree * n / 2))
    n_inter = int(round(inter_ratio * target_edges))
    n_intra = target_edges - n_inter
    pairs = []
    while len(pairs) < n_intra:
        i, j = rng.integers(0, n, size=2)
        if i != j and y[i] == y[j]:
            pairs.append((i, j))
    while len(pairs) < n_intra + n_inter:
        i, j = rng.integers(0, n, size=2)
        if i != j and y[i] != y[j]:
            pairs.append((i, j))
    edges = canonical_edges(np.array(pairs))
    texts = [_node_text(TOPIC_NAMES[c], i, rng) for i, c in enumerate(y)]
    train_mask, val_mask, test_mask = split_nodes(n, seed=seed)
    return TAGData(x=torch.tensor(x, dtype=torch.float32), y=torch.tensor(y, dtype=torch.long), edges=torch.tensor(edges, dtype=torch.long), texts=texts, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

def perturb_random_add(data: TAGData, rate: float, seed: int=0) -> TAGData:
    rng = np.random.default_rng(seed)
    edges = data.edges.numpy()
    existing = {(int(i), int(j)) for i, j in edges}
    budget = math.ceil(rate * len(edges))
    added = set()
    while len(added) < budget:
        i, j = (int(rng.integers(data.n)), int(rng.integers(data.n)))
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key not in existing and key not in added:
            added.add(key)
    new_edges = canonical_edges(np.vstack([edges, np.array(sorted(added))]))
    return replace(data, edges=torch.tensor(new_edges, dtype=torch.long))

def perturb_random_delete(data: TAGData, rate: float, seed: int=0) -> TAGData:
    rng = np.random.default_rng(seed)
    m = data.num_edges
    n_drop = min(m, math.ceil(rate * m))
    keep = np.ones(m, dtype=bool)
    keep[rng.choice(m, size=n_drop, replace=False)] = False
    return replace(data, edges=data.edges[torch.from_numpy(keep)])
