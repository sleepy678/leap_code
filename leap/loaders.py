from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from leap.data import TAGData, canonical_edges, split_nodes
GRAPHCOM_NAMES = ('cora', 'citeseer', 'pubmed')

def largest_connected_component(edges: np.ndarray, n: int):
    adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    n_comp, labels = sp.csgraph.connected_components(adj, directed=False)
    keep_label = np.bincount(labels, minlength=n_comp).argmax()
    keep = labels == keep_label
    relabel = -np.ones(n, dtype=np.int64)
    relabel[keep] = np.arange(int(keep.sum()))
    return (keep, relabel)

def _restrict_to_lcc(x, y, edges, texts, n):
    keep, relabel = largest_connected_component(edges, n)
    edge_keep = keep[edges[:, 0]] & keep[edges[:, 1]]
    edges = relabel[edges[edge_keep]]
    texts = [t for t, k in zip(texts, keep) if k] if texts is not None else None
    return (x[keep], y[keep], canonical_edges(edges), texts)

def normalize_text(text: str) -> str:
    text = text.strip()
    m = re.match('Title:\\s*(.*?)\\s*(?:\\\\t|\\t|\\n)?\\s*Abstract:\\s*(.*)', text, re.S)
    if m:
        return f'{m.group(1).strip()}\n{m.group(2).strip()}'
    m = re.match('Title:\\s*(.*)', text, re.S)
    if m:
        return m.group(1).strip()
    if '\n' in text:
        return text
    dot = text.find('. ')
    if 0 < dot < 300:
        return f'{text[:dot + 1]}\n{text[dot + 2:]}'
    return text

def _finalize(x, y, edges, texts, prognn_split: bool, seed: int) -> TAGData:
    n = x.shape[0]
    if prognn_split:
        x, y, edges, texts = _restrict_to_lcc(x, y, edges, texts, n)
        n = x.shape[0]
    train_mask, val_mask, test_mask = split_nodes(n, seed=seed)
    return TAGData(x=torch.as_tensor(x, dtype=torch.float32), y=torch.as_tensor(y, dtype=torch.long), edges=torch.as_tensor(edges, dtype=torch.long), texts=texts, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

def load_graphcom(root: str | Path, name: str, prognn_split: bool=True, seed: int=0) -> TAGData:
    assert name in GRAPHCOM_NAMES, f'{name!r} not in {GRAPHCOM_NAMES}'
    folder = Path(root) / name
    try:
        data = torch.load(folder / 'processed_data.pt', weights_only=False)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError('loading Graph-COM .pt files requires torch_geometric (pip install torch_geometric)') from exc
    texts_file = folder / 'raw_texts.pt'
    raw_texts = torch.load(texts_file, weights_only=False) if texts_file.exists() else data.raw_texts
    texts = [normalize_text(t) for t in raw_texts]
    edges = canonical_edges(data.edge_index.numpy().T)
    x = data.x.numpy()
    y = data.y.numpy()
    return _finalize(x, y, edges, texts, prognn_split, seed)

def load_dblp_npz(path: str | Path, prognn_split: bool=True, seed: int=0) -> TAGData:
    z = np.load(path, allow_pickle=True)
    n = int(z['adj_shape'][0])
    adj = sp.csr_matrix((z['adj_data'], z['adj_indices'], z['adj_indptr']), shape=(n, n))
    attr = sp.csr_matrix((z['attr_data'].astype(np.float32), z['attr_indices'], z['attr_indptr']), shape=tuple(z['attr_shape']))
    coo = adj.tocoo()
    edges = canonical_edges(np.stack([coo.row, coo.col], axis=1))
    labels = np.asarray(z['labels'])
    if prognn_split:
        keep, relabel = largest_connected_component(edges, n)
        edge_keep = keep[edges[:, 0]] & keep[edges[:, 1]]
        edges = canonical_edges(relabel[edges[edge_keep]])
        attr, labels = (attr[keep], labels[keep])
    return _finalize(attr.toarray(), labels, edges, None, prognn_split=False, seed=seed)

def load_dataset(name: str, root: str | Path='data/raw', prognn_split: bool=True, seed: int=0) -> TAGData:
    name = name.lower()
    if name in GRAPHCOM_NAMES:
        return load_graphcom(root, name, prognn_split, seed)
    if name == 'dblp':
        return load_dblp_npz(Path(root) / 'dblp' / 'dblp.npz', prognn_split, seed)
    raise ValueError(f'unknown dataset {name!r}')
