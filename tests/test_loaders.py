from dataclasses import replace
from pathlib import Path
import pytest
import torch
DATA_ROOT = Path(__file__).resolve().parents[1] / 'data' / 'raw'
needs_data = pytest.mark.skipif(not (DATA_ROOT / 'cora' / 'processed_data.pt').exists(), reason='run scripts/download_data.py first')
needs_dblp = pytest.mark.skipif(not (DATA_ROOT / 'dblp' / 'dblp.npz').exists(), reason='run scripts/download_data.py first')

@pytest.fixture(scope='module')
def cora():
    from leap.loaders import load_dataset
    return load_dataset('cora', root=DATA_ROOT)

@needs_data
def test_cora_matches_prognn_protocol(cora):
    d = cora
    assert d.n == 2485 and d.num_edges == 5069
    assert d.d == 1433 and d.num_classes == 7
    assert len(d.texts) == d.n
    assert not any((t.startswith('Title:') for t in d.texts)), 'markers stripped'
    with_abstract = sum(('\n' in t for t in d.texts))
    assert with_abstract > 0.7 * d.n, 'most nodes carry title+abstract'
    assert int(d.train_mask.sum()) == round(0.1 * d.n)
    assert not (d.train_mask & d.val_mask).any() and (not (d.train_mask & d.test_mask).any())
    assert (d.edges[:, 0] < d.edges[:, 1]).all()

@needs_data
def test_cora_split_is_seeded(cora):
    from leap.loaders import load_dataset
    c = load_dataset('cora', root=DATA_ROOT, seed=1)
    assert int(c.train_mask.sum()) == int(cora.train_mask.sum())
    assert not torch.equal(cora.train_mask, c.train_mask)

@needs_dblp
def test_dblp_loads_without_texts():
    from leap.loaders import load_dataset
    d = load_dataset('dblp', root=DATA_ROOT)
    assert d.num_classes == 4 and d.d == 1639
    assert d.n == 16191, 'largest connected component'
    assert d.texts is None

def test_audit_requires_texts():
    from leap.audit import CosineAuditor
    from leap.config import LEAPConfig
    from leap.data import synthetic_tag
    from leap.train import train_leap
    data = replace(synthetic_tag(n=60, num_classes=3, d=8, seed=0), texts=None)
    with pytest.raises(ValueError, match='LEAP-int'):
        train_leap(data, LEAPConfig(T0=1, T=1, k=2, hidden=8), auditor=CosineAuditor())

def test_normalize_text_variants():
    from leap.loaders import normalize_text
    assert normalize_text('Title: A \tAbstract: B') == 'A\nB'
    assert normalize_text('Title: A\nAbstract: B') == 'A\nB'
    assert normalize_text('Title: Only a title') == 'Only a title'
    out = normalize_text('First sentence. Second sentence follows here.')
    assert out.startswith('First sentence.\n')
    assert normalize_text('already\nsplit') == 'already\nsplit'
