import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leap.candidate import build_candidate_set, retrieve_candidates
from leap.data import synthetic_tag

@pytest.fixture(scope='session')
def tiny():
    return synthetic_tag(n=120, num_classes=4, d=16, avg_degree=6.0, seed=7)

@pytest.fixture(scope='session')
def tiny_candidates(tiny):
    e_plus = retrieve_candidates(tiny.x, tiny.edges, k=3)
    cand, is_obs = build_candidate_set(tiny.edges, e_plus)
    return (cand, is_obs)

@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
