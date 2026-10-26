import pytest
import torch
from leap.audit import MockTopicAuditor
from leap.config import LEAPConfig
from leap.data import perturb_random_add, synthetic_tag
from leap.train import train_gcn_baseline, train_leap
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device')
GPU_CFG = LEAPConfig(k=3, hidden=32, T0=5, T=40, patience=40, seed=0, device='cuda')

def _perturbed():
    data = synthetic_tag(n=200, num_classes=4, d=16, avg_degree=6.0, seed=0)
    return perturb_random_add(data, rate=0.3, seed=0)

@needs_cuda
def test_leap_trains_on_cuda():
    result = train_leap(_perturbed(), GPU_CFG, auditor=MockTopicAuditor())
    assert result.test_acc > 0.6
    assert result.records.alpha.is_cuda
    assert bool(result.records.m.any())

@needs_cuda
def test_gcn_baseline_on_cuda():
    result = train_gcn_baseline(_perturbed(), GPU_CFG)
    assert result.test_acc > 0.4

@needs_cuda
def test_cpu_and_cuda_agree_roughly():
    cpu_cfg = LEAPConfig(k=3, hidden=32, T0=5, T=40, patience=40, seed=0, device='cpu')
    r_cpu = train_leap(_perturbed(), cpu_cfg, auditor=None)
    r_gpu = train_leap(_perturbed(), GPU_CFG, auditor=None)
    assert abs(r_cpu.test_acc - r_gpu.test_acc) < 0.15
