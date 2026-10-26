from __future__ import annotations
import dataclasses
import random
from dataclasses import dataclass, field
import numpy as np
import torch
from leap.audit import AuditCache, BaseAuditor, audit_value, query_priority, run_external_audit, select_queries
from leap.candidate import build_candidate_set, retrieve_candidates
from leap.config import LEAPConfig
from leap.evidence import anomaly_scores, evidence_matrix, static_signals
from leap.fusion import fuse
from leap.internal import InternalScorer, triplet_stats
from leap.losses import classification_loss, confidence_loss, prior_loss, stability_loss
from leap.model import DualPathGCN
from leap.records import EdgeRecords

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum() == 0:
        return float('nan')
    return float((logits[mask].argmax(dim=1) == y[mask]).float().mean())

def _clone_records(r: EdgeRecords) -> EdgeRecords:
    return EdgeRecords(**{f.name: getattr(r, f.name).clone() for f in dataclasses.fields(EdgeRecords)})

@dataclass
class TrainResult:
    test_acc: float
    val_acc: float
    best_epoch: int
    records: EdgeRecords | None
    diagnostics: dict = field(default_factory=dict)
    audit_stats: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

def audit_diagnostics(records: EdgeRecords, config: LEAPConfig) -> dict:
    obs = records.is_observed
    valid = records.m
    n_obs = int(obs.sum())
    n_valid = int(valid.sum())
    out = {'coverage_observed': float((valid & obs).sum()) / max(n_obs, 1), 'high_beta_fraction': float((records.beta >= config.high_beta_threshold).float().mean()), 'num_valid_audits': n_valid}
    if n_valid > 0:
        vin = records.tau_in[valid].argmax(dim=1)
        vout = records.tau_out[valid].argmax(dim=1)
        out['internal_llm_agreement'] = float((vin == vout).float().mean())
        shares = torch.bincount(vout, minlength=3).float() / n_valid
        out['verdict_shares'] = {'supportive': float(shares[0]), 'neutral': float(shares[1]), 'risky': float(shares[2])}
        out['mean_auditor_confidence'] = float(records.tau_out[valid].max(dim=1).values.mean())
    return out

def train_leap(data, config: LEAPConfig | None=None, auditor: BaseAuditor | None=None, cache: AuditCache | str | None=None) -> TrainResult:
    cfg = config or LEAPConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    if isinstance(cache, str):
        cache = AuditCache(cache)
    e_plus = retrieve_candidates(data.x, data.edges, cfg.k, cfg.lambda_x, cfg.eps)
    cand, is_obs = build_candidate_set(data.edges, e_plus)
    static = static_signals(data.x, data.edges, cand, data.y, data.train_mask, cfg.eps)
    cand, is_obs, static = (cand.to(device), is_obs.to(device), static.to(device))
    records = EdgeRecords.empty(cand, is_obs)
    nu = audit_value(cand, is_obs, data.n, cfg.eta_plus, cfg.eps)
    x, y = (data.x.to(device), data.y.to(device))
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)
    model = DualPathGCN(data.d, cfg.hidden, data.num_classes, cfg.n_layers, cfg).to(device)
    scorer = InternalScorer(5, cfg.mlp_hidden).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(scorer.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    with torch.no_grad():
        h_prev = model(x, cand, is_obs.float(), torch.zeros(records.num_edges, device=device))
    history: list = []

    def run_epoch(phase: int, epoch: int):
        nonlocal h_prev
        anom = anomaly_scores(h_prev, cand, cfg.eps)
        z = evidence_matrix(static, anom)
        tau_in = scorer(z)
        c_in, u_in = triplet_stats(tau_in)
        strategy = 'internal_only' if phase == 1 else cfg.fusion
        alpha, beta = fuse(c_in, u_in, tau_in, records.c_out, records.u_out, records.tau_out, records.m, gamma_u=cfg.gamma_u, strategy=strategy)
        logits = model(x, cand, alpha, beta)
        loss = classification_loss(logits, y, train_mask) + cfg.lambda2 * prior_loss(alpha, is_obs) + cfg.lambda3 * stability_loss(logits)
        if phase == 3:
            loss = loss + cfg.lambda1 * confidence_loss(c_in, records.c_out, alpha, beta, records.m)
        opt.zero_grad()
        loss.backward()
        opt.step()
        h_prev = logits.detach()
        records.z_in = z.detach()
        records.tau_in = tau_in.detach()
        records.c_in = c_in.detach()
        records.u_in = u_in.detach()
        records.alpha = alpha.detach()
        records.beta = beta.detach()
        with torch.no_grad():
            metrics = {'phase': phase, 'epoch': epoch, 'loss': float(loss), 'train_acc': accuracy(logits, y, train_mask), 'val_acc': accuracy(logits, y, val_mask), 'test_acc': accuracy(logits, y, test_mask)}
        history.append(metrics)
        return metrics
    for t in range(cfg.T0):
        run_epoch(phase=1, epoch=t)
    audit_stats = {'queried': 0, 'cache_hits': 0, 'failed': 0}
    if auditor is not None and cfg.r_audit > 0:
        if data.texts is None:
            raise ValueError('external auditing needs node texts, but data.texts is None; pass auditor=None to run LEAP-int on this dataset')
        pi = query_priority(records.c_in, records.u_in, nu, cfg.lambda_q, cfg.lambda_v)
        q_idx = select_queries(pi, cfg.r_audit)
        audit_stats = run_external_audit(records, data.texts, q_idx, auditor, cache)
    best = {'val': -1.0, 'test': 0.0, 'epoch': -1, 'records': _clone_records(records)}
    stale = 0
    for t in range(cfg.T):
        metrics = run_epoch(phase=3, epoch=cfg.T0 + t)
        if metrics['val_acc'] > best['val']:
            best.update(val=metrics['val_acc'], test=metrics['test_acc'], epoch=metrics['epoch'], records=_clone_records(records))
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    final_records = best['records']
    return TrainResult(test_acc=best['test'], val_acc=best['val'], best_epoch=best['epoch'], records=final_records, diagnostics=audit_diagnostics(final_records, cfg), audit_stats=audit_stats, history=history)

def train_gcn_baseline(data, config: LEAPConfig | None=None) -> TrainResult:
    cfg = config or LEAPConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    x, y = (data.x.to(device), data.y.to(device))
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)
    edges = data.edges.to(device)
    alpha = torch.ones(edges.shape[0], device=device)
    beta = torch.zeros(edges.shape[0], device=device)
    model = DualPathGCN(data.d, cfg.hidden, data.num_classes, cfg.n_layers, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = {'val': -1.0, 'test': 0.0, 'epoch': -1}
    stale = 0
    history = []
    for t in range(cfg.T0 + cfg.T):
        logits = model(x, edges, alpha, beta)
        loss = classification_loss(logits, y, train_mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            val = accuracy(logits, y, val_mask)
            test = accuracy(logits, y, test_mask)
        history.append({'phase': 0, 'epoch': t, 'loss': float(loss.detach()), 'val_acc': val, 'test_acc': test})
        if val > best['val']:
            best.update(val=val, test=test, epoch=t)
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    return TrainResult(test_acc=best['test'], val_acc=best['val'], best_epoch=best['epoch'], records=None, history=history)
