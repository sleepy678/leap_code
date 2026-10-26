from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leap.audit import CosineAuditor, HTTPAuditor, MockTopicAuditor
from leap.config import LEAPConfig
from leap.data import perturb_random_add, perturb_random_delete, synthetic_tag
from leap.train import train_gcn_baseline, train_leap

def main() -> None:
    ap = argparse.ArgumentParser(description="LEAP")
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--classes', type=int, default=4)
    ap.add_argument('--noise', type=float, default=2.0, help='feature noise; higher makes the graph matter more')
    ap.add_argument('--perturb', choices=['none', 'add', 'delete'], default='add')
    ap.add_argument('--rate', type=float, default=1.0)
    ap.add_argument('--auditor', choices=['mock', 'cosine', 'http'], default='mock')
    ap.add_argument('--cache', type=str, default=None, help='JSON audit-cache path')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--T0', type=int, default=15)
    ap.add_argument('--T', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    cfg = LEAPConfig(k=args.k, hidden=args.hidden, T0=args.T0, T=args.T, seed=args.seed)
    data = synthetic_tag(n=args.n, num_classes=args.classes, feature_noise=args.noise, seed=args.seed)
    clean_edges = data.num_edges
    if args.perturb == 'add':
        data = perturb_random_add(data, args.rate, seed=args.seed)
    elif args.perturb == 'delete':
        data = perturb_random_delete(data, args.rate, seed=args.seed)
    print(f"synthetic TAG: n={data.n}, classes={data.num_classes}, edges {clean_edges} -> {data.num_edges} after '{args.perturb}' @ {args.rate:.0%}")
    auditor = {'mock': MockTopicAuditor(), 'cosine': CosineAuditor(), 'http': HTTPAuditor()}[args.auditor]
    rows = []
    gcn = train_gcn_baseline(data, cfg)
    rows.append(('GCN', gcn))
    leap_int = train_leap(data, cfg, auditor=None)
    rows.append(('LEAP-int', leap_int))
    leap = train_leap(data, cfg, auditor=auditor, cache=args.cache)
    rows.append(('LEAP', leap))
    print(f'\n{'method':<10} {'val acc':>8} {'test acc':>9} {'best ep':>8}')
    for name, res in rows:
        print(f'{name:<10} {res.val_acc:>8.4f} {res.test_acc:>9.4f} {res.best_epoch:>8d}')
    print('\naudit stats:', leap.audit_stats)
    print('diagnostics:', {k: v for k, v in leap.diagnostics.items()})
if __name__ == '__main__':
    main()
