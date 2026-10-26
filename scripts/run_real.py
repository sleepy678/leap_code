from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from leap.audit import CosineAuditor, HTTPAuditor, MockTopicAuditor
from leap.config import LEAPConfig
from leap.data import perturb_random_add, perturb_random_delete
from leap.loaders import load_dataset
from leap.train import train_gcn_baseline, train_leap

def main() -> None:
    ap = argparse.ArgumentParser(description="LEAP")
    ap.add_argument('--dataset', choices=['cora', 'citeseer', 'pubmed', 'dblp'], default='cora')
    ap.add_argument('--root', default=str(Path(__file__).resolve().parents[1] / 'data' / 'raw'))
    ap.add_argument('--methods', nargs='+', default=['gcn', 'leap-int', 'leap'], choices=['gcn', 'leap-int', 'leap'])
    ap.add_argument('--auditor', choices=['cosine', 'mock', 'http'], default='cosine')
    ap.add_argument('--cache', default=None, help='audit cache path (default: data/audit_cache_<dataset>_<auditor>.json)')
    ap.add_argument('--perturb', choices=['none', 'add', 'delete'], default='none')
    ap.add_argument('--rate', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--k', type=int, default=None, help='override candidate pool size (paper default 10)')
    ap.add_argument('--T0', type=int, default=None)
    ap.add_argument('--T', type=int, default=None)
    args = ap.parse_args()
    cfg = LEAPConfig(seed=args.seed, device=args.device)
    for name in ('k', 'T0', 'T'):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    data = load_dataset(args.dataset, root=args.root, seed=args.seed)
    print(f'{args.dataset}: n={data.n}, |E|={data.num_edges}, d={data.d}, C={data.num_classes}, texts={('yes' if data.texts is not None else 'no')}, device={args.device}')
    if args.perturb == 'add':
        data = perturb_random_add(data, args.rate, seed=args.seed)
    elif args.perturb == 'delete':
        data = perturb_random_delete(data, args.rate, seed=args.seed)
    if args.perturb != 'none':
        print(f'perturbed ({args.perturb} @ {args.rate:.0%}): |E|={data.num_edges}')
    auditor = {'cosine': CosineAuditor(), 'mock': MockTopicAuditor(), 'http': HTTPAuditor()}[args.auditor]
    cache = args.cache or str(Path(args.root).parent / f'audit_cache_{args.dataset}_{auditor.name}.json')
    print(f'\n{'method':<10} {'val acc':>8} {'test acc':>9} {'best ep':>8} {'time':>7}')
    for method in args.methods:
        t0 = time.time()
        if method == 'gcn':
            res = train_gcn_baseline(data, cfg)
        elif method == 'leap-int':
            res = train_leap(data, cfg, auditor=None)
        else:
            res = train_leap(data, cfg, auditor=auditor, cache=cache)
        print(f'{method:<10} {res.val_acc:>8.4f} {res.test_acc:>9.4f} {res.best_epoch:>8d} {time.time() - t0:>6.1f}s')
        if method == 'leap':
            print(f'           audit: {res.audit_stats}')
            print(f'           diagnostics: {res.diagnostics}')
if __name__ == '__main__':
    main()
