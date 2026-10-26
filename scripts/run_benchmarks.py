from __future__ import annotations
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from leap.audit import AuditCache, CosineAuditor
from leap.config import LEAPConfig
from leap.data import perturb_random_add, perturb_random_delete
from leap.loaders import load_dataset
from leap.train import train_gcn_baseline, train_leap
SETTINGS = [('none', 0.0), ('add', 0.1), ('add', 0.25), ('delete', 0.1)]

def perturbed(data, kind: str, rate: float, seed: int):
    if kind == 'none':
        return data
    fn = perturb_random_add if kind == 'add' else perturb_random_delete
    return fn(data, rate, seed=seed)

def main() -> None:
    ap = argparse.ArgumentParser(description="LEAP")
    ap.add_argument('--datasets', nargs='+', default=['cora', 'citeseer', 'pubmed', 'dblp'])
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    ap.add_argument('--methods', nargs='+', default=['gcn', 'leap-int', 'leap'])
    ap.add_argument('--root', default=str(Path(__file__).resolve().parents[1] / 'data' / 'raw'))
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'results'))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for name in args.datasets:
        base = load_dataset(name, root=args.root)
        has_text = base.texts is not None
        print(f'\n=== {name}: n={base.n} |E|={base.num_edges} d={base.d} C={base.num_classes} texts={('yes' if has_text else 'no')} ===', flush=True)
        for kind, rate in SETTINGS:
            label = 'clean' if kind == 'none' else f'{kind} {rate:.0%}'
            for method in args.methods:
                if method == 'leap' and (not has_text):
                    print(f'  {label:12s} {method:9s} skipped (dataset has no node texts)', flush=True)
                    continue
                accs, times = ([], [])
                for seed in args.seeds:
                    data = perturbed(base, kind, rate, seed)
                    cfg = LEAPConfig(seed=seed, device=args.device)
                    t0 = time.time()
                    if method == 'gcn':
                        res = train_gcn_baseline(data, cfg)
                    elif method == 'leap-int':
                        res = train_leap(data, cfg, auditor=None)
                    else:
                        cache = AuditCache(str(out_dir / f'cache_{name}_{kind}{rate}_{seed}.json'))
                        res = train_leap(data, cfg, auditor=CosineAuditor(), cache=cache)
                    times.append(time.time() - t0)
                    accs.append(res.test_acc)
                mean = statistics.mean(accs)
                std = statistics.stdev(accs) if len(accs) > 1 else 0.0
                print(f'  {label:12s} {method:9s} {mean:.4f} +- {std:.4f}  ({statistics.mean(times):.1f}s/seed)', flush=True)
                records.append({'dataset': name, 'perturbation': kind, 'rate': rate, 'method': method, 'seeds': args.seeds, 'test_acc': accs, 'mean': mean, 'std': std, 'sec_per_seed': statistics.mean(times)})
                (out_dir / 'benchmarks.json').write_text(json.dumps(records, indent=2))
    print(f'\nwrote {out_dir / 'benchmarks.json'} ({len(records)} rows)')
if __name__ == '__main__':
    main()
