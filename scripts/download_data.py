from __future__ import annotations
import sys
import urllib.request
from pathlib import Path
HF_BASE = 'https://huggingface.co/datasets/Graph-COM/Text-Attributed-Graphs/resolve/main'
FILES = {**{f'{ds}/{f}': f'{HF_BASE}/{ds}/{f}' for ds in ('cora', 'citeseer', 'pubmed') for f in ('processed_data.pt', 'raw_texts.pt', 'categories.csv')}, 'dblp/dblp.npz': 'https://github.com/abojchevski/graph2gauss/raw/master/data/dblp.npz'}

def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')

    def hook(blocks, block_size, total):
        done = blocks * block_size
        mb = done / 1000000.0
        pct = f' {100 * done / total:5.1f}%' if total > 0 else ''
        print(f'\r  {dest.name}: {mb:7.1f} MB{pct}', end='', flush=True)
    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    tmp.rename(dest)
    print()

def main() -> None:
    root = Path(__file__).resolve().parents[1] / 'data' / 'raw'
    for rel, url in FILES.items():
        dest = root / rel
        if dest.exists() and dest.stat().st_size > 0:
            print(f'  {rel}: already present, skipping')
            continue
        print(f'downloading {rel}')
        try:
            fetch(url, dest)
        except Exception as exc:
            print(f'\n  FAILED {rel}: {exc}', file=sys.stderr)
            sys.exit(1)
    print(f'done -> {root}')
if __name__ == '__main__':
    main()
