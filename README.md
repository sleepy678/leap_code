# LEAP

Reference implementation of LEAP (Layer-Persistent Edge-Confidence Auditing and Propagation).

## Install

```bash
pip install -r requirements.txt
pip install torch-geometric   # only for Cora / CiteSeer / PubMed
```

## Quick check (offline, no data download)

```bash
python scripts/run_synthetic.py --n 300 --perturb add --rate 1.0
python -m pytest tests -q
```

## Real data

```bash
python scripts/download_data.py
python scripts/run_real.py --dataset cora --perturb add --rate 0.10
python scripts/run_benchmarks.py --datasets cora citeseer pubmed --seeds 0 1 2 3 4 5 6 7 8 9
```

To use an external LLM auditor, set the following and pass `--auditor http`:

```bash
export LEAP_LLM_BASE_URL="https://<endpoint>/v1"
export LEAP_LLM_API_KEY="<key>"
export LEAP_LLM_MODEL="<model-name>"
```

Audit responses are cached as JSON and reused across runs.
