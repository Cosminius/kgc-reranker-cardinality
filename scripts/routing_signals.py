"""Compare cardinality bins with other ways of grouping queries for score fusion (global weight,
relation categories, bi-encoder margin, relation and direction).

Run after scripts/evaluate.py, which saves the per-query scores:

    python scripts/routing_signals.py --config configs/wn18rr.yaml --seed 0
"""

import argparse
import glob
import json
import pickle
from pathlib import Path

from _common import REPO_ROOT, load_config
from src.datasets import load_triples
from src.signals import compare_groupings


def load_records(mining_dir, split, top_k):
    files = sorted(glob.glob(str(Path(mining_dir) / f"{split}_k{top_k}" / "shard_*.jsonl")))
    return {r["query_id"]: r for f in files for r in map(json.loads, open(f, encoding="utf-8"))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--arm", choices=["unmasked", "masked"], default="unmasked")
    args = p.parse_args()
    cfg = load_config(args.config)

    seed_tag = f"seed_{args.seed}" if args.seed is not None else "no_seed"
    prefix = REPO_ROOT / "results" / seed_tag / f"{cfg['dataset'].lower().replace('-', '')}_{args.arm}"
    mining_dir, top_k = REPO_ROOT / cfg["mining_dir"], int(cfg.get("top_k", 50))
    rows = compare_groupings(
        valid_scored=pickle.load(open(f"{prefix}_valid_scored.pkl", "rb")),
        test_scored=pickle.load(open(f"{prefix}_test_scored.pkl", "rb")),
        valid_records=load_records(mining_dir, "valid", top_k),
        test_records=load_records(mining_dir, "test", top_k),
        train_triples=load_triples(REPO_ROOT / cfg["simkgc_data_dir"], "train"),
        step=float(cfg.get("alpha_grid_step", 0.05)),
    )
    for name, row in rows.items():
        print(f"{name:42s} groups {row['groups']:4d} | test MRR {row['mrr']:.4f}")
    with open(f"{prefix}_routing_signals.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
