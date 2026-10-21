"""Compare cardinality bins with other ways of grouping queries for score fusion (global weight,
relation categories, bi-encoder margin, relation and direction), on test and in a two-fold
cross-validation inside the validation set.

Run after scripts/evaluate.py, which saves the per-query scores:

    python scripts/routing_signals.py --config configs/wn18rr.yaml --seed 0
"""

import argparse
import json

from _common import REPO_ROOT, load_config, load_scored, results_prefix
from src.datasets import load_candidates, load_triples
from src.signals import compare_groupings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--arm", choices=["unmasked", "masked"], default="unmasked")
    args = p.parse_args()
    cfg = load_config(args.config)

    prefix = results_prefix(cfg, args.seed, args.arm)
    mining_dir, top_k = REPO_ROOT / cfg["mining_dir"], int(cfg.get("top_k", 50))
    rows = compare_groupings(
        valid_scored=load_scored(prefix, "valid"),
        test_scored=load_scored(prefix, "test"),
        valid_records={r["query_id"]: r for r in load_candidates(mining_dir, "valid", top_k)},
        test_records={r["query_id"]: r for r in load_candidates(mining_dir, "test", top_k)},
        train_triples=load_triples(REPO_ROOT / cfg["simkgc_data_dir"], "train"),
        step=float(cfg.get("alpha_grid_step", 0.05)),
    )
    for name, row in rows.items():
        print(f"{name:42s} groups {row['groups']:4d} | test MRR {row['mrr']:.4f} "
              f"| validation 2-fold CV MRR {row['cv_mrr']:.4f}")
    with open(f"{prefix}_routing_signals.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
