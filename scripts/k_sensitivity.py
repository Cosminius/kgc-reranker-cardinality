"""Sensitivity to the number K of reranked candidates: recall@K and CARF MRR on test when the
candidate lists are cut to the first K entries (the weights are re-selected on validation).

Run after scripts/evaluate.py, which saves the per-query scores:

    python scripts/k_sensitivity.py --config configs/wn18rr.yaml --seed 0
"""

import argparse
import json

from _common import load_config, load_scored, results_prefix
from src.analysis import total_mrr
from src.evaluation import select_and_evaluate


def truncate(scored, k):
    """Keep the first k candidates; queries whose gold answer drops out keep its bi-encoder rank."""
    out = {}
    for q, r in scored.items():
        if r["source"] == "reranker" and r["gold_idx_in_topk"] < k:
            out[q] = dict(r, bi_scores=r["bi_scores"][:k], rerank_scores=r["rerank_scores"][:k])
        elif r["source"] == "reranker":
            out[q] = {"source": "bi_encoder_fallback", "bi_rank": r["gold_idx_in_topk"] + 1,
                      "direction": r["direction"], "n_filtered": r["n_filtered"]}
        else:
            out[q] = r
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--arm", choices=["unmasked", "masked"], default="unmasked")
    p.add_argument("--ks", type=int, nargs="+", default=[10, 20, 30, 50])
    args = p.parse_args()
    cfg = load_config(args.config)

    prefix = results_prefix(cfg, args.seed, args.arm)
    valid, test = load_scored(prefix, "valid"), load_scored(prefix, "test")

    rows = {}
    for k in args.ks:
        tk = truncate(test, k)
        _, configs, _ = select_and_evaluate(truncate(valid, k), tk, float(cfg.get("alpha_grid_step", 0.05)))
        recall = sum(r["source"] == "reranker" for r in tk.values()) / len(tk)
        rows[k] = {"recall": recall, "mrr": total_mrr(configs["Per-bin fusion"])}
        print(f"K={k:3d}: recall@K {recall:.3f} | CARF MRR {rows[k]['mrr']:.4f}")
    with open(f"{prefix}_k_sensitivity.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
