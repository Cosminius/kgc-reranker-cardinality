"""Check whether the per-bin alpha grid search overfits validation.

Prints:
  1. GATE-ONLY total, which doesn't depend on alpha
  2. fusion_total with the live grid-selected alphas
  3. fusion_total with paper alphas hardcoded (if --paper-alphas is passed)
  4. val MRR vs test MRR per alpha across the grid, for one bin (default 2-9)

If val_MRR keeps rising past where test_MRR peaks, the grid is overfitting.

    python scripts/diagnose_alpha.py --config configs/codex-m.yaml \\
        --checkpoint checkpoints/reranker/CoDEx-M/unmasked/final \\
        --paper-alphas 0.95,1.00,0.90
"""

import argparse

import numpy as np

from _common import REPO_ROOT, load_config
from src.analysis import per_bin_mrr, total_mrr
from src.fusion import BINS, apply_fusion, get_bin, min_max_normalize, select_alphas
from src.gate import apply_cardinality_gate
from src.pipeline import prepare_scored


def per_bin_alpha_mrr(scored, bin_label, T, alpha):
    """MRR at one bin under one fixed alpha, over reranker-admitted queries."""
    rrs = []
    for r in scored.values():
        if r["source"] != "reranker": continue
        if r["n_filtered"] > T: continue
        if get_bin(r["n_filtered"] + 1) != bin_label: continue
        nm_re = min_max_normalize(r["rerank_scores"])
        nm_bi = min_max_normalize(r["bi_scores"])
        fused = alpha * nm_re + (1 - alpha) * nm_bi
        gold = r["gold_idx_in_topk"]
        rank = int((fused > fused[gold]).sum()) + 1
        rrs.append(1.0 / rank)
    return float(np.mean(rrs)) if rrs else None


def print_per_bin(label, ranks, n_train_by_qid):
    mrr, _ = per_bin_mrr(ranks, n_train_by_qid)
    print(f"\n[{label}]")
    for b in BINS:
        v = mrr.get(b)
        print(f"  bin {b:7}: {f'{v:.4f}' if v is not None else '----'}")
    print(f"  total    : {total_mrr(ranks):.4f}")


def alpha_sweep(valid_scored, test_scored, bin_label, T):
    grid = np.arange(0.0, 1.01, 0.05)
    print(f"\n[alpha sweep for bin {bin_label}]")
    print(f"{'alpha':>7} {'val_MRR':>10} {'test_MRR':>10}")
    for a in grid:
        v = per_bin_alpha_mrr(valid_scored, bin_label, T, float(a))
        t = per_bin_alpha_mrr(test_scored, bin_label, T, float(a))
        vs = f"{v:.4f}" if v is not None else "----"
        ts = f"{t:.4f}" if t is not None else "----"
        print(f"  {a:.2f}  {vs:>10} {ts:>10}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--paper-alphas", default=None,
                   help="Comma-separated alphas for bins 1, 2-9, 10-99 (e.g. 0.95,1.0,0.90)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bin", default="2-9")
    args = p.parse_args()

    cfg = load_config(args.config)
    print(f"\n=== DIAGNOSE {cfg['dataset']} (seed={args.seed}) ===")
    print(f"  checkpoint: {args.checkpoint}")

    valid_scored, test_scored, _, T = prepare_scored(
        cfg, REPO_ROOT, args.checkpoint, args.seed
    )
    n_train_test = {qid: r["n_filtered"] for qid, r in test_scored.items()}

    # 1. Alpha-independent baseline
    gated = apply_cardinality_gate(test_scored, threshold_T=T)
    print_per_bin("GATE-ONLY (alpha-independent)", gated, n_train_test)

    # 2. Live grid-selected alphas
    live_alphas = select_alphas(valid_scored, threshold_T=T)
    print(f"\n[live grid-selected alphas]\n  alphas: {live_alphas}")
    fused = apply_fusion(test_scored, live_alphas, threshold_T=T, gated_ranks=gated)
    print_per_bin("LIVE grid fusion", fused, n_train_test)

    # 3. Paper alphas (if provided)
    if args.paper_alphas:
        vals = [float(x) for x in args.paper_alphas.split(",")]
        paper = {"1": vals[0], "2-9": vals[1], "10-99": vals[2], "100-999": None, "1000+": None}
        paper_fused = apply_fusion(test_scored, paper, threshold_T=T, gated_ranks=gated)
        print(f"\n[paper alphas hardcoded]\n  alphas: {paper}")
        print_per_bin("PAPER alphas fusion", paper_fused, n_train_test)

    # 4. Val vs test alpha sweep at one bin
    alpha_sweep(valid_scored, test_scored, args.bin, T)


if __name__ == "__main__":
    main()
