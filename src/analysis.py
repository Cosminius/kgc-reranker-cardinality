import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .fusion import BINS, get_bin


def reciprocal_rank(rank):
    """1/rank; a non-positive rank (gold entity unknown to the bi-encoder) counts as 0."""
    return 1.0 / rank if rank > 0 else 0.0


def per_bin_mrr(ranks, n_train_by_qid):
    by_bin = defaultdict(list)
    for qid, rank in ranks.items():
        by_bin[get_bin(n_train_by_qid[qid] + 1)].append(reciprocal_rank(rank))
    mrr = {b: float(np.mean(by_bin[b])) if by_bin[b] else None for b in BINS}
    counts = {b: len(by_bin[b]) for b in BINS}
    return mrr, counts


def total_mrr(ranks):
    return float(np.mean([reciprocal_rank(r) for r in ranks.values()]))


def hits_at_k(ranks, k):
    return float(np.mean([0 < r <= k for r in ranks.values()]))


def build_result_record(*, configs, n_train_by_qid, selection, significance, meta):
    """Results for one dataset/arm.

    configs: {name: {query_id: rank}} for "Bi-encoder", "Reranker", "Global fusion", "Per-bin fusion".
    selection: the values chosen on validation and the test accuracy of every gate threshold.
    """
    per_bin, totals, hits = {}, {}, {}
    counts = None
    for name, ranks in configs.items():
        per_bin[name], counts = per_bin_mrr(ranks, n_train_by_qid)
        totals[name] = total_mrr(ranks)
        hits[name] = {f"H@{k}": hits_at_k(ranks, k) for k in (1, 3, 10)}
    return {
        **meta,
        "gate_threshold_T": selection["gate_threshold_T"],
        "alphas_per_bin": selection["alphas_per_bin"],
        "global_alpha": selection["global_alpha"],
        "validation_mrr_per_T": {str(k): v for k, v in selection["validation_mrr_per_T"].items()},
        "gate_on_test": {str(k): v for k, v in selection["gate_on_test"].items()},
        "counts_per_bin": counts,
        "mrr_per_bin": per_bin,
        "mrr": totals,
        "hits": hits,
        "significance": significance,
    }


def write_results(record, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def write_rank_csv(out_path, configs, scored, n_train_by_qid):
    """One row per test query with its rank under every configuration (for paired tests)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "direction", "n_train", "bin", "source"] + [f"rank {c}" for c in configs])
        for qid in sorted(scored):
            r = scored[qid]
            n = n_train_by_qid[qid]
            w.writerow([qid, r["direction"], n, get_bin(n + 1), r["source"]] + [configs[c][qid] for c in configs])
