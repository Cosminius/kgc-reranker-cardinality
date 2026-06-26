import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .fusion import BINS, get_bin


def per_bin_mrr(ranks, n_train_by_qid):
    by_bin = defaultdict(list)
    for qid, rank in ranks.items():
        by_bin[get_bin(n_train_by_qid[qid] + 1)].append(rank)
    mrr = {b: float(np.mean([1.0 / r for r in by_bin[b]])) if by_bin[b] else None for b in BINS}
    counts = {b: len(by_bin[b]) for b in BINS}
    return mrr, counts


def total_mrr(ranks):
    return float(np.mean([1.0 / r for r in ranks.values()]))


def build_silaghi_format(*, threshold_T, alphas, gated_ranks, fused_ranks, n_train_by_qid):
    gating_per_bin, counts = per_bin_mrr(gated_ranks, n_train_by_qid)
    fusion_per_bin, _ = per_bin_mrr(fused_ranks, n_train_by_qid)
    return {
        "gating": f"cardinality_only_T{threshold_T}",
        "alphas_bin": alphas,
        "gating_per_bin": gating_per_bin,
        "gating_total": total_mrr(gated_ranks),
        "fusion_per_bin": fusion_per_bin,
        "fusion_total": total_mrr(fused_ranks),
        "counts": counts,
    }


def write_silaghi_json(record, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
