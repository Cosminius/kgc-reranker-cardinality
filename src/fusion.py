import numpy as np

BINS = ("1", "2-9", "10-99", "100-999", "1000+")


def get_bin(cardinality):
    if cardinality == 1:
        return "1"
    if cardinality <= 9:
        return "2-9"
    if cardinality <= 99:
        return "10-99"
    if cardinality <= 999:
        return "100-999"
    return "1000+"


def min_max_normalize(scores):
    a = np.asarray(scores, dtype=np.float64)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-9:
        return np.full_like(a, 0.5)
    return (a - lo) / (hi - lo)


def _fused_rank(r, alpha):
    fused = alpha * min_max_normalize(r["rerank_scores"]) + (1 - alpha) * min_max_normalize(r["bi_scores"])
    gold = r["gold_idx_in_topk"]
    return int((fused > fused[gold]).sum()) + 1


def select_alphas(valid_scored, threshold_T, grid=None):
    """Per-bin alpha grid search on validation. None = bin has no admitted queries."""
    if grid is None:
        grid = np.arange(0.0, 1.01, 0.05)
    alphas = {b: None for b in BINS}
    for b in BINS:
        admitted = [
            qid for qid, r in valid_scored.items()
            if r["source"] == "reranker"
            and r["n_filtered"] <= threshold_T
            and get_bin(r["n_filtered"] + 1) == b
        ]
        if not admitted:
            continue
        best_a, best_score = 0.5, -1.0
        for a in grid:
            total_rr = sum(1.0 / _fused_rank(valid_scored[qid], a) for qid in admitted)
            if total_rr > best_score:
                best_score, best_a = total_rr, float(a)
        alphas[b] = best_a
    return alphas


def apply_fusion(test_scored, alphas, threshold_T, gated_ranks):
    out = {}
    for qid, r in test_scored.items():
        if r["source"] == "bi_encoder_fallback":
            out[qid] = r["bi_rank"]
            continue
        b = get_bin(r["n_filtered"] + 1)
        a = alphas.get(b)
        if a is None or r["n_filtered"] > threshold_T:
            out[qid] = gated_ranks[qid]
        else:
            out[qid] = _fused_rank(r, a)
    return out
