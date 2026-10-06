def apply_cardinality_gate(scored, threshold_T):
    """Rerank a query when it has at most T other known training answers (cardinality <= T + 1);
    otherwise keep the bi-encoder ranking. `n_filtered` must hold the train-only count."""
    out = {}
    for qid, r in scored.items():
        if r["source"] == "bi_encoder_fallback":
            out[qid] = r["bi_rank"]
            continue
        gold = r["gold_idx_in_topk"]
        if r["n_filtered"] <= threshold_T:
            scores = r["rerank_scores"]
        else:
            scores = r["bi_scores"]
        out[qid] = sum(s > scores[gold] for s in scores) + 1
    return out


def select_threshold(valid_scored, grid):
    """Gate threshold with the highest validation MRR (gate only, before fusion).
    Ties go to the earliest value in `grid`. Returns (T, {T: validation MRR})."""
    from .analysis import total_mrr

    table = {T: total_mrr(apply_cardinality_gate(valid_scored, T)) for T in grid}
    return max(grid, key=lambda T: table[T]), table
