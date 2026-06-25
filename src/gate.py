def apply_cardinality_gate(test_scored, threshold_T):
    """Rerank when n_train <= T, else keep bi-encoder ranking."""
    out = {}
    for qid, r in test_scored.items():
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
