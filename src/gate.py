from .fusion import apply_fusion, select_alphas


def select_threshold(valid_fr, grid):
    """Optional gate: a query is reranked only if it has at most T other known training answers.

    T and the per-bin weights are selected together: for every T in `grid` the weights are
    chosen on the admitted validation queries, and T is the value with the highest validation
    MRR after fusion. Ties go to the smaller T (less computation).
    Returns (T, weights for T, {T: validation MRR}, {T: weights}).
    """
    from .analysis import total_mrr

    weights = {T: select_alphas(valid_fr, T) for T in grid}
    table = {T: total_mrr(apply_fusion(valid_fr, weights[T], T)) for T in grid}
    best = max(grid, key=lambda T: table[T])
    return best, weights[best], table, weights


def reranked_share(fr, threshold_T):
    """Share of queries sent to the cross-encoder with threshold T."""
    return sum(fr.n_train[q] <= threshold_T for q in fr.qids) / len(fr.qids)
