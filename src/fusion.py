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


def alpha_grid(step=0.05):
    return np.round(np.arange(0.0, 1.0 + 1e-9, step), 10)


def min_max_normalize(scores):
    a = np.asarray(scores, dtype=np.float64)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-9:
        return np.full_like(a, 0.5)
    return (a - lo) / (hi - lo)


class FusionRanks:
    """Rank of the gold answer under every fusion weight of the grid, computed once per query.

    Weight 0 is the bi-encoder ranking and weight 1 the cross-encoder ranking. Queries whose
    gold answer is outside the top-K keep their bi-encoder rank for every weight.
    `n_filtered` must hold the number of other known answers in the training triples.
    """

    def __init__(self, scored, grid):
        self.grid = [float(a) for a in grid]
        assert self.grid[0] == 0.0 and self.grid[-1] == 1.0, "the grid must include 0 and 1"
        self.qids = sorted(scored)
        self.n_train = {q: scored[q]["n_filtered"] for q in self.qids}
        self.reranked = []                    # queries whose gold answer is in the top-K
        self.table = {}
        a = np.asarray(self.grid)[:, None]
        for q in self.qids:
            r = scored[q]
            if r["source"] == "bi_encoder_fallback":
                self.table[q] = np.full(len(self.grid), r["bi_rank"])
                continue
            fused = a * min_max_normalize(r["rerank_scores"]) + (1 - a) * min_max_normalize(r["bi_scores"])
            g = r["gold_idx_in_topk"]
            self.table[q] = (fused > fused[:, [g]]).sum(axis=1) + 1
            self.reranked.append(q)

    def bin(self, q):
        return get_bin(self.n_train[q] + 1)

    def select(self, group_of, admitted=lambda q: True):
        """Weight with the highest sum of reciprocal ranks in each group (validation).
        Ties go to the smaller weight."""
        groups = {}
        for q in self.reranked:
            if admitted(q):
                groups.setdefault(group_of(q), []).append(q)
        return {g: self.grid[int(np.argmax(np.sum([1.0 / self.table[q] for q in qs], axis=0)))]
                for g, qs in groups.items()}

    def ranks(self, alpha_of):
        """{query: rank} with weight alpha_of(q); None means the bi-encoder ranking."""
        return {q: int(self.table[q][self.grid.index(alpha_of(q) or 0.0)]) for q in self.qids}


def select_alphas(fr, threshold_T):
    """One weight per cardinality bin, from the validation queries the gate admits."""
    return fr.select(fr.bin, admitted=lambda q: fr.n_train[q] <= threshold_T)


def apply_fusion(fr, alphas, threshold_T):
    """Per-bin fusion; queries above the gate threshold keep the bi-encoder ranking."""
    return fr.ranks(lambda q: alphas.get(fr.bin(q)) if fr.n_train[q] <= threshold_T else None)
