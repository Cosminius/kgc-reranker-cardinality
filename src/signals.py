"""Other ways of grouping queries for score fusion: one weight per group, selected on validation.
Groups with too few validation queries use the global weight."""

import random
from collections import defaultdict

import numpy as np

from .analysis import total_mrr
from .fusion import FusionRanks, alpha_grid, get_bin


def relation_categories(train_triples):
    """1-1 / 1-N / N-1 / N-N per relation, from the average number of tails per head and heads
    per tail in the training triples (threshold 1.5, as in TransE and TransH)."""
    tails = defaultdict(lambda: defaultdict(set))
    heads = defaultdict(lambda: defaultdict(set))
    for t in train_triples:
        tails[t["relation"]][t["head_id"]].add(t["tail_id"])
        heads[t["relation"]][t["tail_id"]].add(t["head_id"])
    out = {}
    for rel in tails:
        tph = np.mean([len(v) for v in tails[rel].values()])
        hpt = np.mean([len(v) for v in heads[rel].values()])
        out[rel] = ("N" if hpt >= 1.5 else "1") + "-" + ("N" if tph >= 1.5 else "1")
    return out


def bi_encoder_margin(scored):
    """Gap between the two best bi-encoder scores, for queries whose gold answer is in the top-K."""
    out = {}
    for q, r in scored.items():
        if r["source"] == "reranker":
            top = sorted(r["bi_scores"], reverse=True)
            out[q] = top[0] - top[1]
    return out


def _groupings(fit, scored, records, categories, min_relation_queries):
    margin = bi_encoder_margin(scored)
    edges = np.quantile([margin[q] for q in fit.reranked], [0.2, 0.4, 0.6, 0.8])

    def category(q):
        rel = records[q]["relation"]
        c = categories.get(rel.replace("inverse ", ""), "N-N")
        return c[::-1] if rel.startswith("inverse ") else c   # head prediction swaps the sides

    return {
        "None (global fusion)": (lambda q: "all", 1),
        "Cardinality bin": (lambda q: get_bin(scored[q]["n_filtered"] + 1), 1),
        "Relation category (1-1, 1-N, N-1, N-N)": (category, 1),
        "Bi-encoder margin (5 quantile bins)":
            (lambda q: int(np.searchsorted(edges, margin[q])) if q in margin else 0, 1),
        "Relation and direction": (lambda q: records[q]["relation"], min_relation_queries),
    }


def _fit_and_rank(fit, target, group_of, min_queries):
    g = fit.select(lambda q: "all")["all"]
    counts = defaultdict(int)
    for q in fit.reranked:
        counts[group_of(q)] += 1
    weights = {k: (a if counts[k] >= min_queries else g) for k, a in fit.select(group_of).items()}
    return target.ranks(lambda q: weights.get(group_of(q), g)), len(weights)


def _split_pairs(scored, records, seed):
    def pair(q):
        return records[q]["head_id"], records[q]["relation"]

    pairs = sorted({pair(q) for q in scored})
    random.Random(seed).shuffle(pairs)
    first = set(pairs[: len(pairs) // 2])
    return [q for q in scored if pair(q) in first], [q for q in scored if pair(q) not in first]


def compare_groupings(valid_scored, test_scored, valid_records, test_records, train_triples,
                      step=0.05, min_relation_queries=30, seed=0):
    """For each grouping: the number of weights, test MRR with the weights selected on validation,
    and validation MRR of a two-fold cross-validation inside the validation set.
    `*_records` map query_id to the mined record (for the head and relation of the query)."""
    grid = alpha_grid(step)
    scored = {**valid_scored, **test_scored}
    records = {**valid_records, **test_records}
    categories = relation_categories(train_triples)

    def run(fit, target):
        return {name: _fit_and_rank(fit, target, *g)
                for name, g in _groupings(fit, scored, records, categories, min_relation_queries).items()}

    on_test = run(FusionRanks(valid_scored, grid), FusionRanks(test_scored, grid))
    a, b = (FusionRanks({q: valid_scored[q] for q in half}, grid)
            for half in _split_pairs(valid_scored, valid_records, seed))
    on_b, on_a = run(a, b), run(b, a)
    return {name: {"groups": n, "mrr": total_mrr(ranks),
                   "cv_mrr": total_mrr({**on_a[name][0], **on_b[name][0]})}
            for name, (ranks, n) in on_test.items()}
