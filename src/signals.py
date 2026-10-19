"""Other ways of grouping queries for score fusion: one weight per group, selected on validation.
Groups with too few validation queries use the global weight."""

from collections import defaultdict

import numpy as np

from .analysis import total_mrr
from .fusion import FusionRanks, alpha_grid


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


def _fit(fr, group_of, fallback, min_queries):
    counts = defaultdict(int)
    for q in fr.reranked:
        counts[group_of(q)] += 1
    weights = fr.select(group_of)
    return {g: (a if counts[g] >= min_queries else fallback) for g, a in weights.items()}


def compare_groupings(valid_scored, test_scored, valid_records, test_records, train_triples,
                      step=0.05, min_relation_queries=30):
    """Test MRR of fusion with one weight per group, for each grouping.
    `*_records` map query_id to the mined record (for the relation of the query)."""
    grid = alpha_grid(step)
    valid, test = FusionRanks(valid_scored, grid), FusionRanks(test_scored, grid)
    g = valid.select(lambda q: "all")["all"]

    categories = relation_categories(train_triples)

    def category(records):
        def f(q):
            rel = records[q]["relation"]
            c = categories.get(rel.replace("inverse ", ""), "N-N")
            return c[::-1] if rel.startswith("inverse ") else c   # head prediction swaps the sides
        return f

    valid_margin, test_margin = bi_encoder_margin(valid_scored), bi_encoder_margin(test_scored)
    edges = np.quantile(list(valid_margin.values()), [0.2, 0.4, 0.6, 0.8])

    def margin(m):
        return lambda q: int(np.searchsorted(edges, m[q])) if q in m else 0

    def relation(records):
        return lambda q: records[q]["relation"]

    groupings = {
        "None (global fusion)": (lambda q: "all", lambda q: "all", 1),
        "Cardinality bin": (valid.bin, test.bin, 1),
        "Relation category (1-1, 1-N, N-1, N-N)": (category(valid_records), category(test_records), 1),
        "Bi-encoder margin (5 quantile bins)": (margin(valid_margin), margin(test_margin), 1),
        "Relation and direction": (relation(valid_records), relation(test_records), min_relation_queries),
    }
    out = {}
    for name, (valid_group, test_group, min_q) in groupings.items():
        weights = _fit(valid, valid_group, g, min_q)
        out[name] = {"groups": len(weights),
                     "mrr": total_mrr(test.ranks(lambda q: weights.get(test_group(q), g)))}
    return out
