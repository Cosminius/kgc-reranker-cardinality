"""Weight selection on validation and evaluation on test, from scored queries."""

import numpy as np

from .analysis import reciprocal_rank
from .fusion import BINS, FusionRanks, alpha_grid, apply_fusion, select_alphas
from .stats import paired_test

CONFIG_NAMES = ("Bi-encoder", "Reranker", "Global fusion", "Per-bin fusion")


def select_and_evaluate(valid_scored, test_scored, step=0.05):
    """Return (test FusionRanks, {configuration: {query: rank}}, selected weights)."""
    grid = alpha_grid(step)
    valid, test = FusionRanks(valid_scored, grid), FusionRanks(test_scored, grid)
    alphas = select_alphas(valid)
    g = valid.select(lambda q: "all")["all"]
    configs = {
        "Bi-encoder": test.ranks(lambda q: 0.0),
        "Reranker": test.ranks(lambda q: 1.0),
        "Global fusion": test.ranks(lambda q: g),
        "Per-bin fusion": apply_fusion(test, alphas),
    }
    return test, configs, {"alphas_per_bin": alphas, "global_alpha": g}


def significance(test, configs, n_resamples=10_000):
    """Paired randomisation test of each configuration against the one listed before it,
    on all test queries and within each cardinality bin, plus CARF against the bi-encoder."""
    names = list(configs)
    pairs = list(zip(names[1:], names[:-1])) + [("Per-bin fusion", "Bi-encoder")]
    groups = {"All": list(test.qids)}
    groups.update({b: [q for q in test.qids if test.bin(q) == b] for b in BINS})
    out = {}
    for group, qs in groups.items():
        if not qs:
            continue
        rr = {c: np.array([reciprocal_rank(configs[c][q]) for q in qs]) for c in names}
        out[group] = {f"{a} vs {b}": paired_test(rr[a], rr[b], n_resamples=n_resamples, seed=0)
                      for a, b in pairs}
    return out


def stars(p_value):
    """*** p < 0.001, ** p < 0.01, * p < 0.05."""
    return "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
