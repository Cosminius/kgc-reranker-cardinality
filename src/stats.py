import numpy as np


def paired_test(rr_a, rr_b, n_resamples=10_000, seed=0, chunk=500):
    """Paired comparison of two systems on the same queries (arrays of reciprocal ranks).

    Returns the MRR difference a - b, a bootstrap 95% confidence interval, and the two-sided
    p-value of a randomisation (sign-flip) test. Resamples are drawn in chunks to bound memory.
    """
    d = np.asarray(rr_a, dtype=np.float64) - np.asarray(rr_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    obs = d.mean()
    boot, perm = [], []
    for s in range(0, n_resamples, chunk):
        m = min(chunk, n_resamples - s)
        boot.append(d[rng.integers(0, len(d), size=(m, len(d)))].mean(axis=1))
        signs = rng.integers(0, 2, size=(m, len(d)), dtype=np.int8) * 2 - 1
        perm.append((signs * d).mean(axis=1))
    boot, perm = np.concatenate(boot), np.concatenate(perm)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (n_resamples + 1)
    return {"delta": float(obs), "ci95": [float(lo), float(hi)], "p_value": float(p),
            "n_resamples": n_resamples}
