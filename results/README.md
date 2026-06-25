# `results/`

Per-seed pipeline outputs from `scripts/evaluate.py`.

- `seed_0/` — the seed used for the paper.
- `seed_<N>/` — populated automatically when you run with `--seed N`.
- `no_seed/` — populated when you run without `--seed`.

Files are named `<dataset>_<arm>.json` (e.g. `wn18rr_masked.json`).

## silaghi_format JSON

Each file is one cell of one paper table.

```jsonc
{
  "dataset":        "CoDEx-M",
  "arm":            "unmasked",
  "seed":           0,                                  // null if --seed was omitted
  "checkpoint":     "checkpoints/reranker/.../final",
  "gating":         "cardinality_only_T9",
  "alphas_bin":     { "1": 0.80, "2-9": 0.85, "10-99": 0.85, "100-999": null, "1000+": null },
  "gating_per_bin": { "1": ..., "2-9": ..., ... },     // per-bin MRR after gate, before fusion
  "gating_total":   0.3459,                             // gated full-test MRR
  "fusion_per_bin": { "1": ..., "2-9": ..., ... },     // per-bin MRR after gate + fusion
  "fusion_total":   0.3553,                             // headline AVG MRR (paper-reported)
  "counts":         { "1": 3597, "2-9": 6752, ... }
}
```

`null` alpha = bin not admitted by the gate; fusion is skipped and the
gate's bi-encoder ranking is kept.

## Provenance

| File                       | Paper-reported AVG MRR | Notes                                            |
|----------------------------|:----------------------:|--------------------------------------------------|
| `codex_m_masked.json`      | 0.3531                 |                                                  |
| `wn18rr_masked.json`       | 0.7256                 |                                                  |
| `wn18rr_unmasked.json`     | 0.7315                 |                                                  |
| `fb15k237_masked.json`     | 0.3655                 | max_length=224 retraining                        |
| `fb15k237_unmasked.json`   | 0.3687                 | max_length=224 retraining                        |
| `codex_m_unmasked.json`    | 0.3553                 | from `no_masked.ipynb` saved-output cell; paper Table 6 currently lists 0.3641 with no surviving source artifact, so 0.3553 is the verifiable reproducible value |
