# scripts/

Thin CLI wrappers; the real work lives in `src/`.

| Script               | What it does                                                                 |
|----------------------|------------------------------------------------------------------------------|
| `setup.py`           | Clone `vendored/SimKGC` and patch its AdamW import. Run once per machine.    |
| `train_biencoder.py` | Train the SimKGC bi-encoder for one dataset.                                 |
| `mine_candidates.py` | Bi-encoder candidate mining: test/valid (K=50) and train (K=200) shards.     |
| `train_reranker.py`  | Train the cross-encoder. Default: unmasked. `--masked` enables ablation arm. |
| `evaluate.py`        | Rerank, gate, alpha grid, fusion -> silaghi_format JSON.                     |
| `select_alpha.py`    | Standalone alpha grid search over a saved validation scoring dict.           |

Every script takes `--config configs/<dataset>.yaml` and an optional
`--seed <int>`. Omit `--seed` for true RNG; pass `--seed 0` to reproduce
the paper. Outputs route to `results/seed_<N>/` or `results/no_seed/`.
