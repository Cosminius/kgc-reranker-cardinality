# src/

Importable library; scripts are thin wrappers around these modules.

| Module           | What it has                                                              |
|------------------|--------------------------------------------------------------------------|
| `loss`           | `masked_infonce_loss`                                                    |
| `collator`       | `UnmaskedKGCCollator`, `MaskedKGCCollator`                               |
| `trainer`        | `UnmaskedKGCRerankerTrainer`, `MaskedKGCRerankerTrainer`                 |
| `mining`         | `expand_queries`, `build_filter_index`, `mine_top_k`, `mine_split`       |
| `gate`           | `apply_cardinality_gate(test_scored, T)`                                 |
| `fusion`         | `select_alphas`, `apply_fusion`, `min_max_normalize`, `get_bin`          |
| `scoring`        | `score_queries`, `compute_bi_and_re_ranks`                               |
| `analysis`       | `per_bin_mrr`, `build_silaghi_format`, `write_silaghi_json`              |
| `datasets`       | entity text map, train-only valid-tails index, n_filtered correction    |
| `simkgc_runner`  | clone / patch / train_biencoder subprocess wrappers                     |
| `training`       | `train_reranker(cfg, masked, seed)`                                      |
| `pipeline`       | `evaluate_pipeline(cfg, checkpoint, arm, seed)` end-to-end eval          |
