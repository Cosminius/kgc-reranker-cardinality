# `configs/`

One YAML per dataset, loaded by every script via `--config`.

## Where the values come from

Cross-encoder hyperparameters (`learning_rate`, `weight_decay`, `warmup_ratio`,
`max_grad_norm`, `bf16`, `lr_scheduler_type=linear`) are the SimKGC-derived
defaults that the paper used unchanged across all three datasets.

Per-dataset choices vary by sequence-length distribution and GPU memory:

| Dataset    | max_length | per-step batch | grad_accum | epochs | why                                                |
|------------|-----------:|---------------:|-----------:|-------:|----------------------------------------------------|
| CoDEx-M    | 128        | 16             | 2          | 4      | short query-candidate pairs (median 28 tokens)     |
| WN18RR     | 128        | 16             | 2          | 4      | mostly short; ~0.37% truncated                     |
| FB15k-237  | 224        | 8              | 4          | 3      | richer entity descriptions; full distribution fits at 224 |

Effective batch size = 32 on all three.

Gate threshold `gate_threshold_T: 9` is the validation-selected value
on every dataset.

Per-bin alphas (`alpha_bins_unmasked`, `alpha_bins_masked`) are the
published reference values. `scripts/evaluate.py` re-derives them on
validation at run time, so swapping in your own dataset doesn't require
hand-tuning the configs.

## Adding a new dataset

1. Copy `codex-m.yaml`, update `dataset`, `simkgc_data_dir`, `mining_dir`,
   `output_dir`.
2. Set `max_length` based on your tokenizer's entity description length.
3. Adjust batch / grad-accum to fit your GPU; aim for effective batch 32.
4. `T = 9` is a good default; re-tune on validation if your cardinality
   distribution is unusual.
