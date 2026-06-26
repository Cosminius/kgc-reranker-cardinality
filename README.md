# Cardinality-Gated Reranking for Text-Based KGC

Code for *Cardinality-Gated Reranking for Text-Based Knowledge Graph Completion*
(Cosmin Rosculet, Babes-Bolyai University).

## Pipeline

```
SimKGC bi-encoder -> top-K mining -> BERT cross-encoder
                                          |
                                          v
                              cardinality gate (T=9)
                                          |
                                          v
                              per-bin score fusion -> ranked tail
```

The two new ideas: the **cardinality gate** (rerank only when the
(head, relation) has at most T+1 valid tails in the training graph;
otherwise keep the bi-encoder ranking) and **per-bin score fusion**
(mix rerank + bi with a per-cardinality-bin alpha selected on validation).

## Setup

```bash
python scripts/setup.py   # clones vendored/SimKGC and patches its AdamW import
```

## Data

Each `configs/<dataset>.yaml` has a `simkgc_data_dir` field. Point it at a
folder containing the SimKGC-preprocessed JSON files for that dataset:

```
<simkgc_data_dir>/
  train.txt.json
  valid.txt.json
  test.txt.json
  entities.json
```

This repo does not download datasets. CoDEx-M comes from
`github.com/tsafavi/codex`; FB15k-237 and WN18RR from
`github.com/yao8839836/kg-bert/tree/master/data`. Run them through
`vendored/SimKGC/preprocess.py` to get the JSON layout above.

## Run

```bash
python scripts/train_biencoder.py --config configs/codex-m.yaml
python scripts/mine_candidates.py --config configs/codex-m.yaml
python scripts/train_reranker.py  --config configs/codex-m.yaml
python scripts/evaluate.py        --config configs/codex-m.yaml \
                                  --checkpoint checkpoints/reranker/CoDEx-M/unmasked/final
```

Output: `results/no_seed/codex_m_unmasked.json` (gating_total, fusion_total,
per-bin MRR, selected alphas). For the masked ablation arm, add `--masked`
to step 3 and `--arm masked` to step 4.

## Folder layout

```
configs/   per-dataset YAML
src/       library
scripts/   thin CLI wrappers around src/
results/   per-seed JSONs (silaghi_format)
vendored/  cloned by setup.py (gitignored)
```

## Seed and reproducibility

`--seed <int>` is optional on every script. Omit for true RNG (production
default). Pass `--seed 0` to reproduce the paper numbers; results go to
`results/seed_0/`.

The paper reports single-seed numbers by design. A meaningful mean +/- CI
needs at least ~30 runs; with 3 the deviation is statistically hollow.
The repo is set up for any seed sweep you want (`results/seed_<N>/`),
but the published numbers are single-seed by intent.

The paper numbers in `results/seed_0/*.json` come from a single stack:
RTX 5070, PyTorch 2.8.0 + CUDA 12.9, transformers 5.12.1. Two consecutive
runs of `scripts/evaluate.py --seed 0` produce bitwise-identical output.
A different PyTorch/CUDA combination will drift by a few units in the 4th
MRR decimal due to bf16 numerics, but the conclusions (gate + fusion
improves over the bi-encoder on all three datasets; masked vs unmasked
InfoNCE gives no consistent winner) don't change.

### Known caveat: alpha overfits validation

The per-bin alpha grid search picks alpha near 1.0 on all three datasets,
which is where validation MRR peaks. Test MRR peaks lower (around 0.75-0.80
at bin 2-9). The alphas in the JSONs and Tables 2/3/4/5 are the grid-selected
ones; they're not test-optimal. To see the val-vs-test alpha curve yourself:

    python scripts/diagnose_alpha.py --config configs/codex-m.yaml --checkpoint <ckpt>

## Built on SimKGC vs mine

- **SimKGC** (cloned at setup, not redistributed): the bi-encoder model,
  its training loop, and tokenization. Lives under `vendored/SimKGC/`.
- **Mine**: everything under `src/` and `scripts/`. The cross-encoder
  reranker, the cardinality gate, the per-bin score fusion, the mining
  wrapper, and the analysis/JSON output.

Upstream SimKGC has no LICENSE file. We do not redistribute it; please
respect the original authors' rights. See [NOTICE](NOTICE).

## License

Apache License 2.0 - see [LICENSE](LICENSE).

Advised by Prof. Gheorghe Cosmin Silaghi.
