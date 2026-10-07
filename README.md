## CARF: Cardinality-Aware Reranking and Fusion

Code for the paper "Keeping It Simple: A Text-Only Reranking Pipeline with Cardinality-Aware
Score Fusion for Knowledge Graph Completion" (Cosmin Rosculet, Gheorghe Cosmin Silaghi).

CARF reranks the top-50 candidates of a SimKGC bi-encoder with a BERT-base cross-encoder and
mixes the two scores with one weight per cardinality bin, where cardinality is the number of
known answers of a query in the training graph. The cross-encoder helps queries with few known
answers but not queries with many, and the per-bin weights correct this. The cross-encoder
epoch and the weights are chosen on the validation set.

## Results

Filtered results on the test sets.

| Dataset | Model | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|---|
| WN18RR | SimKGC | 67.1 | 59.5 | 71.5 | 80.5 |
| WN18RR | CARF | **73.9** | **67.3** | **78.2** | **86.0** |
| FB15k-237 | SimKGC | 33.0 | 24.6 | 35.6 | 50.1 |
| FB15k-237 | CARF | **38.4** | **29.9** | **41.9** | **55.0** |
| CoDEx-M | SimKGC | 30.6 | 22.8 | 33.1 | 46.0 |
| CoDEx-M | CARF | **36.2** | **28.7** | **39.2** | **50.9** |

## Requirements

```
pip install -r requirements.txt
python scripts/setup.py
```

`setup.py` downloads [SimKGC](https://github.com/intfloat/SimKGC) into `vendored/`.

## Data

WN18RR and FB15k-237 come from [KG-BERT](https://github.com/yao8839836/kg-bert) and CoDEx-M
from [CoDEx](https://github.com/tsafavi/codex). Convert them with SimKGC's `preprocess.py` and
set `simkgc_data_dir` in `configs/<dataset>.yaml`.

## Usage

Train the SimKGC bi-encoder:
```
python scripts/train_biencoder.py --config configs/wn18rr.yaml --seed 0
```

Mine the top-50 candidates for train, validation and test:
```
python scripts/mine_candidates.py --config configs/wn18rr.yaml
```

Train the cross-encoder:
```
python scripts/train_reranker.py --config configs/wn18rr.yaml --seed 0
```

Select the epoch and the weights on validation and evaluate on test:
```
python scripts/evaluate.py --config configs/wn18rr.yaml --seed 0 \
    --checkpoint-dir checkpoints/reranker/WN18RR/unmasked
```

Results are written to `results/seed_0/`. The configs for CoDEx-M and FB15k-237 are in the
same folder.

## Notes

* `simkgc_task` is passed to SimKGC as `--task`. SimKGC's default (`wn18rr`) rewrites entity
  names in WordNet style, so keep `fb15k237` for the other datasets.
* `--masked` (training) and `--arm masked` (evaluation) remove the other known training answers
  from the loss. This is the filtered model discussed in the paper.
* `gate_threshold_grid` lets the pipeline skip the cross-encoder for high-cardinality queries.
  Validation selects no gate on all three datasets, so the paper does not use it.

## License

Apache 2.0. SimKGC is downloaded at setup and is not included here, see [NOTICE](NOTICE).
