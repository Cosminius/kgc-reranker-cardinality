## Cardinality-Gated Reranking for Knowledge Graph Completion

Code for the paper "Cardinality-Gated Reranking: Making Bi-Encoder Knowledge Graph
Completion Competitive at Low Cost" (Cosmin Rosculet, Gheorghe Cosmin Silaghi).

We rerank the top-50 candidates of a SimKGC bi-encoder with a BERT-base cross-encoder.
The reranker helps on queries with few known answers and hurts on queries with many, so we
mix the scores of the two models with one weight per cardinality bin (the number of known
answers of the query in the training graph). Optionally, queries with more than `T` known
answers skip the cross-encoder to save computation. The cross-encoder epoch, the weights and
`T` are chosen on the validation set (the weights and `T` together).

## Results

Filtered test results. Every gain over SimKGC is significant at p < 0.001 (paired
randomisation test). Validation selects no gate on all three datasets.

| Dataset | Model | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|---|
| CoDEx-M | SimKGC | 30.6 | 22.8 | 33.1 | 46.0 |
| CoDEx-M | + reranking and fusion | **36.2** | **28.7** | **39.2** | **50.9** |
| WN18RR | SimKGC | 67.1 | 59.5 | 71.5 | 80.5 |
| WN18RR | + reranking and fusion | **73.9** | **67.3** | **78.2** | **86.0** |
| FB15k-237 | SimKGC | 33.0 | 24.6 | 35.6 | 50.1 |
| FB15k-237 | + reranking and fusion | **38.4** | **29.9** | **41.9** | **55.0** |

## Requirements

* python>=3.9
* torch>=2.7
* transformers>=5.0

```
pip install -r requirements.txt
python scripts/setup.py
```

`setup.py` downloads [SimKGC](https://github.com/intfloat/SimKGC) into `vendored/`.
Any CUDA GPU works (bf16 on recent GPUs, fp16 on older ones such as the T4).

## Data

For WN18RR and FB15k-237 we use the files from [KG-BERT](https://github.com/yao8839836/kg-bert),
for CoDEx-M the files from [CoDEx](https://github.com/tsafavi/codex). Convert them with
SimKGC's `preprocess.py` and set `simkgc_data_dir` in `configs/<dataset>.yaml`.

## How to Run

Each dataset has a config in `configs/` (`codex-m.yaml`, `wn18rr.yaml`, `fb15k-237.yaml`).

Step 1, train the SimKGC bi-encoder
```
python scripts/train_biencoder.py --config configs/wn18rr.yaml --seed 0
```

Step 2, mine the top-50 candidates for train, validation and test
```
python scripts/mine_candidates.py --config configs/wn18rr.yaml
```

Step 3, train the cross-encoder (one checkpoint per epoch)
```
python scripts/train_reranker.py --config configs/wn18rr.yaml --seed 0
```

Step 4, select the epoch, the weights and `T` on validation and evaluate on test
```
python scripts/evaluate.py --config configs/wn18rr.yaml --seed 0 \
    --checkpoint-dir checkpoints/reranker/WN18RR/unmasked
```

Step 4 writes `results/seed_0/<dataset>_unmasked.json` (MRR and Hits@k per configuration and
cardinality bin, the selected values, the test MRR for every gate threshold, the significance
tests) and a CSV with the rank of every test query.

The paper uses these SimKGC checkpoints: `checkpoint_epoch7.mdl` for CoDEx-M (trained with
SimKGC's FB15k-237 settings), `checkpoint_epoch48.mdl` for WN18RR and `model_best.mdl` for
FB15k-237 (`biencoder_checkpoint` in the configs).

## Notes

* `simkgc_task` in the configs is passed to SimKGC as `--task`. Keep it: SimKGC's default
  (`wn18rr`) rewrites entity names in WordNet style and erases the names of other datasets.
* `--masked` in step 3 and `--arm masked` in step 4 train and evaluate a masked InfoNCE
  variant. It is an extra experiment and is not used in the paper.

## License

Apache 2.0. SimKGC is downloaded at setup and not included here, see [NOTICE](NOTICE).
