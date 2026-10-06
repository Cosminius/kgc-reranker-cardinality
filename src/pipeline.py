"""End-to-end evaluation: score, select on validation, report on test.

Everything that is chosen is chosen on the validation split:
  1. the cross-encoder checkpoint (validation MRR of the reranker alone),
  2. the gate threshold T (validation MRR after the gate, over cfg["gate_threshold_grid"]),
  3. the fusion weight of every admitted cardinality bin (validation MRR of that bin).
The test split is scored once, with those choices, and is only used for reporting.
"""

import gc
import glob
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .analysis import build_result_record, total_mrr, write_rank_csv, write_results
from .collator import UnmaskedKGCCollator
from .datasets import build_train_only_valid_tails, correct_n_to_train_only, load_entity_text_map
from .fusion import apply_fusion, select_alphas
from .gate import apply_cardinality_gate, select_threshold
from .scoring import autocast_dtype, compute_bi_and_re_ranks, score_queries
from .stats import paired_test

CONFIG_NAMES = ("Bi-encoder", "Reranker", "+ Gate", "+ Fusion")
COMPARISONS = (("+ Fusion", "Bi-encoder"), ("Reranker", "Bi-encoder"), ("+ Gate", "Reranker"),
               ("+ Fusion", "+ Gate"), ("+ Fusion", "Reranker"))


def _set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_candidates(mining_dir, split, top_k):
    shards = sorted(glob.glob(str(Path(mining_dir) / f"{split}_k{top_k}" / "shard_*.jsonl")))
    if not shards:
        raise SystemExit(f"No {split}_k{top_k} shards in {mining_dir}; run scripts/mine_candidates.py first")
    return [json.loads(line) for f in shards for line in open(f, encoding="utf-8")]


def _score(records, checkpoint, collator, device, dtype, train_only):
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint)).to(device).eval()
    scored = score_queries(records, collator, model, device, dtype=dtype)
    del model
    torch.cuda.empty_cache(); gc.collect()
    qmap = {r["query_id"]: (r["head_id"], r["relation"], r["gold_entity_id"]) for r in records}
    correct_n_to_train_only(scored, qmap, train_only)   # cardinality from training triples only
    return scored


def evaluate_pipeline(cfg, repo_root, checkpoints, arm, seed):
    _set_seed(seed)
    data_dir = repo_root / cfg["simkgc_data_dir"]
    mining_dir = repo_root / cfg["mining_dir"]
    top_k = int(cfg.get("top_k", 50))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = autocast_dtype(cfg.get("bf16", True))
    print(f"=== Evaluate {cfg['dataset']} ({arm}), {len(checkpoints)} candidate checkpoint(s), autocast {dtype} ===")

    valid_records = load_candidates(mining_dir, "valid", top_k)
    test_records = load_candidates(mining_dir, "test", top_k)
    train_only = build_train_only_valid_tails(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg["pretrained_model"])
    collator = UnmaskedKGCCollator(tokenizer=tokenizer, entity_text_map=load_entity_text_map(data_dir),
                                   max_length=cfg["max_length"])

    # 1) checkpoint: best validation MRR of the reranker alone
    best = None
    checkpoint_mrr = {}
    for ckpt in checkpoints:
        scored = _score(valid_records, ckpt, collator, device, dtype, train_only)
        _, rerank = compute_bi_and_re_ranks(scored)
        checkpoint_mrr[str(ckpt)] = total_mrr(rerank)
        print(f"  {Path(ckpt).name}: validation MRR (reranker) = {checkpoint_mrr[str(ckpt)]:.4f}")
        if best is None or checkpoint_mrr[str(ckpt)] > checkpoint_mrr[str(best[0])]:
            best = (ckpt, scored)
    checkpoint, valid_scored = best
    print(f"  selected checkpoint: {checkpoint}")

    # 2) gate threshold and 3) fusion weights, both on validation
    T, t_table = select_threshold(valid_scored, cfg["gate_threshold_grid"])
    step = float(cfg.get("alpha_grid_step", 0.05))
    alphas = select_alphas(valid_scored, threshold_T=T, grid=np.round(np.arange(0.0, 1.0 + 1e-9, step), 10))
    print(f"  gate T = {T} (validation MRR {t_table[T]:.4f}) | alphas = {alphas}")

    # test, scored once
    test_scored = _score(test_records, checkpoint, collator, device, dtype, train_only)
    bi_r, rr_r = compute_bi_and_re_ranks(test_scored)
    gate_r = apply_cardinality_gate(test_scored, threshold_T=T)
    fuse_r = apply_fusion(test_scored, alphas, threshold_T=T, gated_ranks=gate_r)
    configs = dict(zip(CONFIG_NAMES, (bi_r, rr_r, gate_r, fuse_r)))
    n_train = {qid: r["n_filtered"] for qid, r in test_scored.items()}

    qids = sorted(test_scored)
    rr = {c: np.array([1.0 / ranks[q] if ranks[q] > 0 else 0.0 for q in qids]) for c, ranks in configs.items()}
    significance = {f"{a} vs {b}": paired_test(rr[a], rr[b], seed=0) for a, b in COMPARISONS}

    meta = {
        "dataset": cfg["dataset"], "arm": arm, "seed": seed,
        "biencoder_checkpoint": cfg["biencoder_checkpoint"],
        "reranker_checkpoint": str(checkpoint),
        "reranker_checkpoint_validation_mrr": checkpoint_mrr,
        "top_k": top_k, "max_length": cfg["max_length"],
        "stack": (f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}, "
                  f"torch {torch.__version__}, autocast {str(dtype).replace('torch.', '')}"),
    }
    record = build_result_record(configs=configs, n_train_by_qid=n_train, threshold_T=T,
                                 threshold_table=t_table, alphas=alphas,
                                 significance=significance, meta=meta)

    seed_tag = f"seed_{seed}" if seed is not None else "no_seed"
    slug = cfg["dataset"].lower().replace("-", "")
    out_dir = repo_root / "results" / seed_tag
    write_results(record, out_dir / f"{slug}_{arm}.json")
    write_rank_csv(out_dir / f"{slug}_{arm}_ranks.csv", configs, test_scored, n_train)

    print("  test MRR: " + " | ".join(f"{c} {record['mrr'][c]:.4f}" for c in CONFIG_NAMES))
    print("  test H@1/3/10 (+ Fusion): " + " / ".join(f"{v:.4f}" for v in record["hits"]["+ Fusion"].values()))
    print(f"  saved {out_dir / f'{slug}_{arm}.json'}")
    return record
