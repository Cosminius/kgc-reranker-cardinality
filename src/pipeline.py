"""End-to-end evaluation: score, select on validation, report on test.

Everything that is chosen is chosen on the validation split:
  1. the cross-encoder checkpoint (validation MRR of the reranker alone),
  2. one fusion weight per cardinality bin (CARF),
  3. a single global weight, used only as a baseline.
The test split is scored once, with those choices, and is only used for reporting.
"""

import gc
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .analysis import build_result_record, total_mrr, write_rank_csv, write_results
from .collator import UnmaskedKGCCollator
from .datasets import (build_train_only_valid_tails, correct_n_to_train_only, load_candidates,
                       load_entity_text_map)
from .evaluation import CONFIG_NAMES, select_and_evaluate, significance
from .scoring import autocast_dtype, compute_bi_and_re_ranks, score_queries


def _set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    # checkpoint: best validation MRR of the reranker alone
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

    test_scored = _score(test_records, checkpoint, collator, device, dtype, train_only)
    test, configs, selection = select_and_evaluate(valid_scored, test_scored,
                                                   float(cfg.get("alpha_grid_step", 0.05)))
    print(f"  per-bin weights {selection['alphas_per_bin']} | global weight {selection['global_alpha']}")
    n_train = {qid: r["n_filtered"] for qid, r in test_scored.items()}

    meta = {
        "dataset": cfg["dataset"], "arm": arm, "seed": seed,
        "biencoder_checkpoint": cfg["biencoder_checkpoint"],
        "reranker_checkpoint": str(checkpoint),
        "reranker_checkpoint_validation_mrr": checkpoint_mrr,
        "top_k": top_k, "max_length": cfg["max_length"],
        "stack": (f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}, "
                  f"torch {torch.__version__}, autocast {str(dtype).replace('torch.', '')}"),
    }
    record = build_result_record(configs=configs, n_train_by_qid=n_train, selection=selection,
                                 significance=significance(test, configs), meta=meta)

    seed_tag = f"seed_{seed}" if seed is not None else "no_seed"
    slug = cfg["dataset"].lower().replace("-", "")
    out_dir = repo_root / "results" / seed_tag
    write_results(record, out_dir / f"{slug}_{arm}.json")
    write_rank_csv(out_dir / f"{slug}_{arm}_ranks.csv", configs, test_scored, n_train)
    for split, scored in (("valid", valid_scored), ("test", test_scored)):
        with open(out_dir / f"{slug}_{arm}_{split}_scored.pkl", "wb") as f:
            pickle.dump(scored, f)

    print("  test MRR: " + " | ".join(f"{c} {record['mrr'][c]:.4f}" for c in CONFIG_NAMES))
    print("  test H@1/3/10 (per-bin fusion): " + " / ".join(f"{v:.4f}" for v in record["hits"]["Per-bin fusion"].values()))
    print(f"  saved {out_dir / f'{slug}_{arm}.json'}")
    return record
