"""End-to-end evaluation: score, select on validation, report on test.

Everything that is chosen is chosen on the validation split:
  1. the cross-encoder checkpoint (validation MRR of the reranker alone),
  2. one fusion weight per cardinality bin, together with the optional gate threshold T
     (validation MRR after fusion, over cfg["gate_threshold_grid"]; 10000 means no gate),
  3. a single global weight, used only as a baseline.
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
from .fusion import FusionRanks, alpha_grid, apply_fusion
from .gate import reranked_share, select_threshold
from .scoring import autocast_dtype, compute_bi_and_re_ranks, score_queries
from .stats import paired_test

CONFIG_NAMES = ("Bi-encoder", "Reranker", "Global fusion", "Per-bin fusion")
COMPARISONS = (("Per-bin fusion", "Bi-encoder"), ("Reranker", "Bi-encoder"),
               ("Global fusion", "Reranker"), ("Per-bin fusion", "Global fusion"))


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


def select_and_evaluate(valid_scored, test_scored, cfg):
    """Select the weights (and T) on validation and return the test ranks of every configuration."""
    grid = alpha_grid(float(cfg.get("alpha_grid_step", 0.05)))
    valid, test = FusionRanks(valid_scored, grid), FusionRanks(test_scored, grid)

    T, alphas, t_table, alphas_by_T = select_threshold(valid, cfg["gate_threshold_grid"])
    g = valid.select(lambda q: "all")["all"]
    print(f"  T = {T} (validation MRR {t_table[T]:.4f}) | per-bin weights {alphas} | global weight {g}")

    configs = {
        "Bi-encoder": test.ranks(lambda q: 0.0),
        "Reranker": test.ranks(lambda q: 1.0),
        "Global fusion": test.ranks(lambda q: g),
        "Per-bin fusion": apply_fusion(test, alphas, T),
    }
    # accuracy against computation for every threshold (weights re-selected on validation for each T)
    gate_on_test = {T_: {"reranked_share": reranked_share(test, T_),
                         "mrr": total_mrr(apply_fusion(test, alphas_by_T[T_], T_))}
                    for T_ in cfg["gate_threshold_grid"]}
    selection = {"gate_threshold_T": T, "alphas_per_bin": alphas, "global_alpha": g,
                 "validation_mrr_per_T": t_table, "gate_on_test": gate_on_test}
    return configs, selection


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

    test_scored = _score(test_records, checkpoint, collator, device, dtype, train_only)
    configs, selection = select_and_evaluate(valid_scored, test_scored, cfg)
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
    record = build_result_record(configs=configs, n_train_by_qid=n_train, selection=selection,
                                 significance=significance, meta=meta)

    seed_tag = f"seed_{seed}" if seed is not None else "no_seed"
    slug = cfg["dataset"].lower().replace("-", "")
    out_dir = repo_root / "results" / seed_tag
    write_results(record, out_dir / f"{slug}_{arm}.json")
    write_rank_csv(out_dir / f"{slug}_{arm}_ranks.csv", configs, test_scored, n_train)

    print("  test MRR: " + " | ".join(f"{c} {record['mrr'][c]:.4f}" for c in CONFIG_NAMES))
    print("  test H@1/3/10 (per-bin fusion): " + " / ".join(f"{v:.4f}" for v in record["hits"]["Per-bin fusion"].values()))
    for T, row in selection["gate_on_test"].items():
        print(f"  gate T={T:>5}: reranked {row['reranked_share']:.1%} of test queries, test MRR {row['mrr']:.4f}")
    print(f"  saved {out_dir / f'{slug}_{arm}.json'}")
    return record
