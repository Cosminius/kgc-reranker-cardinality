"""End-to-end pipeline: mine + score + gate + alpha + fusion.

`prepare_scored` returns the per-query score dicts for valid + test, after
correcting n_filtered to the train-only count. It's the heavy work and is
shared with diagnose_alpha.py.

`evaluate_pipeline` adds the gate, alpha grid, fusion, and writes the
silaghi_format JSON.
"""

import gc
import glob
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .analysis import build_silaghi_format, write_silaghi_json
from .collator import UnmaskedKGCCollator
from .datasets import (
    build_train_only_valid_tails, correct_n_to_train_only,
    load_entity_text_map, load_triples, valid_query_map,
)
from .fusion import apply_fusion, select_alphas
from .gate import apply_cardinality_gate
from .mining import expand_queries, mine_top_k
from .scoring import score_queries


def _set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _mine_valid_live(simkgc_data_dir, biencoder, simkgc_repo):
    """Mine validation queries through the bi-encoder (live, so the gate
    sees the same bi-encoder state as scoring)."""
    valid_triples = load_triples(simkgc_data_dir, "valid")
    valid_queries = expand_queries(valid_triples)
    for q in valid_queries:
        q["query_id"] = f"valid_{q['query_id']}"
    valid_records = mine_top_k(
        queries=valid_queries, biencoder_ckpt=str(biencoder),
        simkgc_repo=str(simkgc_repo), simkgc_data_dir=str(simkgc_data_dir), K=50,
    )
    return valid_triples, valid_records


def _load_test_records(mining_dir):
    test_shards = sorted(glob.glob(str(mining_dir / "test_k50" / "shard_*.jsonl")))
    if not test_shards:
        raise SystemExit(f"No test_k50 shards in {mining_dir}; run mine_candidates first")
    return list(Dataset.from_json(test_shards))


def prepare_scored(cfg, repo_root, checkpoint, seed=None):
    """Mine valid, score valid + test with the cross-encoder, correct
    n_filtered to TRAIN-ONLY. Returns (valid_scored, test_scored, test_records, T).

    Shared by evaluate_pipeline and scripts/diagnose_alpha.py.
    """
    _set_seed(seed)
    simkgc_data_dir = repo_root / cfg["simkgc_data_dir"]
    mining_dir = repo_root / cfg["mining_dir"]
    biencoder = repo_root / cfg["biencoder_checkpoint"]
    simkgc_repo = repo_root / "vendored" / "SimKGC"
    T = int(cfg["gate_threshold_T"])
    if not simkgc_repo.exists():
        raise SystemExit("vendored/SimKGC missing; run scripts/setup.py first")

    entity_text = load_entity_text_map(simkgc_data_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg["pretrained_model"])
    collator = UnmaskedKGCCollator(
        tokenizer=tokenizer, entity_text_map=entity_text,
        max_length=cfg["max_length"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("  mining validation...")
    valid_triples, valid_records = _mine_valid_live(simkgc_data_dir, biencoder, simkgc_repo)
    torch.cuda.empty_cache(); gc.collect()

    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint)).to(device).eval()
    print("  scoring validation...")
    valid_scored = score_queries(valid_records, collator, model, device)
    test_records = _load_test_records(mining_dir)
    print("  scoring test...")
    test_scored = score_queries(test_records, collator, model, device)
    del model
    torch.cuda.empty_cache(); gc.collect()

    train_only = build_train_only_valid_tails(simkgc_data_dir)
    valid_qmap = valid_query_map(valid_triples)
    test_qmap = {r["query_id"]: (r["head_id"], r["relation"], r["gold_entity_id"])
                 for r in test_records}
    correct_n_to_train_only(valid_scored, valid_qmap, train_only)
    correct_n_to_train_only(test_scored, test_qmap, train_only)

    return valid_scored, test_scored, test_records, T


def evaluate_pipeline(cfg, repo_root, checkpoint, arm, seed):
    print(f"=== Evaluate {cfg['dataset']} ({arm}) at T={cfg['gate_threshold_T']} ===")
    valid_scored, test_scored, _, T = prepare_scored(cfg, repo_root, checkpoint, seed)

    gated = apply_cardinality_gate(test_scored, threshold_T=T)
    alphas = select_alphas(valid_scored, threshold_T=T)
    print(f"  alphas: {alphas}")
    fused = apply_fusion(test_scored, alphas, threshold_T=T, gated_ranks=gated)

    n_train = {qid: r["n_filtered"] for qid, r in test_scored.items()}
    record = build_silaghi_format(
        threshold_T=T, alphas=alphas,
        gated_ranks=gated, fused_ranks=fused,
        n_train_by_qid=n_train,
    )
    record["dataset"] = cfg["dataset"]
    record["arm"] = arm
    record["seed"] = seed
    record["checkpoint"] = str(checkpoint)

    seed_tag = f"seed_{seed}" if seed is not None else "no_seed"
    slug = cfg["dataset"].lower().replace("-", "_")
    out_path = repo_root / "results" / seed_tag / f"{slug}_{arm}.json"
    write_silaghi_json(record, out_path)

    print(f"  gating={record['gating_total']:.4f}  fusion={record['fusion_total']:.4f}")
    print(f"  saved {out_path}")
    return record
