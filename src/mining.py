import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def build_filter_index(simkgc_data_dir):
    """All known valid tails per (head, relation), across train+valid+test.
    Used at mining time only; the gate uses a train-only index."""
    index = defaultdict(set)
    for split in ("train", "valid", "test"):
        with open(Path(simkgc_data_dir, f"{split}.txt.json"), encoding="utf-8") as f:
            for t in json.load(f):
                index[(t["head_id"], t["relation"])].add(t["tail_id"])
                index[(t["tail_id"], f"inverse {t['relation']}")].add(t["head_id"])
    return index


def expand_queries(triples):
    out = []
    for i, t in enumerate(triples):
        out.append({"query_id": f"{i}_fwd", "direction": "forward",
                    "head_id": t["head_id"], "relation": t["relation"],
                    "gold_entity_id": t["tail_id"]})
        out.append({"query_id": f"{i}_bwd", "direction": "backward",
                    "head_id": t["tail_id"], "relation": f"inverse {t['relation']}",
                    "gold_entity_id": t["head_id"]})
    return out


def _import_simkgc(simkgc_repo):
    if simkgc_repo not in sys.path:
        sys.path.insert(0, simkgc_repo)
    for m in list(sys.modules.keys()):
        if m.startswith(("config", "predict", "models", "doc", "dict_hub",
                         "utils", "triplet", "logger_config", "metric")):
            del sys.modules[m]


def mine_top_k(queries, biencoder_ckpt, simkgc_repo, simkgc_data_dir, K=50):
    _import_simkgc(simkgc_repo)
    sys.argv = [
        "m", "--pretrained-model", "bert-base-uncased",
        "--train-path", str(Path(simkgc_data_dir, "train.txt.json")),
        "--valid-path", str(Path(simkgc_data_dir, "valid.txt.json")),
        "--eval-model-path", biencoder_ckpt, "--is-test",
    ]
    from predict import BertPredictor
    from doc import Example
    from dict_hub import get_entity_dict

    predictor = BertPredictor()
    predictor.load(biencoder_ckpt, use_data_parallel=False)
    entities = get_entity_dict().entity_exs
    eid_to_idx = {e.entity_id: i for i, e in enumerate(entities)}
    idx_to_eid = {i: e.entity_id for i, e in enumerate(entities)}

    ent_emb = predictor.predict_by_entities(entities)
    if predictor.use_cuda:
        ent_emb = ent_emb.cuda()
    ent_emb = F.normalize(ent_emb, dim=-1)

    filter_idx = build_filter_index(simkgc_data_dir)

    examples = [Example(head_id=q["head_id"], relation=q["relation"],
                        tail_id=q["gold_entity_id"]) for q in queries]
    head_emb, _ = predictor.predict_by_examples(examples)
    if predictor.use_cuda:
        head_emb = head_emb.cuda()
    head_emb = F.normalize(head_emb, dim=-1)

    mask_idx = []
    gold_idx = []
    for q in queries:
        gold_idx.append(eid_to_idx.get(q["gold_entity_id"], -1))
        others = filter_idx.get((q["head_id"], q["relation"]), set()) - {q["gold_entity_id"]}
        mask_idx.append([eid_to_idx[x] for x in others if x in eid_to_idx])

    results = []
    chunk = 512
    with torch.no_grad():
        for s in range(0, head_emb.size(0), chunk):
            e = min(s + chunk, head_emb.size(0))
            sims = head_emb[s:e] @ ent_emb.T
            for j in range(e - s):
                idxs = mask_idx[s + j]
                if idxs:
                    sims[j].index_fill_(0, torch.tensor(idxs, dtype=torch.long, device=sims.device), -float("inf"))
            top_v, top_i = sims.topk(K, dim=-1)
            for j in range(e - s):
                qi = s + j
                q = queries[qi]
                gold = q["gold_entity_id"]
                top_eids = [idx_to_eid[int(top_i[j, k])] for k in range(K)]
                if gold in top_eids:
                    rank = top_eids.index(gold) + 1
                elif gold_idx[qi] >= 0:
                    rank = int((sims[j] > sims[j, gold_idx[qi]].item()).sum().item()) + 1
                else:
                    rank = -1
                results.append({
                    "query_id": q["query_id"], "direction": q["direction"],
                    "head_id": q["head_id"], "relation": q["relation"],
                    "gold_entity_id": gold,
                    "top_k_entity_ids": top_eids,
                    "top_k_scores": [float(top_v[j, k]) for k in range(K)],
                    "gold_rank": rank, "gold_in_topk": gold in top_eids,
                    "n_filtered": len(mask_idx[qi]),
                })
    return results


def _write_shards(records, out_dir, per_shard=5000):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(records), per_shard):
        idx = i // per_shard
        with open(out_dir / f"shard_{idx:04d}.jsonl", "w", encoding="utf-8") as f:
            for rec in records[i : i + per_shard]:
                f.write(json.dumps(rec) + "\n")


def mine_split(repo_root, cfg, split, K):
    simkgc_data_dir = repo_root / cfg["simkgc_data_dir"]
    mining_dir = repo_root / cfg["mining_dir"]
    biencoder = repo_root / cfg["biencoder_checkpoint"]
    simkgc_repo = repo_root / "vendored" / "SimKGC"

    if not biencoder.exists():
        raise SystemExit(f"Bi-encoder checkpoint not found: {biencoder}")
    if not simkgc_repo.exists():
        raise SystemExit("vendored/SimKGC missing; run scripts/setup.py")

    triples = json.loads(Path(simkgc_data_dir, f"{split}.txt.json").read_text(encoding="utf-8"))
    queries = expand_queries(triples)
    print(f"  {split}: {len(queries):,} queries, K={K}")

    records = mine_top_k(
        queries=queries, biencoder_ckpt=str(biencoder),
        simkgc_repo=str(simkgc_repo), simkgc_data_dir=str(simkgc_data_dir), K=K,
    )
    for r in records:
        r["query_id"] = f"{split}_{r['query_id']}"

    out_dir = mining_dir / f"{split}_k{K}"
    _write_shards(records, out_dir)
    print(f"  wrote {out_dir} ({len(records):,} records)")
