import glob
import json
from collections import defaultdict
from pathlib import Path


def load_entity_text_map(simkgc_data_dir):
    out = {}
    for it in json.loads(Path(simkgc_data_dir, "entities.json").read_text(encoding="utf-8")):
        eid = it["entity_id"]
        out[eid] = f"{it.get('entity', eid)}: {it.get('entity_desc', '')}".strip(": ").strip()
    return out


def load_triples(simkgc_data_dir, split):
    return json.loads(Path(simkgc_data_dir, f"{split}.txt.json").read_text(encoding="utf-8"))


def build_train_only_valid_tails(simkgc_data_dir):
    """Train-only (head, relation) -> {tail_ids}, used for the cardinality and the masked loss."""
    index = defaultdict(set)
    for t in load_triples(simkgc_data_dir, "train"):
        index[(t["head_id"], t["relation"])].add(t["tail_id"])
        index[(t["tail_id"], f"inverse {t['relation']}")].add(t["head_id"])
    return dict(index)


def load_candidates(mining_dir, split, top_k):
    """Candidate lists written by scripts/mine_candidates.py."""
    shards = sorted(glob.glob(str(Path(mining_dir) / f"{split}_k{top_k}" / "shard_*.jsonl")))
    if not shards:
        raise SystemExit(f"No {split}_k{top_k} shards in {mining_dir}; run scripts/mine_candidates.py first")
    return [json.loads(line) for f in shards for line in open(f, encoding="utf-8")]


def correct_n_to_train_only(scored, query_index, train_only_index):
    """Replace the mining-time count of other answers (train+valid+test) with the train-only count."""
    for qid, r in scored.items():
        h, rel, gold = query_index[qid]
        r["n_filtered"] = len(train_only_index.get((h, rel), set()) - {gold})
