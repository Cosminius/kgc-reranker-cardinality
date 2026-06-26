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
    """Train-only (head, relation) -> {tail_ids}. Used by the mask and the gate;
    must NEVER read valid/test or the gate sees test labels."""
    index = defaultdict(set)
    for t in load_triples(simkgc_data_dir, "train"):
        index[(t["head_id"], t["relation"])].add(t["tail_id"])
        index[(t["tail_id"], f"inverse {t['relation']}")].add(t["head_id"])
    return dict(index)


def valid_query_map(triples):
    out = {}
    for i, t in enumerate(triples):
        out[f"valid_{i}_fwd"] = (t["head_id"], t["relation"], t["tail_id"])
        out[f"valid_{i}_bwd"] = (t["tail_id"], f"inverse {t['relation']}", t["head_id"])
    return out


def correct_n_to_train_only(scored, query_index, train_only_index):
    """Replace mining-time n_filtered (train+valid+test) with TRAIN-ONLY count."""
    missing = 0
    for qid, r in scored.items():
        if qid not in query_index:
            missing += 1
            continue
        h, rel, gold = query_index[qid]
        r["n_filtered"] = len(train_only_index.get((h, rel), set()) - {gold})
    return missing
