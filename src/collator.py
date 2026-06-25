import torch


def _pairs(features, entity_text):
    pairs, labels = [], []
    for f in features:
        cands = f["top_k_entity_ids"]
        gold = cands.index(f["gold_entity_id"])
        labels.append(gold)
        head = entity_text.get(f["head_id"], f["head_id"])
        query = f"{head} {f['relation']}"
        for cid in cands:
            pairs.append((query, entity_text.get(cid, cid)))
    return pairs, labels


def _reshape(tok, n, k, labels):
    L = tok["input_ids"].shape[-1]
    out = {
        "input_ids": tok["input_ids"].view(n, k, L),
        "attention_mask": tok["attention_mask"].view(n, k, L),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if "token_type_ids" in tok:
        out["token_type_ids"] = tok["token_type_ids"].view(n, k, L)
    return out


class UnmaskedKGCCollator:
    def __init__(self, tokenizer, entity_text_map, max_length=128):
        self.tokenizer = tokenizer
        self.entity_text = entity_text_map
        self.max_length = max_length

    def __call__(self, features):
        n = len(features)
        k = len(features[0]["top_k_entity_ids"])
        pairs, labels = _pairs(features, self.entity_text)
        tok = self.tokenizer(
            pairs, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return _reshape(tok, n, k, labels)


class MaskedKGCCollator:
    """Excludes known valid alternative tails (train-only) from the InfoNCE denominator."""

    def __init__(self, tokenizer, valid_tail_index, entity_text_map, max_length=128):
        self.tokenizer = tokenizer
        self.valid_tail_index = valid_tail_index
        self.entity_text = entity_text_map
        self.max_length = max_length

    def __call__(self, features):
        n = len(features)
        k = len(features[0]["top_k_entity_ids"])
        pairs, labels = _pairs(features, self.entity_text)

        mask = torch.zeros((n, k), dtype=torch.bool)
        for i, f in enumerate(features):
            known = self.valid_tail_index.get((f["head_id"], f["relation"]), set())
            for j, cid in enumerate(f["top_k_entity_ids"]):
                if cid in known and j != labels[i]:
                    mask[i, j] = True

        tok = self.tokenizer(
            pairs, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        out = _reshape(tok, n, k, labels)
        out["false_negative_mask"] = mask
        return out
