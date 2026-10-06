import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def autocast_dtype(prefer_bf16=True):
    """bf16 on GPUs with native support (compute capability >= 8, e.g. A100, RTX 30xx+),
    fp16 otherwise (e.g. T4, which only emulates bf16 and runs it far slower)."""
    if prefer_bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def score_queries(records, collator, model, device, batch_size=32, dtype=torch.bfloat16):
    out = {}
    in_topk = [r for r in records if r["gold_in_topk"]]
    fallback = [r for r in records if not r["gold_in_topk"]]

    for r in fallback:
        out[r["query_id"]] = {
            "source": "bi_encoder_fallback",
            "bi_rank": int(r["gold_rank"]),
            "direction": r["direction"],
            "n_filtered": int(r["n_filtered"]),
        }
    if not in_topk:
        return out

    ds = Dataset.from_list(in_topk)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collator)
    rd = list(ds)
    idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="score", leave=False):
            ii = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            tt = batch.get("token_type_ids", torch.zeros_like(ii)).to(device)
            lb = batch["labels"].numpy()
            B, K_, L = ii.shape
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(
                    input_ids=ii.view(-1, L),
                    attention_mask=am.view(-1, L),
                    token_type_ids=tt.view(-1, L),
                ).logits.view(B, K_).float().cpu().numpy()
            for i in range(B):
                r = rd[idx]
                out[r["query_id"]] = {
                    "source": "reranker",
                    "bi_scores": list(r["top_k_scores"]),
                    "rerank_scores": logits[i].tolist(),
                    "gold_idx_in_topk": int(lb[i]),
                    "direction": r["direction"],
                    "n_filtered": int(r["n_filtered"]),
                }
                idx += 1
    return out


def compute_bi_and_re_ranks(scored):
    bi, re = {}, {}
    for qid, r in scored.items():
        if r["source"] == "bi_encoder_fallback":
            bi[qid] = re[qid] = r["bi_rank"]
            continue
        b = np.asarray(r["bi_scores"])
        x = np.asarray(r["rerank_scores"])
        g = r["gold_idx_in_topk"]
        bi[qid] = int((b > b[g]).sum()) + 1
        re[qid] = int((x > x[g]).sum()) + 1
    return bi, re
