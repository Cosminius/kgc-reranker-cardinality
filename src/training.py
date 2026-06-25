import gc
import glob
import os
import random

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

from .collator import MaskedKGCCollator, UnmaskedKGCCollator
from .datasets import build_train_only_valid_tails, load_entity_text_map
from .trainer import MaskedKGCRerankerTrainer, UnmaskedKGCRerankerTrainer


def _set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_shards(mining_dir):
    shards = sorted(glob.glob(str(mining_dir / "train_k50" / "shard_*.jsonl")))
    if not shards:
        shards = sorted(glob.glob(str(mining_dir / "train_k200" / "shard_*.jsonl")))
    if not shards:
        raise SystemExit(f"No training shards in {mining_dir}; run mine_candidates first")
    raw = Dataset.from_json(shards)
    return raw.filter(lambda ex: ex["gold_in_topk"] is True, num_proc=8)


def train_reranker(cfg, repo_root, masked, seed):
    _set_seed(seed)

    simkgc_data_dir = repo_root / cfg["simkgc_data_dir"]
    mining_dir = repo_root / cfg["mining_dir"]
    arm = "masked" if masked else "unmasked"
    output_dir = repo_root / cfg["output_dir"] / arm
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Train {arm} reranker on {cfg['dataset']} (seed={seed}) ===")
    train_dataset = _load_shards(mining_dir)
    print(f"  recoverable: {len(train_dataset):,}")

    entity_text = load_entity_text_map(simkgc_data_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg["pretrained_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["pretrained_model"], num_labels=1, attn_implementation="sdpa"
    )

    if masked:
        collator = MaskedKGCCollator(
            tokenizer=tokenizer,
            valid_tail_index=build_train_only_valid_tails(simkgc_data_dir),
            entity_text_map=entity_text,
            max_length=cfg["max_length"],
        )
        trainer_cls = MaskedKGCRerankerTrainer
    else:
        collator = UnmaskedKGCCollator(
            tokenizer=tokenizer, entity_text_map=entity_text,
            max_length=cfg["max_length"],
        )
        trainer_cls = UnmaskedKGCRerankerTrainer

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache(); gc.collect()

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        num_train_epochs=int(cfg["num_train_epochs"]),
        weight_decay=float(cfg["weight_decay"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        max_grad_norm=float(cfg["max_grad_norm"]),
        lr_scheduler_type="linear",
        bf16=bool(cfg["bf16"]), tf32=True, torch_compile=True,
        dataloader_num_workers=4, dataloader_pin_memory=True,
        remove_unused_columns=False,
        logging_steps=50,
        save_strategy="epoch", save_total_limit=None,
        report_to="none",
        seed=seed if seed is not None else 42,
    )

    trainer = trainer_cls(
        model=model, args=args, train_dataset=train_dataset, data_collator=collator
    )
    trainer.train()

    final = output_dir / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    print(f"saved {final}")
    return final
