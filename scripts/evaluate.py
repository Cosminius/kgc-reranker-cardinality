"""Evaluate one dataset: select checkpoint, gate threshold and fusion weights on validation,
then report MRR, Hits@k and paired significance tests on test.

    python scripts/evaluate.py --config configs/codex-m.yaml \
        --checkpoint-dir checkpoints/reranker/CoDEx-M/unmasked --seed 0
"""

import argparse
import re
from pathlib import Path

from _common import REPO_ROOT, load_config
from src.pipeline import evaluate_pipeline


def epoch_checkpoints(directory):
    """checkpoint-<step> sub-folders written by the trainer (one per epoch), in step order;
    falls back to the folder itself if it is a single saved model."""
    d = Path(directory)
    ckpts = sorted(d.glob("checkpoint-*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0))
    return ckpts or [d]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint-dir", help="training output folder; every checkpoint-* in it is a candidate")
    g.add_argument("--checkpoint", nargs="+", help="one or more saved cross-encoder folders")
    p.add_argument("--arm", choices=["unmasked", "masked"], default="unmasked")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    ckpts = epoch_checkpoints(args.checkpoint_dir) if args.checkpoint_dir else [Path(c) for c in args.checkpoint]
    evaluate_pipeline(cfg=load_config(args.config), repo_root=REPO_ROOT,
                      checkpoints=ckpts, arm=args.arm, seed=args.seed)


if __name__ == "__main__":
    main()
