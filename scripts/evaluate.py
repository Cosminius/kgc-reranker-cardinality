"""End-to-end pipeline: rerank, gate, alpha, fusion, write JSON."""

import argparse
from pathlib import Path

from _common import REPO_ROOT, load_config
from src.pipeline import evaluate_pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--arm", choices=["unmasked", "masked"], default="unmasked")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    evaluate_pipeline(
        cfg=load_config(args.config), repo_root=REPO_ROOT,
        checkpoint=Path(args.checkpoint), arm=args.arm, seed=args.seed,
    )


if __name__ == "__main__":
    main()
