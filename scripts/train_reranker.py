"""Train the cross-encoder reranker. --masked enables the ablation arm."""

import argparse

from _common import REPO_ROOT, load_config
from src.training import train_reranker


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--masked", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    final = train_reranker(load_config(args.config), REPO_ROOT,
                           masked=args.masked, seed=args.seed)
    arm = "masked" if args.masked else "unmasked"
    print(f"next: scripts/evaluate.py --config {args.config} "
          f"--checkpoint {final} --arm {arm}")


if __name__ == "__main__":
    main()
