"""Train the SimKGC bi-encoder."""

import argparse

from _common import REPO_ROOT, load_config
from src.simkgc_runner import run_train_biencoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    ckpt = run_train_biencoder(REPO_ROOT, load_config(args.config), seed=args.seed)
    print(f"checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
