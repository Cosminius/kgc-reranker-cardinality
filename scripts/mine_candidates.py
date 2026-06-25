"""Mine top-K bi-encoder candidates for test/valid/train splits."""

import argparse

from _common import REPO_ROOT, load_config
from src.mining import mine_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--splits", nargs="+", default=["test", "valid", "train"],
                   choices=["test", "valid", "train"])
    p.add_argument("--K_test", type=int, default=50)
    p.add_argument("--K_train", type=int, default=200)
    args = p.parse_args()
    cfg = load_config(args.config)
    print(f"=== Mining {cfg['dataset']} ===")
    for s in args.splits:
        K = args.K_train if s == "train" else args.K_test
        mine_split(REPO_ROOT, cfg, split=s, K=K)


if __name__ == "__main__":
    main()
