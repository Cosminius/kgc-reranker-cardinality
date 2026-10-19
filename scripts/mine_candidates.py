"""Mine top-K bi-encoder candidates: unfiltered for train (the cross-encoder's training lists),
filtered for valid and test (evaluation)."""

import argparse

from _common import REPO_ROOT, load_config
from src.mining import mine_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--splits", nargs="+", default=["test", "valid", "train"],
                   choices=["test", "valid", "train"])
    p.add_argument("--top-k", type=int, default=None, help="default: top_k from the config (50)")
    args = p.parse_args()
    cfg = load_config(args.config)
    K = args.top_k or int(cfg.get("top_k", 50))
    print(f"=== Mining {cfg['dataset']} (SimKGC task: {cfg['simkgc_task']}, K={K}) ===")
    for s in args.splits:
        mine_split(REPO_ROOT, cfg, split=s, K=K)


if __name__ == "__main__":
    main()
