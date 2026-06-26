"""Alpha grid search over a saved validation scoring dict."""

import argparse
import json
from pathlib import Path

from _common import REPO_ROOT  # noqa: F401
from src.fusion import select_alphas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scored", required=True)
    p.add_argument("--T", type=int, default=9)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    alphas = select_alphas(json.loads(Path(args.scored).read_text(encoding="utf-8")), threshold_T=args.T)
    print(f"T={args.T}  alphas: {alphas}")
    if args.out:
        Path(args.out).write_text(json.dumps({"T": args.T, "alphas": alphas}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
