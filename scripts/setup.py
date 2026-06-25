"""Clone vendored/SimKGC and patch its AdamW import."""

import argparse

from _common import REPO_ROOT
from src.simkgc_runner import clone_simkgc, patch_adamw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--commit", default=None)
    args = p.parse_args()
    clone_simkgc(REPO_ROOT, commit=args.commit)
    patch_adamw(REPO_ROOT)
    print("done. next: scripts/train_biencoder.py --config configs/<dataset>.yaml")


if __name__ == "__main__":
    main()
