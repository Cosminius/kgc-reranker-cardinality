import pickle
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def results_prefix(cfg, seed, arm):
    """Common prefix of the files that scripts/evaluate.py writes for one dataset and arm."""
    seed_tag = f"seed_{seed}" if seed is not None else "no_seed"
    return REPO_ROOT / "results" / seed_tag / f"{cfg['dataset'].lower().replace('-', '')}_{arm}"


def load_scored(prefix, split):
    with open(f"{prefix}_{split}_scored.pkl", "rb") as f:
        return pickle.load(f)
