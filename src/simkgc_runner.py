import os
import shutil
import subprocess
import sys
from pathlib import Path

SIMKGC_URL = "https://github.com/intfloat/SimKGC.git"

TASK = {"CoDEx-M": "codex-m", "FB15k-237": "fb15k237", "WN18RR": "wn18rr"}


def vendored(repo_root):
    return repo_root / "vendored" / "SimKGC"


def clone_simkgc(repo_root, commit=None):
    target = vendored(repo_root)
    if target.exists():
        print(f"vendored/SimKGC already at {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", SIMKGC_URL, str(target)], check=True)
    if commit:
        subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", commit], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", commit], check=True)
    return target


def patch_adamw(repo_root):
    """transformers >= 4 dropped AdamW; the upstream `from transformers import AdamW`
    must become `from torch.optim import AdamW`."""
    trainer = vendored(repo_root) / "trainer.py"
    if not trainer.exists():
        return
    text = trainer.read_text(encoding="utf-8")
    old = "from transformers import AdamW"
    new = "from torch.optim import AdamW"
    if old not in text:
        return  # already patched or upstream changed
    backup = trainer.with_suffix(".py.bak")
    shutil.copy2(trainer, backup)
    trainer.write_text(text.replace(old, new), encoding="utf-8")
    print(f"patched {trainer}")


def run_train_biencoder(repo_root, cfg, seed):
    simkgc = vendored(repo_root)
    if not simkgc.exists():
        raise SystemExit("vendored/SimKGC missing; run scripts/setup.py first")

    data_dir = repo_root / cfg["simkgc_data_dir"]
    ckpt_dir = repo_root / Path(cfg["biencoder_checkpoint"]).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(simkgc / "main.py"),
        "--task", TASK[cfg["dataset"]],
        "--train-path", str(data_dir / "train.txt.json"),
        "--valid-path", str(data_dir / "valid.txt.json"),
        "--model-dir", str(ckpt_dir),
        "--pretrained-model", cfg["pretrained_model"],
        "--lr", str(cfg["biencoder_lr"]),
        "--batch-size", str(cfg["biencoder_batch_size"]),
        "--epochs", str(cfg["biencoder_epochs"]),
        "--use-amp", "--use-link-graph",
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(simkgc) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=str(simkgc), env=env)
    return ckpt_dir
