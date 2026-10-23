#!/usr/bin/env python3
"""
Kaggle script kernel: train the 25M model.

Settings: GPU T4 x2, internet ON, and the tokenized dataset (from prep_data)
attached. Expected runtime ~1.2 h.

The tokenized data is mounted read-only at /kaggle/input, so the config is
rewritten here to point at whatever path it actually landed on -- Kaggle nests
dataset directories one level deep, and guessing the path is the most common way
this fails.
"""

import json
import os
import subprocess
import sys

CONFIG_NAME = "configs/tiny25m.json"
# Flip to a small number for a cheap speed/plumbing check, None for the real run.
MAX_STEPS = 200


def resolve_repo() -> str:
    for cand in ("/kaggle/input/tinystories-llm-code", "/kaggle/input/tinystories-llm"):
        if os.path.exists(os.path.join(cand, "train.py")):
            print(f"[repo] using attached dataset {cand}")
            return cand
    url = os.environ.get("REPO_URL", "https://github.com/Narla7/tinystories-llm")
    dest = "/kaggle/working/repo"
    if not os.path.exists(os.path.join(dest, "train.py")):
        print(f"[repo] cloning {url}")
        subprocess.check_call(["git", "clone", "--depth", "1", url, dest])
    return dest


def resolve_data() -> str:
    """Find the directory containing train.bin, anywhere under /kaggle/input."""
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "train.bin" in files:
            print(f"[data] found tokenized corpus at {root}")
            return root
    raise SystemExit("train.bin not found under /kaggle/input -- attach the tokenized dataset")


def main() -> int:
    repo = resolve_repo()
    os.chdir(repo)

    with open(os.path.join(repo, CONFIG_NAME)) as f:
        cfg = json.load(f)

    cfg["data_dir"] = resolve_data()
    cfg["run_dir"] = "/kaggle/working/runs/" + cfg["name"]

    tok_cfg = os.path.join(repo, "configs", "_kaggle_runtime.json")
    with open(tok_cfg, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[cfg] data_dir={cfg['data_dir']}\n[cfg] run_dir={cfg['run_dir']}")

    cmd = [sys.executable, "train.py", "--config", tok_cfg]
    if MAX_STEPS:
        cmd += ["--max-steps", str(MAX_STEPS)]
    print("[run]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

    print("\n[done] checkpoints in", cfg["run_dir"])
    print("Copy best.pt out via the notebook Output pane, or push it to Hugging Face.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
