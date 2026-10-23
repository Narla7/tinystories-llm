#!/usr/bin/env python3
"""
One kernel, start to finish: tokenize the corpus if it isn't already here, then train.

Why this exists: two kernels (prep -> train) is the cleaner design, but it needs a
manual "Create Dataset from kernel output" click in between. This kernel removes
that step. It pays ~25 minutes of GPU quota doing CPU-only tokenization; that is a
worthless trade in the abstract and the right one if the alternative is a human
being in the loop.

If a tokenized corpus is attached under /kaggle/input it will be used and
tokenization is skipped entirely.

Settings: GPU T4 x2, internet ON.
"""

import json
import os
import subprocess
import sys

# Flip to a small number to validate the pipeline cheaply before the real run.
MAX_STEPS = None

VOCAB_SIZE = 4096
TOKENS_DIR = "/kaggle/working/tokens"
CONFIG = "configs/tiny25m.json"


def resolve_repo() -> str:
    for cand in ("/kaggle/input/tinystories-llm-code", "/kaggle/input/tinystories-llm"):
        if os.path.exists(os.path.join(cand, "train.py")):
            print(f"[repo] using attached dataset {cand}", flush=True)
            return cand
    url = os.environ.get("REPO_URL", "https://github.com/Narla7/tinystories-llm")
    dest = "/kaggle/working/repo"
    if not os.path.exists(os.path.join(dest, "train.py")):
        print(f"[repo] cloning {url}", flush=True)
        subprocess.check_call(["git", "clone", "--depth", "1", url, dest])
    return dest


def find_tokens() -> str | None:
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "train.bin" in files:
            print(f"[data] found tokenized corpus at {root}", flush=True)
            return root
    return None


def main() -> int:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tokenizers>=0.20"])
    repo = resolve_repo()
    os.chdir(repo)

    if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "", "-1"):
        print("[warn] no GPU visible -- check the accelerator setting", flush=True)

    data_dir = find_tokens()
    if data_dir is None:
        print("[data] no attached corpus; tokenizing now (~25 min, CPU)", flush=True)
        subprocess.check_call([
            sys.executable, "tokenize_data.py",
            "--out-dir", TOKENS_DIR,
            "--vocab-size", str(VOCAB_SIZE),
        ])
        data_dir = TOKENS_DIR
        meta_path = os.path.join(TOKENS_DIR, "meta.json")
        if os.path.exists(meta_path):
            print("[meta]\n" + open(meta_path).read(), flush=True)

    with open(CONFIG) as f:
        cfg = json.load(f)
    cfg["data_dir"] = data_dir
    cfg["run_dir"] = "/kaggle/working/runs/" + cfg["name"]
    if MAX_STEPS:
        cfg["train"]["max_steps"] = MAX_STEPS
        cfg["train"]["eval_interval"] = max(10, MAX_STEPS // 3)
        cfg["train"]["log_seconds"] = 10
    runtime_cfg = os.path.join(repo, "configs", "_runtime.json")
    with open(runtime_cfg, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[cfg] data_dir={data_dir}\n[cfg] run_dir={cfg['run_dir']}", flush=True)

    cmd = [sys.executable, "train.py", "--config", runtime_cfg]
    print("[run]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

    print("\n[done] checkpoints in", cfg["run_dir"], flush=True)
    print("[done] sample one with:  python sample.py --run-dir " + cfg["run_dir"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
