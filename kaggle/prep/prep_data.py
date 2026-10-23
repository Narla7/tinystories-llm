#!/usr/bin/env python3
"""
Kaggle script kernel: build the tokenized corpus ONCE.

Everything it writes to /kaggle/working becomes the kernel's output, which you
then turn into a private Dataset ("Create Dataset from kernel output"). Every
training run after that mounts the dataset read-only and never downloads the
2.2 GB parquet again.

Settings this kernel needs: GPU off, internet ON.
Expected runtime: 10-20 minutes on 4 shared cores.
"""

import os
import subprocess
import sys

OUT_DIR = "/kaggle/working/tinystories-tokens"
VOCAB_SIZE = 4096


def resolve_repo() -> str:
    """Prefer an attached code dataset; otherwise clone. Keeps the kernel thin."""
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


def main() -> int:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tokenizers>=0.20"])
    repo = resolve_repo()
    os.chdir(repo)

    cmd = [
        sys.executable, "tokenize_data.py",
        "--out-dir", OUT_DIR,
        "--vocab-size", str(VOCAB_SIZE),
    ]
    print("[run]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

    meta = os.path.join(OUT_DIR, "meta.json")
    if os.path.exists(meta):
        print("\n[meta]\n" + open(meta).read())
    print(f"\n[done] artifacts in {OUT_DIR} -- now do 'Create Dataset from kernel output'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
