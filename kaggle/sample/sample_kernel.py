#!/usr/bin/env python3
"""
Print samples from the trained checkpoint, straight into the kernel log.

Cheaper than downloading 3 x 320 MB of checkpoints to look at five paragraphs of
text. Attaches the prep kernel (tokenizer) and the train kernel (weights) as
kernel sources, so nothing is copied by hand.

Settings: no GPU needed, internet ON (to clone the repo).
"""

import os
import subprocess
import sys

PROMPTS = [
    "Once upon a time",
    "One day, a little girl named Lily",
    "The dog was very sad because",
    "Tom and his friend went to the park.",
]


def find(pattern: str) -> str:
    for root, _dirs, files in os.walk("/kaggle/input"):
        if pattern in files:
            path = os.path.join(root, pattern)
            print(f"[find] {path}", flush=True)
            return path
    raise SystemExit(f"{pattern} not found under /kaggle/input")


def main() -> int:
    url = os.environ.get("REPO_URL", "https://github.com/Narla7/tinystories-llm")
    dest = "/kaggle/working/repo"
    if not os.path.exists(os.path.join(dest, "sample.py")):
        subprocess.check_call(["git", "clone", "--depth", "1", url, dest])
    os.chdir(dest)

    ckpt = find("best.pt")
    tok = find("tokenizer.json")

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n===== {i}. prompt: {prompt!r} =====", flush=True)
        subprocess.check_call([
            sys.executable, "sample.py",
            "--ckpt", ckpt,
            "--tokenizer", tok,
            "--prompt", prompt,
            "--num-samples", "2",
            "--max-new-tokens", "220",
            "--temperature", "0.8",
            "--top-k", "40",
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
