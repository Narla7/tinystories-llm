#!/usr/bin/env python3
"""Copy the tokenizer + meta out of the prep kernel's output so they can be
downloaded without pulling 1.1 GB of token bins."""

import json
import os
import shutil

OUT = "/kaggle/working/nemo-tokenizer"
WANT = ("tokenizer.json", "meta.json")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    found = 0
    for root, _dirs, files in os.walk("/kaggle/input"):
        for w in WANT:
            if w in files:
                src = os.path.join(root, w)
                if os.path.exists(os.path.join(OUT, w)):
                    continue
                shutil.copy2(src, os.path.join(OUT, w))
                print(f"[copy] {src} -> {OUT}/{w} ({os.path.getsize(src)} bytes)", flush=True)
                found += 1
    meta = os.path.join(OUT, "meta.json")
    if os.path.exists(meta):
        print(open(meta).read(), flush=True)
    print(f"[done] {found} files in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
