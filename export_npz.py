#!/usr/bin/env python3
"""
Pack the trained weights into a single .npz that numpy can load directly.

No GGUF, no llama.cpp, no Ollama, no torch at runtime. The harness that ships
with this model reads this file and does the forward pass itself, so the only
thing a user installs is numpy.

    python export_npz.py --hf export/nemo --out nemo_weights.npz

Storage is float16 to keep the download at ~55 MB; the engine casts to float32
once at load. Tensor layout is HF's GPT-2 (Conv1D projections are (in, out)),
which is what makes `x @ W` correct without a transpose at runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="export/nemo", help="directory with model.safetensors")
    ap.add_argument("--out", default="nemo_weights.npz")
    args = ap.parse_args()

    src = Path(args.hf)
    tensors = {}
    with safe_open(str(src / "model.safetensors"), framework="numpy") as f:
        for name in f.keys():
            arr = f.get_tensor(name)
            tensors[name] = arr.astype(np.float16)

    # tied embeddings: the checkpoint has no lm_head, GPT-2 re-ties it, and the
    # engine uses wte.T as the output head, so do not store a second copy
    np.savez(args.out, **tensors)
    total = sum(v.size for v in tensors.values())
    print(f"[npz] {len(tensors)} tensors, {total/1e6:.2f}M params -> {args.out} "
          f"({Path(args.out).stat().st_size/1e6:.0f} MB fp16)")

    # tokenizer as plain JSON so the engine can carry its own BPE
    tok = json.loads((src / "tokenizer.json").read_text())
    tok_out = {
        "vocab": tok["model"]["vocab"],
        "merges": tok["model"]["merges"],
        "eos_id": tok["added_tokens"][0]["id"] if tok.get("added_tokens") else 0,
        "block_size": json.loads((src / "config.json").read_text())["n_ctx"],
    }
    Path("nemo_tokenizer.json").write_text(json.dumps(tok_out))
    print(f"[npz] tokenizer -> nemo_tokenizer.json "
          f"({len(tok_out['vocab'])} tokens, {len(tok_out['merges'])} merges, "
          f"eos {tok_out['eos_id']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
