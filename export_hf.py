#!/usr/bin/env python3
"""
Turn a training checkpoint into an HF-layout model directory that llama.cpp's
converter can read.

Our checkpoints are nanoGPT-shaped: `blocks.N.attn.c_attn.weight`, `ln_f.bias`,
and so on. llama.cpp wants GPT-2's names: `transformer.h.N.attn.c_attn.weight`.
The tensors are identical; only the spelling differs. That rename is the whole
job of this file.

    python export_hf.py --ckpt out-train/runs/tiny25m/best.pt \
                        --tokenizer out-tok/nemo-tokenizer/tokenizer.json \
                        --out export/nemo

Produces:
    config.json          GPT2Config fields matching the training config
    model.safetensors    HF-named weights, fp32
    tokenizer.json       copy of the original BPE
    vocab.json           {token: id} for llama.cpp
    merges.txt           "#version: 0.2" + merges, for llama.cpp
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")


def to_hf_keys(state: dict) -> dict:
    out = {}
    for name, tensor in state.items():
        if name == "lm_head.weight":
            continue  # tied to wte; GPT2LMHeadModel re-ties on load
        if name.startswith("wte.") or name.startswith("wpe.") or name.startswith("ln_f."):
            out["transformer." + name] = tensor
            continue
        m = BLOCK_RE.match(name)
        if m:
            out[f"transformer.h.{m.group(1)}.{m.group(2)}"] = tensor
            continue
        raise KeyError(f"unexpected parameter name: {name}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["model_cfg"]
    state = {k: v.contiguous() for k, v in to_hf_keys(ckpt["model"]).items()}
    # HF's GPT-2 stores c_attn/c_proj/c_fc as Conv1D, i.e. (in, out), not the
    # (out, in) of nn.Linear. Transposing here makes the export a genuine HF
    # checkpoint *and* is what llama.cpp's converter expects -- it transposes
    # these back on the way in.
    for k in list(state):
        if k.endswith(("c_attn.weight", "c_proj.weight", "c_fc.weight")):
            state[k] = state[k].t().contiguous()
    save_file(state, str(out / "model.safetensors"))
    print(f"[export] {len(state)} tensors -> {out/'model.safetensors'} "
          f"({os.path.getsize(out/'model.safetensors')/1e6:.1f} MB)")

    config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": cfg["vocab_size"],
        "n_positions": cfg["block_size"],
        "n_ctx": cfg["block_size"],             # llama.cpp's gpt2 converter reads this one
        "n_embd": cfg["n_embd"],
        "n_layer": cfg["n_layer"],
        "n_head": cfg["n_head"],
        "n_inner": 4 * cfg["n_embd"],
        "activation_function": "gelu",          # exact gelu, not gelu_new
        "resid_pdrop": 0.0, "embd_pdrop": 0.0, "attn_pdrop": 0.0,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "tie_word_embeddings": True,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "torch_dtype": "float32",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    shutil.copy2(args.tokenizer, out / "tokenizer.json")
    tok = json.loads(Path(args.tokenizer).read_text())
    model = tok["model"]
    vocab = model["vocab"]                      # {token: id}
    merges = model["merges"]                    # ["a b", ...]
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False))
    with open(out / "merges.txt", "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for m in merges:
            f.write(f"{m}\n")

    # Without this, transformers appends its default <|endoftext|> at id 4096 as an
    # added token, which pushes max(vocab) past vocab_size and makes llama.cpp's
    # converter fail an assertion. Point the specials at the <eos> we already have.
    (out / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "GPT2Tokenizer",
        "eos_token": "<eos>",
        "bos_token": "<eos>",
        "unk_token": None,
        "add_bos_token": False,
        "add_prefix_space": False,
        "model_max_length": cfg["block_size"],
    }, indent=2) + "\n")
    (out / "special_tokens_map.json").write_text(json.dumps({
        "eos_token": "<eos>",
        "bos_token": "<eos>",
    }, indent=2) + "\n")
    print(f"[export] vocab {len(vocab)} tokens, {len(merges)} merges")

    # sanity: every token id 0..vocab-1 present, no holes
    ids = sorted(vocab.values())
    holes = [i for i in range(len(ids)) if ids[i] != i]
    print(f"[export] id range {ids[0]}..{ids[-1]}, contiguous: {not holes}")
    print(f"[export] wrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
