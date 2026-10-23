#!/usr/bin/env python3
"""
Sample from a checkpoint.

    python sample.py --run-dir runs/smoke --prompt "Once upon a time"
    python sample.py --run-dir runs/tiny25m --num-samples 5 --temperature 0.8 --top-k 40

Temperature 0.8 / top-k 40 is a decent default for TinyStories. Temperature 0.0
is greedy and will loop; temperature >1.2 falls apart.
"""

from __future__ import annotations

import argparse
import os

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None, help="directory holding last.pt/best.pt")
    ap.add_argument("--ckpt", default=None, help="explicit checkpoint path")
    ap.add_argument("--tokenizer", default=None, help="defaults to <data-dir>/tokenizer.json")
    ap.add_argument("--data-dir", default=None, help="where tokenizer.json lives")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    ckpt_path = args.ckpt
    if ckpt_path is None:
        if not args.run_dir:
            ap.error("need --run-dir or --ckpt")
        for name in ("best.pt", "last.pt"):
            cand = os.path.join(args.run_dir, name)
            if os.path.exists(cand):
                ckpt_path = cand
                break
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise SystemExit(f"no checkpoint found ({ckpt_path})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig.from_dict(ckpt["model_cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[load] {ckpt_path} | step {ckpt['step']} | {ckpt['params']/1e6:.2f}M params | {device}")

    tok_path = args.tokenizer
    if tok_path is None:
        cands = [args.data_dir, ckpt.get("data_dir")]
        for d in cands:
            if d and os.path.exists(os.path.join(d, "tokenizer.json")):
                tok_path = os.path.join(d, "tokenizer.json")
                break
    if tok_path is None or not os.path.exists(tok_path):
        raise SystemExit("tokenizer.json not found -- pass --tokenizer or --data-dir")
    tok = Tokenizer.from_file(tok_path)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    for i in range(args.num_samples):
        ids = tok.encode(args.prompt, add_special_tokens=False).ids
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        text = tok.decode(out[0].tolist())
        if "<eos>" in text:
            text = text.split("<eos>")[0]
        print(f"\n--- sample {i+1} ---\n{text.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
