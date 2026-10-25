#!/usr/bin/env python3
"""
Turn TinyStories into two flat uint16 token files plus a BPE tokenizer.

This is a ONE-TIME job. Run it once, then commit the output to a private Kaggle
Dataset and mount that read-only in every training run. Re-tokenizing 2.2 GB of
text per session burns ten minutes of shared CPU for nothing.

    # full corpus, 4k vocab (the real thing)
    python tokenize_data.py --out-dir data/tinystories-4k --vocab-size 4096

    # smoke test: streams only the first N rows, costs a few MB not 673 MB
    python tokenize_data.py --out-dir data/sample --vocab-size 512 --limit-docs 400

Output:
    train.bin    flat uint16 token ids
    val.bin      flat uint16 token ids, --val-frac held out
    tokenizer.json
    meta.json    token counts, chars/token, vocab size, eos id
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from itertools import islice

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def load_texts(dataset: str, split: str, field: str, limit: int | None) -> list[str]:
    from datasets import load_dataset

    t0 = time.time()
    if limit:
        # streaming = HTTP range requests, so we only pull the first row groups
        ds = load_dataset(dataset, split=split, streaming=True)
        texts = [r[field] for r in islice(ds, limit)]
        print(f"[data] streamed {len(texts):,} docs in {time.time()-t0:.1f}s", flush=True)
    else:
        ds = load_dataset(dataset, split=split)
        texts = ds[field]
        print(f"[data] loaded {len(texts):,} docs in {time.time()-t0:.1f}s", flush=True)
    return texts


def train_tokenizer(texts: list[str], vocab_size: int, sample_docs: int) -> Tokenizer:
    sample = texts[:sample_docs] if len(texts) > sample_docs else texts
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    t0 = time.time()
    tok.train_from_iterator(sample, trainer=trainer)
    print(f"[tok] trained {tok.get_vocab_size()}-token BPE on {len(sample):,} docs "
          f"in {time.time()-t0:.1f}s", flush=True)
    return tok


def encode_all(tok: Tokenizer, texts: list[str], eos_id: int, out_path: str, chunk: int = 5000) -> int:
    """Encode in chunks, appending <eos> after every document, flush to disk as we go."""
    total = 0
    with open(out_path, "wb") as f:
        for start in range(0, len(texts), chunk):
            encs = tok.encode_batch(texts[start: start + chunk], add_special_tokens=False)
            buf = []
            for e in encs:
                buf.extend(e.ids)
                buf.append(eos_id)
            arr = np.asarray(buf, dtype=np.uint16)
            arr.tofile(f)
            total += arr.size
            if start % (chunk * 20) == 0 and start:
                print(f"  {start:,}/{len(texts):,} docs -> {total/1e6:.1f}M tokens", flush=True)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="karpathy/tinystories-gpt4-clean")
    ap.add_argument("--split", default="train")
    ap.add_argument("--field", default="text")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--min-chars", type=int, default=100)
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--tokenizer-docs", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    texts = load_texts(args.dataset, args.split, args.field, args.limit_docs)

    n_raw = len(texts)
    texts = [t for t in texts if t and len(t) >= args.min_chars]
    dropped = n_raw - len(texts)
    total_chars = sum(len(t) for t in texts)
    print(f"[data] {len(texts):,} docs kept, {dropped:,} dropped as too short, "
          f"{total_chars/1e9:.2f}B chars", flush=True)
    if len(texts) < 100:
        print("[data] WARNING: fewer than 100 docs. Vocab will be tiny and the model "
              "will overfit instantly. Fine for a plumbing test, useless as a model.")

    random.Random(args.seed).shuffle(texts)
    n_val = max(1, int(len(texts) * args.val_frac))
    val_texts, train_texts = texts[:n_val], texts[n_val:]
    print(f"[data] split: {len(train_texts):,} train / {len(val_texts):,} val docs", flush=True)

    tok = train_tokenizer(train_texts, args.vocab_size, args.tokenizer_docs)
    eos_id = tok.token_to_id("<eos>")
    tok.save(os.path.join(args.out_dir, "tokenizer.json"))

    print("[enc] encoding train...", flush=True)
    n_train = encode_all(tok, train_texts, eos_id, os.path.join(args.out_dir, "train.bin"))
    print("[enc] encoding val...", flush=True)
    n_val_tok = encode_all(tok, val_texts, eos_id, os.path.join(args.out_dir, "val.bin"))

    cpt = total_chars / max(n_train + n_val_tok, 1)
    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "vocab_size": tok.get_vocab_size(),
        "eos_id": eos_id,
        "train_tokens": n_train,
        "val_tokens": n_val_tok,
        "train_docs": len(train_texts),
        "val_docs": len(val_texts),
        "chars": total_chars,
        "chars_per_token": round(cpt, 3),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    print(f"\n[done] {args.out_dir}")
    print(f"       train.bin {os.path.getsize(os.path.join(args.out_dir,'train.bin'))/1e6:.0f} MB, "
          f"val.bin {os.path.getsize(os.path.join(args.out_dir,'val.bin'))/1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
