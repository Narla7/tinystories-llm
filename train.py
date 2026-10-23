#!/usr/bin/env python3
"""
Train the TinyStories GPT.

Durability guarantees, in priority order:

  1. A checkpoint is never half-written. Every save goes to `<name>.tmp`, is
     flushed and fsynced, then os.replace()d into place, then the directory is
     fsynced. os.replace is atomic on ext4 and on NTFS, so the file you resume
     from is always a complete one -- a power cut mid-save leaves the previous
     good file untouched rather than a truncated new one.

  2. Worst-case lost work is bounded in MINUTES, not steps, because the
     checkpoint cadence is wall-clock (`ckpt_minutes`), not step-count. Step
     time varies with hardware; wall-clock does not.

  3. Three rotation slots (last / prev / best) so one bad file is not a dead run.

  4. SIGINT and SIGTERM save before exiting, covering Ctrl-C, a closed terminal,
     and a shutdown. A power cut cannot be caught -- which is precisely why (2)
     exists.

Resume restores model, optimizer moments, LR schedule position, step counter,
RNG state, and the best-val bookkeeping. Adam without its moments restarts badly,
so all of it or none.

Usage:
    python train.py --config configs/tiny25m.json
    python train.py --config configs/tiny25m.json --max-steps 200   # bench run
    python train.py --config configs/tiny25m.json --fresh           # ignore last.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import sys
import time

import numpy as np
import torch

from model import GPT, GPTConfig

STOP = {"flag": False, "reason": ""}


def _install_signal_handlers():
    def handler(signum, frame):
        STOP["flag"] = True
        STOP["reason"] = signal.Signals(signum).name
        print(f"\n[signal] {STOP['reason']} received -- saving a checkpoint, then exiting.", flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


# --------------------------------------------------------------------------- #
# durability
# --------------------------------------------------------------------------- #

def atomic_save(obj, path: str) -> None:
    """Write obj to path such that a crash leaves either the old file or the new one."""
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    # Windows: antivirus / Explorer can hold the target open for a moment and
    # os.replace raises PermissionError. Retry rather than losing the save.
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            time.sleep(0.2 * (attempt + 1))
    else:
        raise RuntimeError(f"could not replace {path} after 8 attempts")
    try:  # persist the rename itself; not available on Windows
        dfd = os.open(os.path.dirname(path) or ".", getattr(os, "O_DIRECTORY", 0))
        os.fsync(dfd)
        os.close(dfd)
    except OSError:
        pass


def rotate(src: str, dst: str) -> None:
    if os.path.exists(src):
        try:
            os.replace(src, dst)
        except OSError:
            pass
# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

class TokenDataset:
    """Flat uint16 token stream, random windows. No shuffling needed: sampling is random."""

    def __init__(self, path: str, block_size: int):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} -- run tokenize_data.py first")
        self.block_size = block_size
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        if len(self.data) <= block_size + 1:
            raise ValueError(f"{path} has {len(self.data)} tokens, need > {block_size + 1}")

    def __len__(self) -> int:
        return len(self.data)

    def batch(self, bs: int, device: str):
        hi = len(self.data) - self.block_size - 1
        ix = np.random.randint(0, hi, size=bs)
        x = np.stack([self.data[i: i + self.block_size] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1: i + 1 + self.block_size] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


# --------------------------------------------------------------------------- #
# schedule / eval
# --------------------------------------------------------------------------- #

def lr_at(step: int, max_steps: int, lr: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def fmt_n(n: float) -> str:
    return f"{n/1e6:.1f}M" if n >= 1e6 else f"{n/1e3:.0f}K"


@torch.no_grad()
def estimate_loss(model, datasets, eval_iters, batch_size, device, amp_ctx):
    was_training = model.training
    model.eval()
    out = {}
    for split, ds in datasets.items():
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = ds.batch(batch_size, device)
            with amp_ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train(was_training)
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="override for bench runs")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing last.pt")
    ap.add_argument("--device", default=None, help="cpu | cuda | cuda:0")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    tcfg = cfg["train"]
    mcfg = GPTConfig.from_dict(cfg["model"])
    data_dir = cfg["data_dir"]
    run_dir = cfg["run_dir"]
    os.makedirs(run_dir, exist_ok=True)

    seed = args.seed if args.seed is not None else tcfg.get("seed", 1337)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- device & precision -------------------------------------------------
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if torch.cuda.is_bf16_supported():
            amp_dtype, use_scaler = torch.bfloat16, False
        else:
            amp_dtype, use_scaler = torch.float16, True  # T4: Turing, no bf16
        print(f"[device] {torch.cuda.get_device_name(0)} | {amp_dtype} | scaler={use_scaler}")
    else:
        amp_dtype, use_scaler = torch.float32, False
        threads = tcfg.get("cpu_threads") or os.cpu_count() or 1
        torch.set_num_threads(int(threads))
        print(f"[device] cpu | fp32 | threads={torch.get_num_threads()}")

    amp_ctx = torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=(device_type == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ---- data --------------------------------------------------------------
    train_ds = TokenDataset(os.path.join(data_dir, "train.bin"), mcfg.block_size)
    val_path = os.path.join(data_dir, "val.bin")
    val_ds = TokenDataset(val_path, mcfg.block_size) if os.path.exists(val_path) else None
    datasets = {"train": train_ds, **({"val": val_ds} if val_ds is not None else {})}
    print(f"[data] train={len(train_ds):,} tokens"
          + (f" val={len(val_ds):,} tokens" if val_ds is not None else " (no val.bin)"))

    # ---- model -------------------------------------------------------------
    model = GPT(mcfg).to(device)
    print(f"[model] {model.num_params() / 1e6:.2f}M total, "
          f"{model.num_params(non_embedding=True) / 1e6:.2f}M non-embedding, "
          f"vocab={mcfg.vocab_size} block={mcfg.block_size} layers={mcfg.n_layer}")
    n_all = model.num_params()

    # ---- steps -------------------------------------------------------------
    tokens_per_step = int(tcfg["batch_size"]) * mcfg.block_size * int(tcfg.get("grad_accum", 1))
    if tcfg.get("max_steps"):
        max_steps = int(tcfg["max_steps"])
    else:
        target = float(tcfg["target_tokens"])
        max_steps = max(1, int(round(target / tokens_per_step)))
        epochs = target / len(train_ds)
        print(f"[plan] target {target/1e6:.0f}M tokens over {len(train_ds)/1e6:.0f}M available "
              f"= {epochs:.2f} epochs")
    if args.max_steps:
        max_steps = args.max_steps
    warmup = int(tcfg.get("warmup_steps") or max(10, int(tcfg.get("warmup_frac", 0.02) * max_steps)))
    lr = float(tcfg["lr"])
    min_lr = lr * float(tcfg.get("min_lr_frac", 0.1))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    eval_interval = int(tcfg.get("eval_interval", 500))
    eval_iters = int(tcfg.get("eval_iters", 50))
    log_interval = int(tcfg.get("log_interval", 50))
    ckpt_seconds = float(tcfg.get("ckpt_minutes", 10)) * 60.0
    batch_size = int(tcfg["batch_size"])
    grad_accum = int(tcfg.get("grad_accum", 1))

    optimizer = model.configure_optimizers(
        weight_decay=float(tcfg.get("weight_decay", 0.1)), lr=lr, device_type=device_type
    )

    print(f"[plan] {max_steps} steps x {tokens_per_step:,} tokens/step = "
          f"{fmt_n(max_steps * tokens_per_step)} tokens | warmup {warmup} | lr {lr:g}->{min_lr:g}")

    # ---- resume ------------------------------------------------------------
    start_step, best_val = 0, float("inf")
    last_path = os.path.join(run_dir, "last.pt")
    if os.path.exists(last_path) and not args.fresh:
        try:
            ckpt = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if use_scaler and ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
            start_step = int(ckpt["step"]) + 1
            best_val = float(ckpt.get("best_val", float("inf")))
            if ckpt.get("rng"):
                try:
                    torch.set_rng_state(ckpt["rng"]["torch"].cpu() if hasattr(ckpt["rng"]["torch"], "cpu") else ckpt["rng"]["torch"])
                    np.random.set_state(ckpt["rng"]["numpy"])
                except Exception:
                    pass
            print(f"[resume] {last_path} at step {start_step}, best_val {best_val:.4f}")
        except Exception as e:  # noqa: BLE001
            fallback = os.path.join(run_dir, "prev.pt")
            print(f"[resume] last.pt unreadable ({type(e).__name__}: {e})", flush=True)
            if os.path.exists(fallback):
                ckpt = torch.load(fallback, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                start_step = int(ckpt["step"]) + 1
                best_val = float(ckpt.get("best_val", float("inf")))
                print(f"[resume] fell back to prev.pt at step {start_step}")
            else:
                print("[resume] no fallback either -- starting from scratch")

    def save(tag: str) -> None:
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_scaler else None,
            "step": step,
            "best_val": best_val,
            "model_cfg": mcfg.__dict__,
            "train_cfg": tcfg,
            "data_dir": data_dir,
            "seed": seed,
            "wall": time.time(),
            "params": n_all,
            "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()},
        }
        if tag == "last" and os.path.exists(last_path):
            # keep the previous good file so one bad write is never fatal
            try:
                shutil.copyfile(last_path, os.path.join(run_dir, "prev.pt"))
            except OSError:
                pass
        atomic_save(payload, os.path.join(run_dir, f"{tag}.pt"))
        print(f"[ckpt] saved step {step} ({tag})", flush=True)

    # ---- train -------------------------------------------------------------
    if start_step >= max_steps:
        print(f"[done] checkpoint is already at step {start_step} but max_steps={max_steps}; "
              f"nothing to do. Raise max_steps to continue training.")
        return 0

    _install_signal_handlers()
    model.train()
    t0 = time.time()
    last_ckpt = t0
    last_log = t0
    tokens_seen = start_step * tokens_per_step
    running_loss = None
    steps_run = 0
    step = start_step

    for step in range(start_step, max_steps):
        if STOP["flag"]:
            break

        cur_lr = lr_at(step, max_steps, lr, min_lr, warmup)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            x, y = train_ds.batch(batch_size, device)
            with amp_ctx:
                _, loss = model(x, y)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            micro = loss.item() * grad_accum
            running_loss = micro if running_loss is None else 0.9 * running_loss + 0.1 * micro

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        tokens_seen += tokens_per_step
        steps_run += 1

        now = time.time()

        # ---- bench output: measured tok/s and a recomputed finish time ----
        if now - last_log >= tcfg.get("log_seconds", 10) or step == start_step:
            elapsed = now - t0
            done = step - start_step + 1
            tok_s = (done * tokens_per_step) / max(elapsed, 1e-9)
            remaining = (max_steps - step - 1) * tokens_per_step / max(tok_s, 1e-9)
            print(
                f"step {step+1:>6}/{max_steps} | loss {running_loss:7.4f} | lr {cur_lr:.2e} | "
                f"{tok_s:8.0f} tok/s | elapsed {elapsed/60:6.1f}m | eta {remaining/3600:5.2f}h",
                flush=True,
            )
            last_log = now

        # ---- eval ----
        if val_ds is not None and (step + 1) % eval_interval == 0:
            losses = estimate_loss(model, datasets, eval_iters, batch_size, device, amp_ctx)
            print(f"[eval] step {step+1} " + " ".join(f"{k} {v:.4f}" for k, v in losses.items()), flush=True)
            if losses["val"] < best_val:
                best_val = losses["val"]
                save("best")

        # ---- time-based checkpoint ----
        if now - last_ckpt >= ckpt_seconds:
            save("last")
            last_ckpt = now

    # ---- final save ----
    if steps_run == 0:
        print("[done] stopped before completing a step; checkpoint unchanged")
        return 0
    save("last")
    elapsed = time.time() - t0
    print(f"\n[done] step {step+1}/{max_steps} in {elapsed/60:.1f}m "
          f"({fmt_n(tokens_seen)} tokens total, {steps_run*tokens_per_step/max(elapsed,1e-9):,.0f} tok/s)")
    print(f"[done] checkpoint: {os.path.join(run_dir, 'last.pt')}")
    if STOP["flag"]:
        print(f"[done] stopped early by {STOP['reason']}; rerun the same command to resume.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
