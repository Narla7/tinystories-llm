#!/usr/bin/env python3
"""
Nemo is a 27.58M-parameter storyteller trained from scratch on TinyStories,
with its own terminal UI and its own inference engine baked in.

nemo_engine.py runs the forward pass in numpy and this file draws curses on
top of it. The only thing to install is numpy.

    python nemo.py                      # the TUI
    python nemo.py "Once upon a time"   # one-shot, no UI
    python nemo.py --fetch              # just download the weights

Keys:  Enter send · Ctrl+Q quit · Ctrl+L clear · Ctrl+N next seed
Slash: /clear  /temp 0.9  /topk 40  /seed 7  /len 300  /help
"""

from __future__ import annotations

import argparse
import curses
import os
import queue
import shutil
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "nemo_weights.npz")
TOKENIZER = os.path.join(HERE, "nemo_tokenizer.json")
RELEASE_URL = ("https://github.com/Narla7/tinystories-llm/releases/download/"
               "nemo-v2/nemo_weights.npz")

HELP = ("Nemo is a 27.58M-parameter GPT trained from scratch on 350M tokens of "
        "TinyStories. It writes 2-3 paragraph children's stories and knows "
        "nothing else. 512 tokens of context, no memory beyond that.")


def fetch(verbose: bool = True) -> None:
    if os.path.exists(WEIGHTS) and os.path.exists(TOKENIZER):
        return
    if not os.path.exists(WEIGHTS):
        if verbose:
            print(f"downloading weights from\n  {RELEASE_URL}")
        tmp = WEIGHTS + ".part"
        with urllib.request.urlopen(RELEASE_URL, timeout=180) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(tmp, WEIGHTS)
        if verbose:
            print(f"got {os.path.getsize(WEIGHTS)/1e6:.0f} MB")
    if not os.path.exists(TOKENIZER):
        sys.exit(f"missing {TOKENIZER} -- it ships with the repo, re-clone")


def load_engine():
    from nemo_engine import Nemo
    t0 = time.time()
    eng = Nemo(WEIGHTS, TOKENIZER)
    print(f"[nemo] {len(eng.layers)} layers, {eng.n_embd}d, "
          f"{sum(v.size for v in eng.p.values())/1e6:.2f}M params, "
          f"loaded in {time.time()-t0:.1f}s")
    return eng


# layout

def wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        cur = ""
        for word in para.split(" "):
            while len(word) > width:
                if cur:
                    lines.append(cur)
                    cur = ""
                lines.append(word[:width])
                word = word[width:]
            if not cur:
                cur = word
            elif len(cur) + 1 + len(word) <= width:
                cur += " " + word
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
    return lines


def draw(stdscr, st) -> None:
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    if h < 8 or w < 34:
        stdscr.addnstr(0, 0, "terminal too small", max(w, 1))
        stdscr.refresh()
        return

    stdscr.attron(curses.A_REVERSE)
    stdscr.addnstr(0, 0, " NEMO  27.58M params · from scratch on TinyStories ".ljust(w), w)
    stdscr.attroff(curses.A_REVERSE)

    status = (f" temp {st['opts']['temperature']:.2f} · top_k {st['opts']['top_k']}"
              f" · seed {st['opts']['seed']} · {st['rate']:.0f} tok/s"
              f" · Ctrl+Q quit ")
    stdscr.attron(curses.A_DIM)
    stdscr.addnstr(h - 2, 0, status.ljust(w), w)
    stdscr.attroff(curses.A_DIM)

    inner = h - 4
    body = wrap(st["text"], w - 2)
    if st["scroll"]:
        shown = body[max(0, len(body) - inner - st["scroll"]): max(0, len(body) - st["scroll"])]
    else:
        shown = body[-inner:] if len(body) > inner else body
    for i, line in enumerate(shown):
        stdscr.addnstr(1 + i, 1, line, w - 2)

    stdscr.attron(curses.A_BOLD)
    stdscr.addnstr(h - 3, 0, "> ", 2)
    stdscr.attroff(curses.A_BOLD)
    avail = w - 3
    view = st["buf"][-avail:] if len(st["buf"]) > avail else st["buf"]
    stdscr.addnstr(h - 3, 2, view, avail)
    if not st["busy"]:
        try:
            stdscr.move(h - 3, min(2 + len(view), w - 1))
        except curses.error:
            pass
    stdscr.refresh()


def worker(eng, prompt, opts, q: "queue.Queue") -> None:
    try:
        t0 = time.time()
        n = 0
        for piece in eng.generate(prompt, **opts):
            q.put(("chunk", piece))
            n += 1
        q.put(("done", (n, time.time() - t0)))
    except Exception as e:  # noqa: BLE001
        q.put(("error", f"{type(e).__name__}: {e}"))


def tui(stdscr, eng) -> None:
    curses.curs_set(1)
    stdscr.keypad(True)
    stdscr.nodelay(True)

    st = {"text": HELP + "\n\n", "buf": "", "busy": False, "scroll": 0, "rate": 0.0,
          "opts": {"max_new": 300, "temperature": 0.8, "top_k": 40, "top_p": 0.95,
                   "seed": 7, "repeat_penalty": 1.1}}
    q: queue.Queue = queue.Queue()

    draw(stdscr, st)
    while True:
        try:
            while True:
                kind, payload = q.get_nowait()
                if kind == "chunk":
                    st["text"] += payload
                elif kind == "done":
                    n, dt = payload
                    st["rate"] = n / dt if dt else 0
                    st["text"] += "\n\n"
                    st["busy"] = False
                elif kind == "error":
                    st["text"] += f"\n[error] {payload}\n\n"
                    st["busy"] = False
        except queue.Empty:
            pass

        ch = stdscr.getch()
        if ch == 17:                      # Ctrl+Q
            return
        elif ch == 12:                    # Ctrl+L
            st["text"] = ""
        elif ch == 14:                    # Ctrl+N: next seed
            st["opts"]["seed"] = (st["opts"]["seed"] + 1) % 100000
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            st["buf"] = st["buf"][:-1]
        elif ch == curses.KEY_UP:
            st["scroll"] += 3
        elif ch == curses.KEY_DOWN:
            st["scroll"] = max(0, st["scroll"] - 3)
        elif ch in (10, 13, curses.KEY_ENTER) and not st["busy"]:
            line = st["buf"].strip()
            st["buf"] = ""
            if line.startswith("/"):
                slash(line, st)
            elif line:
                st["busy"] = True
                st["scroll"] = 0
                st["text"] += ">>> " + line + "\n"
                threading.Thread(target=worker, args=(eng, line, dict(st["opts"]), q),
                                 daemon=True).start()
        elif ch != -1 and 32 <= ch < 127 and not st["busy"]:
            if len(st["buf"]) < 400:
                st["buf"] += chr(ch)

        draw(stdscr, st)
        time.sleep(0.02)


def slash(line: str, st) -> None:
    parts = line.split()
    cmd, arg = parts[0], (parts[1] if len(parts) > 1 else None)
    if cmd == "/clear":
        st["text"] = ""
    elif cmd == "/help":
        st["text"] += HELP + "\n\n"
    elif cmd in ("/temp", "/topk", "/seed", "/len") and arg:
        try:
            if cmd == "/temp":
                st["opts"]["temperature"] = max(0.0, min(2.0, float(arg)))
            elif cmd == "/topk":
                st["opts"]["top_k"] = max(1, int(arg))
            elif cmd == "/seed":
                st["opts"]["seed"] = int(arg)
            else:
                st["opts"]["max_new"] = max(1, min(500, int(arg)))
        except ValueError:
            st["text"] += f"[bad value for {cmd}]\n\n"

# main

def main() -> int:
    ap = argparse.ArgumentParser(description="Nemo — a tiny storyteller, self-contained")
    ap.add_argument("prompt", nargs="*")
    ap.add_argument("--fetch", action="store_true", help="download the weights and exit")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--len", dest="max_new", type=int, default=300)
    args = ap.parse_args()

    fetch()
    if args.fetch:
        print("weights in place. run:  python nemo.py")
        return 0

    eng = load_engine()

    if args.prompt:
        text = " ".join(args.prompt)
        sys.stdout.write(text)
        sys.stdout.flush()
        t0, n = time.time(), 0
        for piece in eng.generate(text, max_new=args.max_new, temperature=args.temp,
                                  top_k=args.topk, seed=args.seed):
            sys.stdout.write(piece)
            sys.stdout.flush()
            n += 1
        dt = time.time() - t0
        print(f"\n\n[{n} tokens, {n/dt:.0f} tok/s]")
        return 0

    try:
        curses.wrapper(tui, eng)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
