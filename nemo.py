#!/usr/bin/env python3
"""
Nemo — a 27.58M-parameter storyteller, trained from scratch on TinyStories,
with its own terminal UI baked in.

Nothing here downloads a model from someone else's repo, because there isn't
one: Nemo was pretrained, tokenised and quantised by hand. This file is the
harness. It talks to a local Ollama daemon over HTTP and draws a curses UI on
top, so the model and the way you talk to it ship together.

    python nemo.py                      # the TUI
    python nemo.py "Once upon a time"   # one-shot, no UI
    python nemo.py --install            # just fetch the weights and create the model

Keys:  Enter send · Ctrl+Q quit · Ctrl+L clear
Slash: /clear  /temp 0.9  /topk 40  /seed 7  /help
"""

from __future__ import annotations

import argparse
import curses
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
MODEL = "nemo"
HERE = os.path.dirname(os.path.abspath(__file__))
GGUF = os.path.join(HERE, "nemo-f16.gguf")
MODELFILE = os.path.join(HERE, "Modelfile")
RELEASE_URL = ("https://github.com/Narla7/tinystories-llm/releases/download/"
               "nemo-v1/nemo-f16.gguf")

HELP = ("Nemo writes 2-3 paragraph children's stories. It was trained on 350M "
        "tokens of TinyStories, so it is good at simple narrative and knows "
        "nothing else. Context is 512 tokens; it has no memory beyond that.")


# --------------------------------------------------------------------------- #
# ollama plumbing
# --------------------------------------------------------------------------- #

def api(path: str, payload=None, timeout=10):
    url = OLLAMA + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def daemon_up() -> bool:
    try:
        api("/api/tags", timeout=3)
        return True
    except Exception:
        return False


def have_model() -> bool:
    try:
        return any(m["name"].split(":")[0] == MODEL for m in api("/api/tags")["models"])
    except Exception:
        return False


def ensure_model(verbose=True) -> None:
    """Download the GGUF if needed, then `ollama create nemo`."""
    if have_model():
        return
    if not os.path.exists(GGUF):
        if verbose:
            print(f"downloading {RELEASE_URL}")
        os.makedirs(HERE, exist_ok=True)
        tmp = GGUF + ".part"
        with urllib.request.urlopen(RELEASE_URL, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(tmp, GGUF)
        if verbose:
            print(f"got {os.path.getsize(GGUF)/1e6:.0f} MB")
    if not os.path.exists(MODELFILE):
        sys.exit(f"missing {MODELFILE}")
    if verbose:
        print("creating the ollama model (55 MB, one time)")
    subprocess.check_call(["ollama", "create", MODEL, "-f", MODELFILE],
                          stdout=subprocess.DEVNULL)
    if not have_model():
        sys.exit("ollama create reported success but the model is not listed")


def stream(prompt: str, opts: dict, out: "queue.Queue") -> None:
    """Run one generation, pushing ('chunk', text) then ('done', stats) into out."""
    payload = {"model": MODEL, "prompt": prompt, "stream": True, "options": opts}
    req = urllib.request.Request(
        OLLAMA + "/api/generate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                if not raw.strip():
                    continue
                msg = json.loads(raw)
                if msg.get("response"):
                    out.put(("chunk", msg["response"]))
                if msg.get("done"):
                    out.put(("done", msg))
    except Exception as e:  # noqa: BLE001
        out.put(("error", str(e)))


# --------------------------------------------------------------------------- #
# text layout
# --------------------------------------------------------------------------- #

def wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        cur = ""
        for word in para.split(" "):
            while len(word) > width:  # a single unbreakable token longer than the pane
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


# --------------------------------------------------------------------------- #
# TUI
# --------------------------------------------------------------------------- #

def draw(stdscr, st) -> None:
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    w = max(w, 40)
    if h < 8:
        stdscr.addnstr(0, 0, "terminal too small", w)
        stdscr.refresh()
        return

    # header
    head = f" NEMO  27.58M params · trained from scratch on TinyStories "
    stdscr.attron(curses.A_REVERSE)
    stdscr.addnstr(0, 0, head.ljust(w), w)
    stdscr.attroff(curses.A_REVERSE)

    # status line
    st_line = (f" temp {st['opts']['temperature']:.2f} · top_k {st['opts']['top_k']} "
               f"· seed {st['opts']['seed']} · Ctrl+Q quit")
    if st["rate"]:
        st_line += f" · {st['rate']:.0f} tok/s"
    stdscr.attron(curses.A_DIM)
    stdscr.addnstr(h - 2, 0, st_line.ljust(w), w)
    stdscr.attroff(curses.A_DIM)

    # transcript
    body = wrap(st["text"], w - 2)
    inner = h - 4
    if st["busy"]:
        shown = body[-(inner - 1):] if len(body) >= inner else body
    else:
        offset = st["scroll"]
        if offset:
            shown = body[max(0, len(body) - inner - offset): max(0, len(body) - offset)]
        else:
            shown = body[-inner:] if len(body) > inner else body
    for i, line in enumerate(shown):
        stdscr.addnstr(1 + i, 1, line, w - 2)

    # input
    stdscr.attron(curses.A_BOLD)
    prompt = "> "
    stdscr.addnstr(h - 3, 0, prompt, w)
    stdscr.attroff(curses.A_BOLD)
    avail = w - len(prompt) - 1
    buf = st["buf"]
    view = buf[-avail:] if len(buf) > avail else buf
    stdscr.addnstr(h - 3, len(prompt), view, avail)
    if not st["busy"]:
        try:
            stdscr.move(h - 3, min(len(prompt) + len(view), w - 1))
        except curses.error:
            pass
    stdscr.refresh()


def tui(stdscr) -> None:
    curses.curs_set(1)
    stdscr.keypad(True)
    stdscr.nodelay(True)
    try:
        curses.start_color()
    except curses.error:
        pass

    st = {
        "text": HELP + "\n",
        "buf": "",
        "busy": False,
        "scroll": 0,
        "rate": 0.0,
        "opts": {"temperature": 0.8, "top_k": 40, "seed": 7, "num_predict": 400,
                 "num_ctx": 512, "repeat_penalty": 1.1},
    }
    out: queue.Queue = queue.Queue()
    worker = None
    pending = ""  # text waiting to be appended to st["text"]

    draw(stdscr, st)
    while True:
        # ---- drain the stream ----
        try:
            while True:
                kind, payload = out.get_nowait()
                if kind == "chunk":
                    pending += payload
                elif kind == "done":
                    st["busy"] = False
                    n = payload.get("eval_count") or 0
                    ns = payload.get("eval_duration") or 1
                    st["rate"] = n / (ns / 1e9) if n else 0.0
                    st["text"] += pending + "\n\n"
                    pending = ""
                elif kind == "error":
                    st["busy"] = False
                    st["text"] += pending + f"\n[error] {payload}\n\n"
                    pending = ""
        except queue.Empty:
            pass

        # ---- keys ----
        ch = stdscr.getch()
        if ch == 17:  # Ctrl+Q
            return
        elif ch == 12:  # Ctrl+L
            st["text"] = ""
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            st["buf"] = st["buf"][:-1]
        elif ch == curses.KEY_UP:
            st["scroll"] += 3
        elif ch == curses.KEY_DOWN:
            st["scroll"] = max(0, st["scroll"] - 3)
        elif ch in (10, 13, curses.KEY_ENTER):
            line = st["buf"].strip()
            st["buf"] = ""
            if line.startswith("/"):
                handle_input(line, st)
            elif line:
                st["busy"] = True
                st["scroll"] = 0
                threading.Thread(target=stream, args=(line, st["opts"], out), daemon=True).start()
        elif ch != -1 and 32 <= ch < 127 and not st["busy"]:
            if len(st["buf"]) < 400:
                st["buf"] += chr(ch)

        if pending:
            st["text"] += pending
            pending = ""
        draw(stdscr, st)
        time.sleep(0.05)


def handle_input(line: str, st) -> None:
    """Slash commands are handled locally and never reach the model."""
    if not line.startswith("/"):
        return
    parts = line.split()
    cmd, arg = parts[0], (parts[1] if len(parts) > 1 else None)
    if cmd == "/clear":
        st["text"] = ""
    elif cmd == "/help":
        st["text"] += HELP + "\n\n"
    elif cmd in ("/temp", "/topk", "/seed") and arg:
        try:
            if cmd == "/temp":
                st["opts"]["temperature"] = max(0.0, min(2.0, float(arg)))
            elif cmd == "/topk":
                st["opts"]["top_k"] = max(1, int(arg))
            else:
                st["opts"]["seed"] = int(arg)
        except ValueError:
            st["text"] += f"[bad value for {cmd}]\n\n"


# --------------------------------------------------------------------------- #
# entry
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Nemo — a tiny storyteller with a TUI")
    ap.add_argument("prompt", nargs="*", help="one-shot prompt; omit for the TUI")
    ap.add_argument("--install", action="store_true", help="fetch weights and create the model")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tokens", type=int, default=400)
    args = ap.parse_args()

    if not daemon_up():
        sys.exit("ollama is not running. start it with:  ollama serve")
    ensure_model()
    if args.install:
        print("ready. run:  python nemo.py")
        return 0

    opts = {"temperature": args.temp, "top_k": args.topk, "seed": args.seed,
            "num_predict": args.tokens, "num_ctx": 512, "repeat_penalty": 1.1}

    if args.prompt:
        text = " ".join(args.prompt)
        q: queue.Queue = queue.Queue()
        t = threading.Thread(target=stream, args=(text, opts, q), daemon=True)
        t.start()
        while True:
            kind, payload = q.get()
            if kind == "chunk":
                sys.stdout.write(payload)
                sys.stdout.flush()
            elif kind == "done":
                n, ns = payload.get("eval_count") or 0, payload.get("eval_duration") or 1
                print(f"\n\n[{n} tokens, {n/(ns/1e9):.0f} tok/s]" if n else "")
                return 0
            elif kind == "error":
                sys.exit(f"[error] {payload}")

    try:
        curses.wrapper(tui)
    except KeyboardInterrupt:
        pass
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
