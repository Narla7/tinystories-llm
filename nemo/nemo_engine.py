"""
Nemo's inference engine: a GPT-2 forward pass in numpy, plus the byte-level BPE
that goes with it.

The model is 27.58M parameters, so a forward pass is 55 MFLOP per token
generated token and it collapses from ~100 tok/s to ~0.2.

Shapes follow HF's GPT-2 exactly: projections are Conv1D (in, out), so a layer is
`x @ W + b`, and the output head is the input embedding transposed.
"""

from __future__ import annotations

import codecs
import json
import math
from functools import lru_cache

import numpy as np

# byte-level BPE

def _bytes_to_unicode() -> dict[int, str]:
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_ENCODER = _bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}

_SPLIT = None


def _split_pattern():
    global _SPLIT
    if _SPLIT is None:
        import re
        _SPLIT = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d"
            r"| ?[A-Za-z]+"
            r"| ?[0-9]+"
            r"| ?[^\sA-Za-z0-9]+"
            r"|\s+(?!\S)"
            r"|\s+"
        )
    return _SPLIT


class BPETokenizer:
    def __init__(self, path: str):
        d = json.load(open(path))
        self.vocab: dict[str, int] = d["vocab"]
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.eos_id: int = d.get("eos_id", 0)
        self.block_size: int = d.get("block_size", 512)
        self._ranks = {}
        for i, m in enumerate(d["merges"]):
            pair = tuple(m) if isinstance(m, (list, tuple)) else tuple(m.split(" "))
            self._ranks[pair] = i
        self._cache: dict[str, int] = {}

    def _bpe(self, token: str) -> list[int]:
        word = tuple(BYTE_ENCODER[b] for b in token.encode("utf-8"))
        if len(word) == 1:
            return [self.vocab[word[0]]]
        while True:
            best, best_rank = None, None
            for pair in zip(word, word[1:]):
                r = self._ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    out.append(word[i] + word[i + 1])
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = tuple(out)
            if len(word) == 1:
                break
        return [self.vocab[w] for w in word]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in _split_pattern().findall(text):
            ids.extend(self._bpe(piece))
        return ids

    def decode(self, ids: list[int]) -> str:
        s = "".join(self.id_to_token.get(i, "") for i in ids if i != self.eos_id)
        raw = bytes(BYTE_DECODER.get(c, 63) for c in s)
        return raw.decode("utf-8", errors="replace")


# forward pass

def _erf(x: np.ndarray) -> np.ndarray:
    """Vectorised erf, Abramowitz & Stegun 7.1.26. Max error 1.5e-7.

    numpy has no erf and the tanh GELU approximation is a different function from
    the exact one this model was trained with, so approximate erf properly.
    """
    sign = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return sign * y


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


class Nemo:
    def __init__(self, weights: str, tokenizer: str):
        z = np.load(weights)
        p = {k: z[k].astype(np.float32) for k in z.files}
        self.p = p
        self.tok = BPETokenizer(tokenizer)
        self.n_layer = 1 + max(int(k.split(".")[2]) for k in p if k.startswith("transformer.h."))
        self.n_head = 8
        self.n_embd = p["transformer.wte.weight"].shape[1]
        self.head_dim = self.n_embd // self.n_head
        self.block = p["transformer.wpe.weight"].shape[0]
        self.wte = p["transformer.wte.weight"]
        self.wpe = p["transformer.wpe.weight"]
        self.layers = [{
            "ln1w": p[f"transformer.h.{i}.ln_1.weight"], "ln1b": p[f"transformer.h.{i}.ln_1.bias"],
            "qkv": p[f"transformer.h.{i}.attn.c_attn.weight"], "qkvb": p[f"transformer.h.{i}.attn.c_attn.bias"],
            "proj": p[f"transformer.h.{i}.attn.c_proj.weight"], "projb": p[f"transformer.h.{i}.attn.c_proj.bias"],
            "ln2w": p[f"transformer.h.{i}.ln_2.weight"], "ln2b": p[f"transformer.h.{i}.ln_2.bias"],
            "fc": p[f"transformer.h.{i}.mlp.c_fc.weight"], "fcb": p[f"transformer.h.{i}.mlp.c_fc.bias"],
            "fc2": p[f"transformer.h.{i}.mlp.c_proj.weight"], "fc2b": p[f"transformer.h.{i}.mlp.c_proj.bias"],
        } for i in range(self.n_layer)]
        self.lnfw = p["transformer.ln_f.weight"]
        self.lnfb = p["transformer.ln_f.bias"]
        self.reset()

    # state
    def reset(self) -> None:
        self.pos = 0
        self.k = [np.zeros((self.n_head, 0, self.head_dim), np.float32) for _ in range(self.n_layer)]
        self.v = [np.zeros((self.n_head, 0, self.head_dim), np.float32) for _ in range(self.n_layer)]

    @staticmethod
    def _ln(x, w, b, eps=1e-5):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * w + b

    # one token
    def step(self, token_id: int) -> np.ndarray:
        if self.pos >= self.block:
            # KV cache is full: drop the oldest position and shift everything down.
            self.reset()
        L = self.layers
        x = self.wte[token_id] + self.wpe[self.pos]
        for i, ly in enumerate(L):
            a = self._ln(x, ly["ln1w"], ly["ln1b"])
            qkv = a @ ly["qkv"] + ly["qkvb"]
            q, k, v = np.split(qkv, 3)
            q = q.reshape(self.n_head, self.head_dim)
            k = k.reshape(self.n_head, self.head_dim)
            v = v.reshape(self.n_head, self.head_dim)
            self.k[i] = np.concatenate([self.k[i], k[:, None, :]], axis=1)
            self.v[i] = np.concatenate([self.v[i], v[:, None, :]], axis=1)
            K, V = self.k[i], self.v[i]
            scores = (q[:, None, :] @ K.transpose(0, 2, 1)) / math.sqrt(self.head_dim)  # (H,1,T)
            scores -= scores.max(-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(-1, keepdims=True)
            att = (probs @ V).reshape(self.n_embd)
            x = x + (att @ ly["proj"] + ly["projb"])
            m = self._ln(x, ly["ln2w"], ly["ln2b"])
            x = x + (gelu(m @ ly["fc"] + ly["fcb"]) @ ly["fc2"] + ly["fc2b"])
        x = self._ln(x, self.lnfw, self.lnfb)
        self.pos += 1
        return x @ self.wte.T          # tied output head

    def prefill(self, ids: list[int]) -> np.ndarray:
        logits = None
        for t in ids[-self.block:]:
            logits = self.step(t)
        return logits

    # sampling
    def generate(self, prompt: str, max_new: int = 300, temperature: float = 0.8,
                 top_k: int = 40, top_p: float = 0.95, seed: int = 0,
                 repeat_penalty: float = 1.1):
        """Yield decoded text as it is produced.

        Tokens are byte-level, so a single token can hold half a UTF-8 sequence.
        An incremental decoder is the only way to stream without mangling
        multi-byte characters at chunk boundaries.
        """
        rng = np.random.default_rng(seed)
        self.reset()
        ids = self.tok.encode(prompt)
        if not ids:
            raise ValueError("empty prompt")
        logits = self.prefill(ids)
        out = list(ids)
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        for _ in range(max_new):
            lg = logits.astype(np.float64).copy()
            # kids' stories repeat themselves
            for prev in set(out[-64:]):
                lg[prev] /= repeat_penalty if lg[prev] > 0 else 1.0
            if temperature <= 1e-6:
                nxt = int(np.argmax(lg))
            else:
                lg /= temperature
                if top_k and top_k < lg.size:
                    thresh = np.sort(lg)[-top_k]
                    lg[lg < thresh] = -np.inf
                p = np.exp(lg - lg.max())
                p /= p.sum()
                if top_p and top_p < 1.0:
                    order = np.argsort(p)[::-1]
                    cum = np.cumsum(p[order])
                    cut = order[cum > top_p][1:] if (cum > top_p).any() else []
                    p[cut] = 0.0
                    p /= p.sum()
                nxt = int(rng.choice(p.size, p=p))
            if nxt == self.tok.eos_id:
                tail = dec.decode(b"", True)
                if tail:
                    yield tail
                return
            out.append(nxt)
            token_str = self.tok.id_to_token.get(nxt, "")
            raw = bytes(BYTE_DECODER.get(c, 63) for c in token_str)
            text = dec.decode(raw)
            if text:
                yield text
            logits = self.step(nxt)
