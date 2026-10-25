# Nemo

The model, its engine and its terminal UI live in this directory. Setup and
usage are in the [repository README](../README.md).

```bash
python -m venv .venv
.venv/bin/pip install numpy
.venv/bin/python nemo.py
```

Files here:

```
nemo.py              terminal UI, standard library only
nemo_engine.py       forward pass + byte-level BPE, numpy only
nemo_tokenizer.json  vocabulary and merges
nemo_weights.npz     55 MB, downloaded from the release on first run
```
