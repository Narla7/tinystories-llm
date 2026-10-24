# tinystories-llm

A 27M-parameter GPT trained from scratch on TinyStories. Not a wrapper, not a
fine-tune — tokenizer, model, training loop, sampling, checkpoints, all of it.

The target is the same one the [TinyStories paper](https://arxiv.org/abs/2305.07759)
hit: a model small enough to train on a free GPU in an hour that still writes
coherent, grammatical English, because the corpus is deliberately narrow instead
of the whole web. Their 28M model beat GPT-2-XL on their own prompts. That's the
bar.

## Run it

The weights are 55 MB and ship as a GGUF, so Ollama runs them natively. No
Python, no torch, no GPU.

```bash
ollama serve                       # if it isn't already running
python nemo.py --install           # downloads the GGUF, runs `ollama create nemo`
python nemo.py                     # the TUI
python nemo.py "Once upon a time"  # or one-shot, no UI
ollama run nemo                    # or skip the harness entirely
```

The TUI is curses, standard library only: history scrolls, Enter sends, Ctrl+Q
quits, and `/temp 0.9`, `/topk 40`, `/seed 7`, `/clear` adjust it live.

What it is good at: two or three paragraphs of simple narrative with a thing that
goes wrong and a gentle ending. What it is not: a chatbot. It was trained only on
children's stories, so ask it about the weather and it will write you a story
about the weather.

## The numbers

| | |
|---|---|
| params | 27.58M (8 layers, d_model 512, 8 heads, block 512) — measured, not estimated |
| vocab | 4096, byte-level BPE trained on this corpus |
| tokens | 548M in the corpus; the shipped run used 350M (0.64 epochs) |
| measured throughput | 49,845 tok/s training-only on one T4 |
| the run that exists | 5,341 steps, 122 min, val loss 1.2121 |
| full epoch would be | ~3.1 h |
| checkpoint size | ~0.33 GB each (fp32 weights + Adam state) |

`configs/tiny25m.json` holds all of it. `configs/smoke.json` is a 140K-parameter
toy for testing the plumbing in under a minute.

## Results

TinyStories, 27.58M params, 350M tokens, 122 minutes on a free Kaggle T4.

```
[eval] step  250  train 2.5017  val 2.4946
[eval] step  750  train 1.7197  val 1.7165
[eval] step 2000  train 1.4099  val 1.4256
[eval] step 3000  train 1.3132  val 1.3222
[eval] step 4000  train 1.2414  val 1.2537
[eval] step 5250  train 1.2102  val 1.2121
```

Val tracks train to within 0.01, so at 0.64 epochs this model is still
undertrained, not overfitting. It writes:

```
Once upon a time, in a big forest, there was a little boy named Tim. Tim had a
toy spear that he loved to play with. One day, he met a foolish squirrel named
Sam. Sam always wanted to play with Tim's spear. Tim didn't want to share his
spear, so he told Sam to go away. ... When they got to the ant hill, they found
out that the spear wasn't a toy after all! It was a magic spear that could make
wishes come true. Tim and Sam learned that sharing is important.
```

## Quickstart (local, CPU, smoke test only)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# streams only 400 docs over HTTP range requests -- a few MB, not 673 MB
python tokenize_data.py --out-dir data/sample --vocab-size 512 --limit-docs 400
python train.py --config configs/smoke.json
python sample.py --run-dir runs/smoke --prompt "Once upon a time"
```

That produces garbage text, and it should: 512-token vocab, 400 stories, 80
steps. It proves the tokenizer, the data loader, the loss, the atomic
checkpoint, the resume path, and the sampler all work before you spend a single
GPU-hour.

## The real run (Kaggle, free, no card)

Two script kernels. `kaggle/kaggle.json` API token needed (Kaggle -> Settings ->
Create New API Token).

```bash
# 1. build the tokenized corpus once (~10-20 min, CPU, no GPU quota burned)
kaggle/push.sh prep

# 2. when it finishes, open the kernel's Output pane and click
#    "Create Dataset from kernel output" -> name it tinystories-tokens, private

# 3. train (~1.2 h on a T4)
kaggle/push.sh train

# 4. pull the checkpoint down
kaggle/push.sh pull-train
```

The train kernel hardcodes `narla/tinystories-tokens` in
`kaggle/train/kernel-metadata.json` and `narla/tinystories-train-25m` as its id —
change both to your own username before pushing.

The code itself is cloned from GitHub by the kernels (`REPO_URL` env var, default
`https://github.com/Narla7/tinystories-llm`). If you'd rather not publish the
repo, upload it as a private Kaggle Dataset named `tinystories-llm-code` instead
and the kernels will find it at `/kaggle/input`.

## Durability

The thing that kills from-scratch runs is not a bad loss curve, it's a lost
checkpoint. Three defences, all in `train.py`:

1. **Atomic writes.** `atomic_save()` writes to `.tmp`, flushes, fsyncs, then
   `os.replace()`s into place and fsyncs the directory. On ext4 and NTFS the
   replace is atomic, so a power cut mid-save leaves the previous good file, not
   a truncated new one. `torch.save` alone does not guarantee this.
2. **Wall-clock cadence.** Checkpoints fire on elapsed minutes
   (`ckpt_minutes`), not step count, so the guaranteed loss window stays put
   when throughput drifts.
3. **Three slots + signal handlers.** `last.pt`, `prev.pt`, `best.pt`; SIGINT and
   SIGTERM save before exiting. Resume restores model, Adam moments, LR schedule
   position, step, RNG, and best-val, because Adam without its moments restarts
   badly.

Resume is automatic — rerun the same command. `--fresh` starts over.

## What to expect from the loss curve

  - step 0: loss ≈ ln(4096) ≈ 8.3
  - ~200 steps: under 5, the model has learned word shapes
  - ~2000 steps: around 2.5, grammar is mostly there
  - end: 1.6–2.0 for this model/corpus combination

If val loss turns up while train loss keeps falling, you are overfitting the
550M tokens; stop, or add dropout.

## Rebuilding the GGUF

`export_hf.py` turns a checkpoint into an HF GPT-2 layout, then llama.cpp's
`convert_hf_to_gguf.py` produces the GGUF:

```bash
python export_hf.py --ckpt out-train/runs/tiny25m/best.pt \
                    --tokenizer out-tok/nemo-tokenizer/tokenizer.json \
                    --out export/nemo
python /path/to/llama.cpp/convert_hf_to_gguf.py export/nemo \
       --outfile export/nemo-f16.gguf --outtype f16
```

One snag, because it will bite you: llama.cpp fingerprints the tokenizer by
hashing the token ids of a fixed test string, so a BPE you trained yourself can
never match a known family and the converter raises
`BPE pre-tokenizer was not recognized`. Our pre-tokenizer *is* HuggingFace
ByteLevel, which is the GPT-2 regex, so patch `conversion/base.py` to set
`res = "gpt-2"` immediately after `res = None` in `get_vocab_base_pre()`. This
only affects conversion; users of the released GGUF never touch it.

## Files

```
model.py            GPT: pre-LN, learned positions, GELU, tied embeddings
train.py            loop, cosine LR, fp16 autocast, atomic checkpoints, resume
tokenize_data.py    parquet -> BPE -> flat uint16 bins
sample.py           top-k sampling from a checkpoint
configs/            smoke.json, tiny25m.json
kaggle/prep/        script kernel: build the corpus once
kaggle/train/       script kernel: train
kaggle/push.sh      thin kaggle-CLI wrapper
```

## Data and licence

`karpathy/tinystories-gpt4-clean` on Hugging Face — one parquet, 673 MB on disk,
2.2 GB of text, GPT-4 stories only, normalised to printable ASCII. Licence is
CDLA-Sharing-1.0, which is a data licence with a sharing provision, not MIT.
Training on it and publishing the weights is the ordinary use. Redistributing
the dataset itself is where the terms start mattering.

## Known upgrade paths, in order of payoff

1. More data. TinyStories caps quality at ~25-35M params. Nothing else on this
   list beats more tokens.
2. RoPE + RMSNorm instead of learned positions + LayerNorm.
3. `torch.compile` on the GPU run, once the thing works.
4. Gradient accumulation tuning, then LR sweep. Boring, high payoff.
