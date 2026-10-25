# Nemo

A 27.58M parameter language model that writes children's stories, running on a
CPU with nothing but numpy.

```
$ python nemo.py "The little robot wanted a friend"

The little robot wanted a friend to help him. He thought, "Maybe I can find a
friend for me too!" So he went to look for a friend. He found a friend, a big
bear. The big bear was very strong and could lift heavy things.

But the little robot was not happy. He still wanted a friend. The big bear saw
the little robot was sad and asked, "Why are you sad?" The little robot said,
"I want a friend like you, but I can't."

The big bear thought for a moment and then said, "I know! You are very nice and
I will be your friend." The little robot was so happy. They both played together
and had lots of fun. From that day on, the little robot was never sad again
because he had a friend to share his toys with.
```

That is 27 million parameters writing a story with a beginning, a problem and an
ending, on a 2014 desktop CPU, with no GPU and no external runtime. `numpy` is
the entire dependency list.

## Run it

```bash
git clone https://github.com/Narla7/tinystories-llm
cd tinystories-llm/nemo

python -m venv .venv
.venv/bin/pip install numpy
.venv/bin/python nemo.py
```

Works in fish, bash and zsh. The first run fetches `nemo_weights.npz` (55 MB)
from the release into that directory.

One-shot, no UI:

```bash
.venv/bin/python nemo.py "Once upon a time, there was a little girl named Lily"
.venv/bin/python nemo.py "The dog was" --temp 0.7 --topk 30 --seed 4 --len 200
```

If you already have numpy in your environment, drop the `.venv/bin/` prefix and
run `python nemo.py`.

## The terminal UI

Enter sends. Ctrl+Q quits, Ctrl+L clears, Ctrl+N rolls the seed, arrows scroll
back through the transcript, and the status bar shows live tokens per second.

Slash commands:

```
/temp 0.9    sampling temperature, 0 is greedy
/topk 40     keep the 40 most likely tokens
/seed 7      fix the seed; change it for a different story
/len 300     maximum tokens per generation
/clear       wipe the transcript
/help        what the model is
```

## What it is

| | |
|---|---|
| parameters | 27,579,392 |
| shape | 8 layers, 8 heads, d_model 512, context 512, pre-LayerNorm, GELU, tied embeddings |
| vocabulary | 4096, byte-level BPE fitted to the corpus |
| trained on | 350M tokens of [TinyStories](https://arxiv.org/abs/2305.07759), which is why the English is clean and the world knowledge is nil |
| loss | 1.2102 train, 1.2121 validation |
| weights | 55 MB float16 |
| speed | about 55 tok/s with other work on the machine, ~90 on an idle one |
| good at | two or three paragraphs of simple narrative with a thing that goes wrong and a gentle ending |
| not good at | everything else |

It is not a chatbot. Ask it a question and it answers with a story, because
stories are the only thing it has ever seen. It also has no memory: 512 tokens
of context, and past that the oldest tokens fall out of the window.

## How the engine works

`nemo_engine.py` is 250 lines of numpy and does three things.

The forward pass is plain GPT-2: pre-LayerNorm blocks, multi-head attention, a
4x GELU MLP, and the output head tied to the token embedding.

The KV cache is what makes it usable. Without one, every new token re-runs the
whole 512-token context and throughput falls from ~85 tok/s to about 0.2. With
it, each token costs a single pass over 27.58M parameters.

The tokenizer is written from scratch, because `tokenizers` is a Rust extension
and depending on it would have cost more than the model. The vocabulary and merge
list live in a 145 KB JSON beside the code and the encoder runs in Python.

Correctness was checked against a PyTorch reference implementation rather than
eyeballed:

```
logits over the vocabulary   max abs diff 0.00314, mean 0.00072
top-5 next-token ranking     identical
token ids on test strings    10/10 exact, including accents, quotes,
                             newlines, contractions and digits
```

The weights are float16 in a single `.npz`, cast to float32 once at load. There
is no quantisation trick and nothing is approximated except the error function,
which numpy lacks — the tanh shortcut is a different function from the one the
model was trained with, so `nemo_engine.py` implements erf properly instead.

## Licence

MIT, see [LICENSE](LICENSE).

The training data was `karpathy/tinystories-gpt4-clean`, published under
CDLA-Sharing-1.0. That covers the dataset rather than this repository, but if you
plan to use the weights commercially, read it instead of assuming MIT settles
the question.
