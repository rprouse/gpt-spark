# GPT Spark

A small GPT-2 style language model, trained from scratch on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

This is a **learning project**. The goal is not a useful model, it is to
see every moving part of a real pretraining run in one readable file
([main.py](main.py), ~220 lines) and be able to change any of them.

The deliberate design choice: *don't reimplement the transformer*. The
architecture, tokenizer, LR schedule and sampling all come from Hugging Face
`transformers`. What's left, and what this project is actually about, is the
**training loop**: how you pack data, accumulate gradients, manage precision,
schedule the learning rate, evaluate, and checkpoint.

## Quick start

```bash
uv sync

uv run main.py                                  # train with the defaults (51M params)
uv run main.py --max_docs 50000 --steps 500     # quick smoke test
uv run main.py --resume --steps 30000           # continue where a run left off
uv run main.py --sample "Once upon a time"      # generate only, no training
uv run tensorboard --logdir runs                # watch loss / lr / sample text
```

Training writes a checkpoint into `--out` (default `ckpt_8x512/`) every
`--eval_every` steps, so a run is always resumable and always samplable.

## Dependencies

This project targets an **AMD Strix Halo EVO-X2** (Ryzen AI Max+ 395, `gfx1151`)
running Windows, using
[AMD's ROCm PyTorch builds](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?fam=ryzen&gpu=amd-ryzen-ai-max-pro-395&os=windows&rocm-ver=10.0.0&pytorch-ver=2.12.0&i=pip&w=compute&gfx=gfx1151).
[pyproject.toml](pyproject.toml) pins the ROCm wheels from AMD's index; `uv sync`
handles the rest. The code itself is device-agnostic and falls back to CPU.

## What the model is

The defaults in [main.py](main.py) build a **51.2M parameter** GPT-2 variant:

|                          | this project     | GPT-2 small |
| ------------------------ | ---------------- | ----------- |
| layers (`--n_layer`)     | 8                | 12          |
| heads (`--n_head`)       | 8                | 12          |
| embedding dim (`--n_embd`) | 512            | 768         |
| context (`--block_size`) | 512              | 1024        |
| vocabulary               | 50257 (GPT-2 BPE) | 50257      |
| **total params**         | 51.2M            | 124M        |
| non-embedding params     | 25.2M            | 85M         |

Half the parameter count is the token embedding table (`50257 × 512 = 25.7M`),
a reminder that at small scale, vocabulary dominates. The 8 transformer blocks
are only 3.15M parameters each.

Dropout is set to `0.0` throughout. TinyStories is large relative to this model,
so the run is data-rich rather than overfitting-prone, and regularisation would
just slow learning down.

At 15,000 iterations, the loss was down to *1.321* so I ran again with
`uv run main.py --resume --steps 30000`. This dropped the loss slightly to *1.270*.

This produces somewhat coherent text. To improve, I might want to custom train a BPE tokenizer for the smaller vocabulary of this training text.

```sh
uv run .\main.py --sample "It was a cold fall morning"
```

It was a cold fall morning. The wind was blowing and it was bitter outside. The little girl was sad and started to cry.

Suddenly, a rainbow appeared in the sky! It was so beautiful and colourful. The little girl stopped crying and smiled.

The rainbow made the little girl happy. She ran around the park, chasing the rainbow. When the sun set, the rainbow disappeared.

The little girl was so happy. She felt so happy that she hugged her mum. Then she skipped around the park one last time, feeling safe and warm inside her mum's arms.

## How the code works

[main.py](main.py) runs top to bottom inside a single `if __name__ == "__main__":`
block. Each section is described below.

### 1. The `__main__` guard is load-bearing

Not stylistic. `datasets`' `.map(num_proc=...)` spawns worker processes, and on
Windows a spawned worker **re-executes this file** to rebuild `__main__`. Without
the guard, every worker would re-import torch, build the model on the GPU, and
call `build_data()` again — spawning another generation of workers, recursively.

The same constraint explains why the tokenizer is passed to `tokenize()` through
`fn_kwargs` instead of being captured as a closure ([main.py:103-113](main.py#L103-L113)):
inside a worker, `tok` was never defined, so a closure over it would `NameError`.
`fn_kwargs` is pickled and travels *by value*.

### 2. Data: pack once, sample forever ([main.py:84-126](main.py#L84-L126))

`build_data()` follows the nanoGPT approach:

1. Tokenise the whole corpus in parallel across every CPU core.
2. Append `<|endoftext|>` to each document as a separator.
3. Concatenate **everything** into one flat `uint16` array per split and write it
   to a `.bin` file.
4. Reopen it with `np.memmap`, so the OS pages tokens in on demand and the array
   never has to fit in RAM.

The `.bin` cache is keyed by dataset name, so the expensive tokenisation happens
exactly once per configuration.

`get_batch()` then draws training examples by picking `batch_size` random offsets
into that array and slicing `block_size` tokens from each. There are no document
boundaries, no padding, and no shuffled epochs. Every token position is a valid
training example, and the model occasionally learns to read across an EOS. That's
the tradeoff: maximum data efficiency, slightly noisy document boundaries.

Note the labels: `model(x, labels=x)`. Hugging Face shifts labels internally, so
predicting token *n+1* from tokens *≤n* needs no manual offsetting.

### 3. Mixed precision ([main.py:65-76](main.py#L65-L76))

Chosen automatically:

- **bf16** on compute capability 8.0+ (Ampere and later, and ROCm), wide
  exponent range, so no loss scaling needed.
- **fp16 + `GradScaler`** on older CUDA cards, which lack native bf16. The scaler
  multiplies the loss up before `.backward()` to keep small gradients from
  flushing to zero, then unscales before the optimiser step.
- **fp32** on CPU, unless `--bf16` is forced.

The `GradScaler` stays in the loop unconditionally but is a no-op when disabled,
so one code path serves all three cases.

### 4. Optimiser ([main.py:165-172](main.py#L165-L172))

AdamW with `betas=(0.9, 0.95)`, the GPT-2 convention, a shorter second-moment
memory than PyTorch's `0.999` default.

Parameters are split into two groups: weight decay `0.1` on matrices
(`dim() >= 2`), and `0.0` on biases and LayerNorm gains. Decaying a LayerNorm
gain pulls it toward zero, which fights the normalisation it exists to perform.

Learning rate follows a cosine schedule with 500 warmup steps. Warmup matters
because Adam's second-moment estimates are unreliable in the first few dozen
steps; a full-size LR there can wreck the model before it stabilises.

### 5. The training step ([main.py:201-222](main.py#L201-L222))

```
for each step:
    for grad_accum micro-batches:      # accumulate gradients
        loss = model(x, labels=x).loss / grad_accum
        loss.backward()
    unscale → clip grad norm to 1.0 → optimiser step → scheduler step → zero grads
```

**Gradient accumulation** decouples the batch size that fits in VRAM from the
batch size you want statistically. With the defaults:

```
32 (batch_size) × 2 (grad_accum) × 512 (block_size) = 32,768 tokens per step
```

Dividing the loss by `grad_accum` before backward makes the accumulated gradient
the *mean* over the effective batch rather than the sum, so the effective
learning rate doesn't change when you trade `batch_size` for `grad_accum`.

**Gradient clipping** to norm 1.0 caps the occasional pathological batch. Without
it, one bad update can undo thousands of good ones.

### 6. Evaluation, sampling, checkpointing

Every `--eval_every` steps the loop:

- averages validation loss over 20 batches,
- generates a sample from `"Once upon a time"` (temperature 0.8, top-k 50, top-p 0.95),
- logs both to TensorBoard,
- saves model + tokenizer + optimiser/scheduler/step state.

The sample text in TensorBoard is the most honest progress signal you have. Val
loss tells you it's improving, but only the samples tell you *how*: gibberish →
word-shaped noise → grammatical nonsense → actual little stories.

`sample()` reads the context window off `raw_model.config`, not off `args`
([main.py:147-160](main.py#L147-L160)). GPT-2 has *learned* position embeddings for
exactly `n_positions` slots, so prompt + generated tokens must fit inside them,
overrunning indexes off the end of `wpe`, which is a hard abort in ROCm's
flash-attention kernel rather than a clean Python error.

Checkpointing splits into two files by design: `save_pretrained()` writes a
standard HF model directory (loadable by anything in the ecosystem), while
`train_state.pt` holds the optimiser moments, scheduler position and step
count — the things `--resume` needs but a released model doesn't.

## Command-line reference

| flag                                | default                   | what it does                                    |
| ----------------------------------- | ------------------------- | ----------------------------------------------- |
| `--dataset`                         | `roneneldan/TinyStories`  | any HF dataset with a `text` column             |
| `--max_docs`                        | *all*                     | subsample training docs (fast smoke tests)      |
| `--block_size`                      | 512                       | context window in tokens                        |
| `--n_layer` / `--n_head` / `--n_embd` | 8 / 8 / 512             | model shape                                     |
| `--batch_size`                      | 32                        | micro-batch — lower this first if you hit OOM   |
| `--grad_accum`                      | 2                         | micro-batches per optimiser step                |
| `--lr`                              | 1e-3                      | peak LR (high, but fine at this size)           |
| `--warmup`                          | 500                       | linear warmup steps before cosine decay         |
| `--steps`                           | 15000                     | total optimiser steps                           |
| `--eval_every`                      | 250                       | eval + sample + checkpoint interval             |
| `--out`                             | `ckpt_8x512`              | checkpoint + tokenised `.bin` cache directory   |
| `--resume`                          | off                       | load model *and* optimiser state, continue      |
| `--compile`                         | off                       | `torch.compile` — faster steps, slow first step |
| `--bf16`                            | off                       | force bf16 autocast on CPU                      |
| `--sample`                          | —                         | prompt to generate from; skips training         |

## Layout

```
main.py          the entire project
pyproject.toml   ROCm-pinned dependencies (uv)
ckpt_8x512/      current 8-layer × 512-dim checkpoint + token cache
ckpt/            an earlier run's checkpoint
runs/            TensorBoard event files, one directory per run
```

`ckpt*` is gitignored. Checkpoints and the tokenised `.bin` files are large and
fully reproducible from the code.

## Things to try

- **Sweep the model shape.** `--n_layer 12 --n_embd 768` is GPT-2 small. Watch
  where val loss stops improving for the compute you're willing to spend.
- **Change the corpus.** `--dataset` takes any HF dataset with a `text` column.
  TinyStories is deliberately simple; a harder corpus needs a bigger model before
  the samples stop being gibberish.
- **Turn `--compile` on** and measure the step-time difference against the
  one-time compilation cost.
- **Resolve the open `TODO(you)`** at [main.py:95-101](main.py#L95-L101):
  `--max_docs` currently shrinks the *train* split only, so a 2000-doc smoke test
  still tokenises all 22k validation docs. 0.4M train tokens against 4.8M val,
  with that tokenisation dominating startup. The tradeoff is real: a bigger
  validation set gives a lower-variance loss estimate, a smaller one makes short
  runs start fast. Pick a policy and implement it.
