---
license: mit
datasets:
  - roneneldan/TinyStories
language:
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - gpt2
  - tinystories
  - small-language-model
  - trained-from-scratch
  - educational
  - rocm
widget:
  - text: "Once upon a time"
  - text: "It was a cold fall morning"
  - text: "Tom and Sara found a big box in the garden"
model-index:
  - name: tinystories-gpt-51m
    results:
      - task:
          type: text-generation
          name: Causal Language Modeling
        dataset:
          name: TinyStories (validation)
          type: roneneldan/TinyStories
          split: validation
        metrics:
          - type: loss
            value: 1.3029
            name: Validation loss (nats/token)
          - type: perplexity
            value: 3.68
            name: Validation perplexity
---

# tinystories-gpt-51m

A 51.2M parameter GPT-2 style language model pretrained from scratch on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). It writes
simple, mostly coherent children's stories in the style of its training corpus,
and essentially nothing else.

This model is the artifact of a **learning project**
([gpt-spark](https://github.com/rprouse/gpt-spark)) whose goal was to expose every
moving part of a real pretraining run in one readable file: data packing, gradient
accumulation, mixed precision, LR scheduling, evaluation and checkpointing. The
transformer itself is unmodified Hugging Face `GPT2LMHeadModel` — deliberately, so
that the training loop is the thing being learned rather than reimplemented.

It is published for inspection and teaching, not because it is a useful model.

## Model details

|                      |                                         |
| -------------------- | --------------------------------------- |
| Architecture         | `GPT2LMHeadModel` (unmodified)          |
| Total parameters     | 51,213,824 (51.2M)                      |
| Non-embedding params | 25,220,096 (25.2M)                      |
| Layers               | 8                                       |
| Attention heads      | 8                                       |
| Embedding dim        | 512                                     |
| Context window       | 512 tokens                              |
| Tokenizer            | GPT-2 BPE, 50,257 tokens (reused as-is) |
| Dropout              | 0.0 (embd / resid / attn)               |
| Position embeddings  | Learned, exactly 512 slots              |
| Weight tying         | `lm_head` tied to `wte`                 |
| Checkpoint precision | float32 (204.9 MB safetensors)          |
| Language             | English                                 |

Half the parameter count is the token embedding table (`50257 × 512 = 25.7M`) — at
this scale, vocabulary dominates. Each of the 8 transformer blocks is only ~3.15M
parameters.

For comparison with the smallest published GPT-2:

|               | this model | GPT-2 small |
| ------------- | ---------- | ----------- |
| layers        | 8          | 12          |
| heads         | 8          | 12          |
| embedding dim | 512        | 768         |
| context       | 512        | 1024        |
| total params  | 51.2M      | 124M        |

## Usage

```python
from transformers import pipeline

gen = pipeline("text-generation", model="rprouse/tinystories-gpt-51m")
print(gen("Once upon a time", max_new_tokens=180, do_sample=True,
          temperature=0.8, top_k=50, top_p=0.95)[0]["generated_text"])
```

Or explicitly:

```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

repo = "rprouse/tinystories-gpt-51m"
tok = GPT2TokenizerFast.from_pretrained(repo)
model = GPT2LMHeadModel.from_pretrained(repo).eval()

ids = tok("Once upon a time", return_tensors="pt").input_ids
# Position embeddings cover exactly 512 slots: prompt + new tokens must fit inside.
n = min(180, model.config.n_positions - ids.shape[1])
out = model.generate(ids, max_new_tokens=n, do_sample=True, temperature=0.8,
                     top_k=50, top_p=0.95, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0], skip_special_tokens=True))
```

**The context window is a hard limit.** GPT-2 uses *learned* position embeddings, so
there are exactly `n_positions = 512` of them. Requesting `prompt + max_new_tokens`
beyond 512 indexes off the end of `wpe`. On CUDA that raises an error; under ROCm's
flash-attention kernel it was observed to abort the process rather than raise a
clean Python exception. Clamp `max_new_tokens` as shown above.

## Training

### Data

The full TinyStories corpus, tokenised once with the GPT-2 BPE tokenizer and packed
nanoGPT-style: every document gets `<|endoftext|>` appended, everything is
concatenated into one flat `uint16` array per split, and batches are drawn by
slicing 512-token windows at random offsets.

| Split      | Tokens      |
| ---------- | ----------- |
| train      | 473,992,236 |
| validation | 4,765,918   |

There are no document boundaries within a batch and no padding: every token position
is a valid training example, and the model sometimes learns to read across an EOS.
That is the deliberate tradeoff — maximum data efficiency for slightly noisy document
boundaries.

### Hyperparameters

| Setting               | Value                                                           |
| --------------------- | --------------------------------------------------------------- |
| Optimiser             | AdamW, `betas=(0.9, 0.95)`, fused                                |
| Weight decay          | 0.1 on matrices (`dim() >= 2`), 0.0 on biases / LayerNorm gains  |
| Peak LR               | 1e-3                                                             |
| Schedule              | Cosine decay, 500 linear warmup steps                            |
| Grad clipping         | Global norm 1.0                                                  |
| Micro-batch           | 32                                                               |
| Gradient accumulation | 2                                                                |
| Tokens per step       | 32 × 2 × 512 = 32,768                                            |
| Total steps           | 30,000                                                           |
| Tokens seen           | ~983M (~2.07 epochs)                                             |
| Precision             | bf16 autocast, fp32 master weights                               |
| Seed                  | 1337                                                             |

Weight decay is withheld from LayerNorm gains on purpose: decaying a gain pulls it
toward zero, which fights the normalisation it exists to perform.

### Hardware

Trained on a single **AMD Strix Halo EVO-X2** (Ryzen AI Max+ 395, `gfx1151`) on
Windows using AMD's ROCm PyTorch builds. Roughly **12 hours** wall clock, in two
15,000-step runs of about 6 hours each.

### A note on the second half of training

The run was done in two parts: 15,000 steps, then `--resume --steps 30000`. Because
`--steps` also defines the cosine schedule's horizon, resuming built a *fresh*
30,000-step cosine and loaded the saved scheduler state into it at step 15,000. The
learning rate therefore jumped from ~0 (the end of the first cosine) back up to
5.13e-4 (the midpoint of the new one) and decayed a second time.

The measured cost: validation loss was 1.319 at step 15,000 and 1.308 at step
30,000. **The second 15,000 steps — half the total compute — bought roughly 0.01
nats.** The model spent them relearning ground it had already covered. A single
30,000-step run under one continuous cosine would very likely have done better for
the same cost. This is documented rather than quietly re-run because it is the more
instructive artifact.

## Evaluation

A single deterministic pass over the **entire** packed validation split:
non-overlapping 512-token windows, token-weighted mean negative log-likelihood.
Reproduce with [`eval_ppl.py`](https://github.com/rprouse/gpt-spark/blob/main/eval_ppl.py):

```bash
uv run eval_ppl.py --out ckpt_8x512
```

| Metric                       | Value             |
| ---------------------------- | ----------------- |
| Validation loss (nats/token) | **1.3029**        |
| Validation perplexity        | **3.680**         |
| Bits per token               | 1.8797            |
| Windows / predicted tokens   | 9,308 / 4,756,388 |

For context, the training loop logs a cheaper estimate — the mean over 20 random
batches — which across the final ten checkpoints ranged 1.287 to 1.321. The full-set
number lands inside that band, so the sampled estimate was unbiased but far too noisy
to quote a single best value from. The full-set figure is the one reported here.

No downstream benchmarks were run. Perplexity on the corpus a model was trained on
measures fit to that corpus and nothing more.

## Sample outputs

All generated with `temperature=0.8, top_k=50, top_p=0.95, seed=1337`.

> **Once upon a time**, there was a little girl named Lily. She loved to play
> outside in the sunshine. One day, she went to the park with her mommy and saw a
> beautiful butterfly. "Wow, mommy! Look at the pretty butterfly!" she said.
>
> Suddenly, Lily saw a man who was very impatient. He was trying to catch the
> butterfly to catch it. "Mommy, why is he so impatient?" she asked. "He wants to
> catch the butterfly," her mommy replied.

> **It was a cold fall morning.** The wind was blowing and it was bitter outside.
> The little girl was sad and started to cry.
>
> Suddenly, a rainbow appeared in the sky! It was so beautiful and colourful. The
> little girl stopped crying and smiled. […] She felt so happy that she hugged her
> mum. Then she skipped around the park one last time, feeling safe and warm inside
> her mum's arms.

Note the flaws visible even in the good samples: *"trying to catch the butterfly to
catch it"*, and the *"so happy … so happy"* repetition. Short-range fluency is solid;
phrase-level redundancy is not.

Given an out-of-domain prompt, the model does not decline or hedge — it converts the
prompt into a TinyStories fable:

> **In 1789, the French Revolution began because** the family had to move away from
> the shore.
>
> One day, a brave little boy decided to take the family on an adventure. He was
> determined to help them, and they followed him. […] From that day on, the family
> lived happily ever after on the shore.

This is the single most useful thing to understand about the model, and no perplexity
number would have told you it.

## Intended use

**In scope:**

- Teaching and studying the mechanics of a pretraining run — a real, small,
  fully reproducible checkpoint to load, inspect, probe and fine-tune.
- A baseline or starting point for TinyStories-scale experiments: tokenizer
  ablations, model-shape sweeps, distillation, and interpretability work where a
  51M-parameter model that runs on a laptop is a feature.
- Generating simple English children's stories, in the narrow style of the corpus,
  for demos and tests.

**Out of scope:**

- Any factual, informational or advisory use. The model has no knowledge base; the
  French Revolution sample above is what "answering a question" looks like.
- Instruction following, chat or dialogue. This is a base LM with no
  instruction-tuning or RLHF of any kind.
- Any production or user-facing deployment, and any use where a plausible-sounding
  wrong output carries a cost.
- Content aimed at children without human review, despite the corpus. Being trained
  on children's stories is not the same as being safe for children.

## Limitations and biases

- **The domain is extremely narrow.** TinyStories is synthetic, deliberately simple
  English pitched at a 3–4 year old vocabulary. The model has never seen code, code
  switching, technical prose, long-form argument or adult register, and its outputs
  collapse toward simple narrative regardless of the prompt.
- **No world knowledge.** It was trained on ~474M tokens of synthetic fiction.
  Anything factual it emits should be assumed false.
- **512-token context**, with the hard cutoff described under Usage.
- **Repetition and phrase looping**, visible in the samples above.
- **The tokenizer is oversized for the corpus.** The GPT-2 BPE vocabulary was reused
  unchanged, so 50.2% of the parameters sit in an embedding table covering CJK, emoji
  and other scripts that never appear in TinyStories. Training a corpus-specific BPE
  vocabulary is the most promising next change: it would move a large share of the
  parameter budget out of the embedding table and into the transformer blocks.
- **It inherits the biases of its corpus, and of the model that generated it.**
  TinyStories was synthesised by GPT-3.5 and GPT-4, so this model is a distillation
  of those models' distribution over children's stories — including whatever skews in
  names, family structures, gender roles and settings that carries. No bias
  evaluation was performed.
- **No safety training.** Nothing prevents the model from producing unsuitable output
  given an adversarial prompt.

## Reproducing

```bash
git clone https://github.com/rprouse/gpt-spark && cd gpt-spark
uv sync
uv run gpt_train.py --steps 30000          # one continuous cosine — see the note above
uv run gpt_train.py --sample "Once upon a time"
uv run eval_ppl.py                         # the validation figures reported above
uv run tensorboard --logdir runs
```

The training loop, its rationale and a full command-line reference are in the
project [README](https://github.com/rprouse/gpt-spark).

## Licence and attribution

- **Weights and training code:** MIT.
- **Training data:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
  by Ronen Eldan and Yuanzhi Li, licensed **CDLA-Sharing-1.0**. Users of this model
  should review that licence with respect to their own use of the data.
- **Tokenizer and architecture:** GPT-2, by OpenAI, via Hugging Face `transformers`
  (MIT).

```bibtex
@misc{eldan2023tinystories,
  title         = {TinyStories: How Small Can Language Models Be and Still Speak Coherent English?},
  author        = {Ronen Eldan and Yuanzhi Li},
  year          = {2023},
  eprint        = {2305.07759},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```
