"""
eval_ppl.py - full-validation-set perplexity for a gpt_train.py checkpoint.

    uv run eval_ppl.py                          # evaluate ckpt_8x512
    uv run eval_ppl.py --out ckpt_12x768        # a different checkpoint
    uv run eval_ppl.py --batch_size 8           # lower this first on OOM

The training loop's own eval (`evaluate()` in gpt_train.py) averages 20 random
batches, which is the right tradeoff mid-run: cheap enough to call every 250 steps,
good enough to watch a curve descend. It is not good enough to *quote*. Across the
last ten checkpoints of the 30k-step run it ranged 1.287-1.321 nats, so reporting
the best of those would be reporting the low end of the noise rather than the model.

This makes exactly one pass over every token in the packed validation .bin. Same
number every time, and it's the one that belongs in a model card.
"""
import argparse, math, os
import numpy as np
import torch
from transformers import GPT2LMHeadModel

p = argparse.ArgumentParser()
p.add_argument("--out",        default="ckpt_8x512", help="checkpoint dir (also holds the .bin cache)")
p.add_argument("--dataset",    default="roneneldan/TinyStories", help="picks which .bin to read")
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--bf16",       action="store_true", help="force bf16 autocast (auto on CUDA)")
args = p.parse_args()

# Same .bin naming scheme build_data() uses, so this reads the cache training wrote.
bin_path = f"{args.out}/{args.dataset.replace('/', '_')}_validation.bin"
if not os.path.exists(bin_path):
    raise SystemExit(f"no validation cache at {bin_path} - run gpt_train.py first")

# Precision picked exactly as in gpt_train.py: evaluating under different numerics
# than training would make the number incomparable to the logged val curve.
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
    amp_dtype = torch.bfloat16
elif device == "cuda":
    amp_dtype = torch.float16
else:
    amp_dtype = torch.bfloat16 if args.bf16 else None
autocast = torch.autocast(device_type=device, dtype=amp_dtype or torch.float32,
                          enabled=amp_dtype is not None)

model = GPT2LMHeadModel.from_pretrained(args.out, attn_implementation="sdpa").to(device).eval()
block = model.config.n_positions   # off the checkpoint, not a flag - see sample() in gpt_train.py

arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
n_win = len(arr) // block
print(f"{len(arr):,} val tokens -> {n_win:,} non-overlapping {block}-token windows "
      f"on {device} ({amp_dtype or torch.float32})")

# Windows are non-overlapping and block-wide. A 512-token window yields 511
# predictions: its first token has no context, so it is scored by nothing. The
# alternative - sliding the window one token at a time so every position gets full
# context - is 512x the compute for a slightly lower number, and isn't what the
# training curve measured either.
total_nll, total_tok = 0.0, 0
with torch.no_grad():
    for s in range(0, n_win, args.batch_size):
        idx = range(s, min(s + args.batch_size, n_win))
        x = torch.from_numpy(
            np.stack([arr[i * block:(i + 1) * block].astype(np.int64) for i in idx])
        ).to(device, non_blocking=True)
        with autocast:
            logits = model(x).logits
        # Shift by hand rather than passing labels=x. HF would return a per-batch
        # *mean*, and averaging those means weights a short final batch as heavily
        # as a full one. Summing NLL and dividing once at the end is token-weighted.
        nll = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            x[:, 1:].reshape(-1),
            reduction="sum",
        )
        total_nll += nll.item()
        total_tok += x[:, 1:].numel()
        if s % (args.batch_size * 20) == 0:
            print(f"  {s:,}/{n_win:,} windows | running ppl {math.exp(total_nll/total_tok):.4f}",
                  flush=True)

loss = total_nll / total_tok
print(f"\nwindows {n_win:,} | predicted tokens {total_tok:,}")
print(f"val loss (nats/token) {loss:.4f}")
print(f"val perplexity        {math.exp(loss):.4f}")
print(f"bits per token        {loss/math.log(2):.4f}")
