"""
main.py - GPT-2 style pretraining with the standard tooling.

    uv sync

    uv run main.py                                  # TinyStories, ~16M params
    uv run main.py --max_docs 50000 --steps 500     # quick smoke test
    uv run main.py --n_layer 8 --n_embd 512 --n_head 8 --steps 20000
    uv run main.py --resume --steps 30000           # continue a run
    uv run main.py --sample "Once upon a time"      # generate from checkpoint
    uv run tensorboard --logdir runs

Libraries doing the work:
  transformers  GPT2Config / GPT2LMHeadModel (the real architecture), tokenizer,
                cosine LR schedule with warmup, generate() with top-k / top-p
  datasets      download + parallel tokenisation of the corpus
  torch         AdamW, autocast (bf16), torch.compile, grad clipping
  tensorboard   loss / lr curves and sample text per eval
"""
import argparse, itertools, os, sys, time
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast,
                          get_cosine_schedule_with_warmup)

# GPT-2's BPE vocabulary covers CJK, emoji and other non-Latin text, and an untrained
# model emits it freely. The Windows console is cp1252 by default and raises
# UnicodeEncodeError on the first such token, so print through UTF-8 instead.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Everything below runs in the parent process only. datasets' .map(num_proc=...)
# starts its workers with the "spawn" start method, and a spawned worker rebuilds
# __main__ by re-executing this file. Without this guard every worker would
# re-import torch, rebuild the model on the GPU, and call build_data() again -
# which spawns another generation of workers, and so on.
if __name__ == "__main__":
    # ---------------- args ----------------
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="roneneldan/TinyStories")
    p.add_argument("--max_docs",   type=int, default=None, help="subsample training docs")
    p.add_argument("--block_size", type=int, default=256)   # GPT-2: 1024
    p.add_argument("--n_layer",    type=int, default=4)     # GPT-2 small: 12
    p.add_argument("--n_head",     type=int, default=4)     # GPT-2 small: 12
    p.add_argument("--n_embd",     type=int, default=256)   # GPT-2 small: 768
    p.add_argument("--batch_size", type=int, default=8)     # micro-batch; 8 fits a 4GB card
    p.add_argument("--grad_accum", type=int, default=8)     # effective batch = 8*8*256 = 16k tokens
    p.add_argument("--lr",         type=float, default=6e-4)
    p.add_argument("--warmup",     type=int, default=200)
    p.add_argument("--steps",      type=int, default=5000)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--out",        default="ckpt")
    p.add_argument("--resume",     action="store_true")
    p.add_argument("--compile",    action="store_true")
    p.add_argument("--bf16",       action="store_true", help="force bf16 autocast (auto on CUDA)")
    p.add_argument("--sample",     type=str, default=None, help="prompt; generate only, no training")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337); np.random.seed(1337)
    os.makedirs(args.out, exist_ok=True)
    # Mixed precision: bf16 on Ampere+ (compute capability 8.x), fp16 + loss scaling on older
    # CUDA cards (no native bf16), fp32 on CPU unless --bf16 is given.
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        amp_dtype = torch.bfloat16
    elif device == "cuda":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16 if args.bf16 else None
    autocast = torch.autocast(device_type=device, dtype=amp_dtype or torch.float32,
                              enabled=amp_dtype is not None)
    scaler = torch.amp.GradScaler(device, enabled=(amp_dtype == torch.float16))
    print(f"precision: {amp_dtype or torch.float32}")

    # ---------------- tokenizer ----------------
    # GPT-2's actual BPE vocabulary (50257 tokens). EOS doubles as the document separator.
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    # ---------------- data ----------------
    def build_data():
        """Tokenise the corpus once, pack every document end-to-end with EOS between,
        and cache as a flat uint16 array per split (nanoGPT-style)."""
        tag = args.dataset.replace("/", "_") + (f"_{args.max_docs}" if args.max_docs else "")
        paths = {s: f"{args.out}/{tag}_{s}.bin" for s in ("train", "validation")}
        if all(os.path.exists(v) for v in paths.values()):
            return {s: np.memmap(v, dtype=np.uint16, mode="r") for s, v in paths.items()}

        ds = load_dataset(args.dataset)
        if args.max_docs:
            ds["train"] = ds["train"].select(range(args.max_docs))
            # TODO(you): --max_docs shrinks train only, so a 2000-doc smoke test still
            # tokenises all 22k validation docs - 0.4M train tokens against 4.8M val,
            # and that tokenisation dominates startup. Decide how validation scales:
            #   - leave as-is:  val loss stays comparable across every run size
            #   - fixed floor:  ds["validation"].select(range(min(len(...), 2000)))
            #   - proportional: max_docs // 10
            # Bigger val = lower-variance loss estimate; smaller = short runs start fast.

        # The tokenizer arrives via fn_kwargs rather than as a captured global: a
        # spawned worker re-executes this file, where the __main__ guard above means
        # `tok` is never defined, so a closure over it would NameError in the worker.
        # fn_kwargs is pickled and shipped with the job, so it travels by value.
        def tokenize(batch, tok):
            ids = tok(batch["text"])["input_ids"]
            return {"ids": [x + [tok.eos_token_id] for x in ids],
                    "len": [len(x) + 1 for x in ids]}

        ds = ds.map(tokenize, batched=True, num_proc=os.cpu_count(), fn_kwargs={"tok": tok},
                    remove_columns=ds["train"].column_names, desc="tokenising")
        for split, path in paths.items():
            total = int(np.sum(ds[split]["len"]))
            arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total,))
            arr[:] = np.fromiter(itertools.chain.from_iterable(ds[split]["ids"]),
                                 dtype=np.uint16, count=total)
            arr.flush()
            print(f"{split}: {total/1e6:.1f}M tokens -> {path}")
        return {s: np.memmap(v, dtype=np.uint16, mode="r") for s, v in paths.items()}

    def get_batch(arr):
        ix = np.random.randint(0, len(arr) - args.block_size - 1, args.batch_size)
        x = torch.stack([torch.from_numpy(arr[i:i + args.block_size].astype(np.int64)) for i in ix])
        return x.to(device, non_blocking=True)

    # ---------------- model ----------------
    if args.resume or args.sample:
        model = GPT2LMHeadModel.from_pretrained(args.out, attn_implementation="sdpa")
    else:
        cfg = GPT2Config(vocab_size=len(tok), n_positions=args.block_size,
                         n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head,
                         embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,   # no dropout at this data size
                         bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                         attn_implementation="sdpa")   # constructor takes it via config, not as a kwarg
        model = GPT2LMHeadModel(cfg)
    model.to(device)
    raw_model = model
    if args.compile:
        model = torch.compile(model)

    n_all = sum(p.numel() for p in raw_model.parameters())
    n_emb = raw_model.transformer.wte.weight.numel() + raw_model.transformer.wpe.weight.numel()
    print(f"{n_all/1e6:.1f}M params total, {(n_all-n_emb)/1e6:.1f}M non-embedding, on {device}")

    @torch.no_grad()
    def sample(prompt, n=150):
        """Generate a continuation, clipped to the context window. GPT-2 has learned
        position embeddings for exactly n_positions slots, so prompt + new tokens must
        fit inside them; asking for more indexes off the end of wpe (a hard abort in
        ROCm's flash-attention kernel). Read the window off the model, not off args,
        so --sample against a checkpoint uses that checkpoint's real size."""
        raw_model.eval()
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        n = max(1, min(n, raw_model.config.n_positions - ids.shape[1]))
        out = raw_model.generate(ids, max_new_tokens=n, do_sample=True, temperature=0.8,
                                 top_k=50, top_p=0.95, pad_token_id=tok.eos_token_id)
        raw_model.train()
        return tok.decode(out[0], skip_special_tokens=True)

    if args.sample:
        print(sample(args.sample, n=300)); raise SystemExit

    # ---------------- optimiser ----------------
    # Weight decay on matrices only, not on biases / LayerNorm gains - the GPT-2 / nanoGPT convention.
    decay    = [p for p in raw_model.parameters() if p.dim() >= 2]
    no_decay = [p for p in raw_model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=(device == "cuda"))
    sched = get_cosine_schedule_with_warmup(opt, args.warmup, args.steps)

    start_step = 0
    state_path = f"{args.out}/train_state.pt"
    if args.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=device)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); start_step = st["step"]
        print(f"resumed at step {start_step}")

    def save(step):
        raw_model.save_pretrained(args.out); tok.save_pretrained(args.out)
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, state_path)

    @torch.no_grad()
    def evaluate(arr, iters=20):
        raw_model.eval()
        losses = []
        for _ in range(iters):
            x = get_batch(arr)
            with autocast:
                losses.append(model(x, labels=x).loss.item())   # HF shifts labels internally
        raw_model.train()
        return float(np.mean(losses))

    # ---------------- train ----------------
    data = build_data()
    writer = SummaryWriter(f"runs/{time.strftime('%Y%m%d-%H%M%S')}")
    model.train()
    pbar = tqdm(range(start_step, args.steps), initial=start_step, total=args.steps)
    for step in pbar:
        for _ in range(args.grad_accum):
            x = get_batch(data["train"])
            with autocast:
                loss = model(x, labels=x).loss / args.grad_accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)

        train_loss = loss.item() * args.grad_accum
        writer.add_scalar("train/loss", train_loss, step)
        writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
        pbar.set_postfix(loss=f"{train_loss:.3f}")

        if step > 0 and step % args.eval_every == 0:
            val = evaluate(data["validation"])
            writer.add_scalar("val/loss", val, step)
            text = sample("Once upon a time")
            writer.add_text("sample", text, step)
            tqdm.write(f"step {step} | val {val:.3f} | {text[:120]!r}")
            save(step)

    save(args.steps)
    print(sample("Once upon a time", n=300))
