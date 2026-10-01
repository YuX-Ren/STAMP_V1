#!/usr/bin/env python3
"""Finetune the MolT5 c4+zinc pretrain checkpoint on ChEBI-20.

Same script drives both directions via --source-column/--target-column:
  - mol2text: source=mol2text_source (STAMP [BOS]...[EOS]), target=mol2text_target (English)
  - text2mol: source=text2mol_source (English),            target=text2mol_target (STAMP)

Standard T5 seq2seq finetune, single-GPU (small dataset). Loss = CE on target ids
with -100 padding. Saves best-val checkpoint by val_loss + a final checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ChebiSeq2SeqDataset(Dataset):
    """Load pre-computed source/target text columns from a parquet."""

    def __init__(self, parquet_path: str, *, source_col: str, target_col: str) -> None:
        tbl = pq.read_table(parquet_path, columns=[source_col, target_col])
        self.sources = tbl.column(source_col).to_pylist()
        self.targets = tbl.column(target_col).to_pylist()

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.sources[idx], self.targets[idx]


def _collate(batch, *, tokenizer, max_source_len: int, max_target_len: int):
    srcs, tgts = zip(*batch)
    src = tokenizer(
        list(srcs),
        max_length=max_source_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with tokenizer.as_target_tokenizer() if hasattr(tokenizer, "as_target_tokenizer") else _null_ctx():
        tgt = tokenizer(
            list(tgts),
            max_length=max_target_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
    labels = tgt["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        "input_ids": src["input_ids"],
        "attention_mask": src["attention_mask"],
        "labels": labels,
    }


class _null_ctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def _autocast(device: torch.device, amp: str):
    if amp == "bf16" and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    if amp == "fp16" and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return _null_ctx()


def _linear_warmup_decay(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 1.0 - progress)


def _cosine_warmup_decay(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    # cosine from 1.0 to 0.0 over the post-warmup phase
    import math as _m
    return 0.5 * (1.0 + _m.cos(_m.pi * progress))


@torch.inference_mode()
def _evaluate(model, loader, tokenizer, device, amp: str) -> dict:
    model.eval()
    tot_loss, tot_toks = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with _autocast(device, amp):
            out = model(**batch)
        n_tok = int((batch["labels"] != -100).sum().item())
        tot_loss += float(out.loss.detach().item()) * n_tok
        tot_toks += n_tok
    model.train()
    mean = tot_loss / max(1, tot_toks)
    return {"valid_loss": mean, "valid_ppl": math.exp(min(20.0, mean)), "valid_tokens": tot_toks}


def _save(model, tokenizer, out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    raw.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "finetune_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-ckpt", required=True, help="Path to the pretrain checkpoint dir.")
    p.add_argument("--train-parquet", required=True)
    p.add_argument("--valid-parquet", required=True)
    p.add_argument("--source-column", required=True, choices=["mol2text_source", "text2mol_source"])
    p.add_argument("--target-column", required=True, choices=["mol2text_target", "text2mol_target"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-source-len", type=int, default=256)
    p.add_argument("--max-target-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every-steps", type=int, default=500)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--early-stop-patience", type=int, default=6,
                   help="Number of consecutive eval events without best-val improvement before stopping.")
    p.add_argument("--dropout", type=float, default=None,
                   help="Override model dropout (both attention + FFN). MolT5 official finetune uses 0.0; "
                        "T5.1.1-base default is 0.1. None keeps whatever is in the init checkpoint.")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="Label smoothing coefficient for CE loss. 0.0 = none.")
    p.add_argument("--lr-schedule", choices=["linear", "cosine"], default="linear",
                   help="Post-warmup decay shape.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from transformers import AutoTokenizer, T5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(args.init_ckpt, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(args.init_ckpt, local_files_only=True)
    if args.dropout is not None:
        model.config.dropout_rate = float(args.dropout)
        for mod in model.modules():
            if hasattr(mod, "p") and isinstance(getattr(mod, "p"), float):
                mod.p = float(args.dropout)
        print(f"[finetune] overrode model dropout to {args.dropout}", flush=True)
    model.to(device).train()

    train_ds = ChebiSeq2SeqDataset(args.train_parquet, source_col=args.source_column, target_col=args.target_column)
    valid_ds = ChebiSeq2SeqDataset(args.valid_parquet, source_col=args.source_column, target_col=args.target_column)

    def _mk_collate():
        return lambda batch: _collate(
            batch,
            tokenizer=tokenizer,
            max_source_len=args.max_source_len,
            max_target_len=args.max_target_len,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_mk_collate(),
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_mk_collate(),
        drop_last=False,
    )

    steps_per_epoch = max(1, len(train_loader) // max(1, args.grad_accum))
    total_optim_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_optim_steps * args.warmup_ratio)

    decay_params, nodecay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith(".bias") or "layer_norm" in n.lower() or "norm.weight" in n:
            nodecay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    _schedule_fn = _cosine_warmup_decay if args.lr_schedule == "cosine" else _linear_warmup_decay
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _schedule_fn(step, warmup_steps, total_optim_steps),
    )

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "finetune_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    log_path = out_dir / "train.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    log(f"[finetune] init_ckpt={args.init_ckpt}")
    log(f"[finetune] train_rows={len(train_ds)} valid_rows={len(valid_ds)} steps_per_epoch={steps_per_epoch} total_optim_steps={total_optim_steps} warmup_steps={warmup_steps}")
    log(f"[finetune] direction: source={args.source_column} target={args.target_column}")
    log(f"[finetune] tokenizer vocab_size={len(tokenizer)}  pad_id={tokenizer.pad_token_id}  eos_id={tokenizer.eos_token_id}")

    optim_step = 0
    micro_step = 0
    best_val = float("inf")
    patience = 0
    history = []
    t0 = time.time()
    running_loss = 0.0
    running_toks = 0
    for epoch in range(args.epochs):
        for batch in train_loader:
            micro_step += 1
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with _autocast(device, args.amp):
                if args.label_smoothing > 0:
                    # Custom loss so we can apply label smoothing.
                    out = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    import torch.nn.functional as _F
                    logits = out.logits
                    step_loss = _F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        batch["labels"].view(-1),
                        ignore_index=-100,
                        label_smoothing=float(args.label_smoothing),
                    )
                else:
                    out = model(**batch)
                    step_loss = out.loss
            loss = step_loss / args.grad_accum
            loss.backward()
            running_loss += float(step_loss.detach().item())
            running_toks += int((batch["labels"] != -100).sum().item())

            if micro_step % args.grad_accum == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optim_step += 1

                if optim_step % args.log_every_steps == 0:
                    elapsed = max(1e-6, time.time() - t0)
                    log(
                        f"[finetune] step={optim_step}/{total_optim_steps} "
                        f"epoch={epoch} loss={running_loss/max(1, args.log_every_steps*args.grad_accum):.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.3e} toks/s={running_toks/elapsed:.0f}"
                    )
                    running_loss = 0.0
                    running_toks = 0
                    t0 = time.time()

                if optim_step > 0 and optim_step % args.eval_every_steps == 0:
                    metrics = _evaluate(model, valid_loader, tokenizer, device, args.amp)
                    metrics.update({"step": optim_step, "epoch": epoch})
                    history.append(metrics)
                    (out_dir / "eval_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
                    log(f"[finetune] eval step={optim_step} valid_loss={metrics['valid_loss']:.4f} valid_ppl={metrics['valid_ppl']:.3f}")
                    if metrics["valid_loss"] < best_val - 1e-4:
                        best_val = metrics["valid_loss"]
                        patience = 0
                        _save(model, tokenizer, out_dir / "best", {"step": optim_step, "epoch": epoch, "valid_loss": best_val, **vars(args)})
                        log(f"[finetune] saved best_val={best_val:.4f} to {out_dir / 'best'}")
                    else:
                        patience += 1
                        log(f"[finetune] no improvement, patience={patience}/{args.early_stop_patience}")
                        if patience >= args.early_stop_patience:
                            log(f"[finetune] early-stop after {args.early_stop_patience} unimproved evals")
                            _save(model, tokenizer, out_dir / "final", {"step": optim_step, "epoch": epoch, "valid_loss": metrics["valid_loss"], **vars(args)})
                            return 0

    _save(model, tokenizer, out_dir / "final", {"step": optim_step, "epoch": args.epochs - 1, "valid_loss": best_val, **vars(args)})
    log(f"[finetune] finished epochs={args.epochs}; best_val={best_val:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
