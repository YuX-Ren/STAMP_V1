"""Fine-tune AO-GPT with linker-biased ordering on (decontaminated) STAMP pretrain
parquet. Forks train_ao_gpt_parquet.py's pipeline.

Key difference from pretrain:
  - Order sampling: with probability `linker_prob`, pick 1-3 contiguous motifs
    in the STAMP token sequence as the "linker region", place those positions
    LAST in the AO permutation (model predicts them given full piece context),
    place all other valid positions FIRST (shuffled).
  - Loss: cross-entropy reweighted by 5x at linker positions.

This biases the model toward "given pieceA + pieceB, generate the linker"
without changing anything else (no new architecture, no new tokens).

Usage:
  python scripts/benchmark/finetune_linker_biased.py \\
      --parquet /scratch/yuxuan.ren/data/dataset_stamp_decontam.parquet \\
      --stamp-vocab /scratch/.../motif_vocab.txt \\
      --init-ckpt /scratch/.../stamp_hybrid_30m_d512_l8_e10_cut5k/checkpoints/epoch_0007/model.pt \\
      --output-dir /scratch/.../linker_ft_e1 \\
      --epochs 1 --batch-size 128 --lr 5e-5 --linker-prob 0.7
"""
from __future__ import annotations
import argparse, math, os, random, sys, time, json, shutil
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F
from rdkit import RDLogger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from stamp.benchmark.common import iterable_with_progress
from stamp.benchmark.lm import (
    LMConfig, NextTokenDataset, TinyDecoderLM, TokenVocab,
    _resolve_amp_mode, _resolve_torch_device, _set_seed,
    build_dynamic_lm_collate_fn,
)
from stamp.benchmark.hybrid_vocab import HybridVocab, is_stamp_structural
from stamp.vocab.branch_paths import resolve_branch_output_dir


# --- distributed helpers (copy from train_ao_gpt_parquet) ---
def _dist_rank(): return int(os.environ.get("RANK", "0"))
def _dist_world_size(): return int(os.environ.get("WORLD_SIZE", "1"))
def _dist_local_rank(): return int(os.environ.get("LOCAL_RANK", "0"))
def _is_dist(): return _dist_world_size() > 1
def _is_main(): return _dist_rank() == 0
def _setup_dist(dev):
    if not _is_dist() or dist.is_initialized(): return
    dist.init_process_group("nccl" if dev.type == "cuda" else "gloo")
def _cleanup_dist():
    if dist.is_initialized(): dist.destroy_process_group()
def _unwrap(model):
    return model.module if hasattr(model, "module") else model


# --- linker-biased ordering (fully vectorized) ---
def _sample_ao_orders_linker_biased(
    clean_ids: torch.Tensor,
    labels: torch.Tensor,
    structural_id_set: set,
    structural_id_tensor: torch.Tensor,
    linker_prob: float = 0.7,
    min_motifs: int = 1,
    max_motifs: int = 3,
):
    """Vectorized. Returns (orders, position_weights).

    Per row, with prob `linker_prob` AND if n_motifs >= 3, mark a contiguous
    span of k ∈ [min_motifs, min(max_motifs, n_motifs-2)] motifs as the linker
    region (avoid the first and last motifs). Order key layout:
      BOS:        -1.0
      non-linker: rand in [0, 1)
      linker:     2 + rand in [2, 3)   (sorts AFTER non-linker)
      EOS:        4.0
      invalid:    5 + i/seqlen          (stable tail)
    Argsort gives the AO permutation. Linker positions get weight 5.0.
    """
    bsz, seqlen = labels.shape
    device = clean_ids.device
    valid_mask = labels != -100
    is_struct = torch.isin(clean_ids, structural_id_tensor)
    is_motif = (~is_struct) & valid_mask  # (bsz, seqlen)

    # cumsum gives the # motifs at-or-before each position (1-indexed for motifs;
    # 0 for positions before any motif). motif_idx = cumsum - 1; for non-motif
    # tokens this points to the most recent motif's index (or -1 if before any).
    motif_cumsum = is_motif.long().cumsum(dim=-1)            # (bsz, seqlen)
    motif_idx = motif_cumsum - 1                              # in [-1, n_motifs-1]
    n_motifs = is_motif.sum(dim=-1)                            # (bsz,)

    # Per-row: should we apply linker bias?
    apply_linker = (torch.rand(bsz, device=device) < linker_prob) & (n_motifs >= 3)

    # k = number of linker motifs in [min_motifs, min(max_motifs, n_motifs - 2)]
    kmax = torch.clamp(n_motifs - 2, min=min_motifs, max=max_motifs)
    k_rng = torch.rand(bsz, device=device)
    k = (min_motifs + (k_rng * (kmax - min_motifs + 1).float()).long()).clamp(min=min_motifs, max=max_motifs)

    # start motif index in [0, n_motifs - k]; pick uniform integer
    max_start = torch.clamp(n_motifs - k, min=0)
    s_rng = torch.rand(bsz, device=device)
    start = (s_rng * (max_start + 1).float()).long().clamp(max=max_start)

    # linker_pos_mask: True where motif_idx is in [start, start+k) and the row applies
    s_b = start.unsqueeze(-1)
    k_b = k.unsqueeze(-1)
    linker_pos_mask = (motif_idx >= s_b) & (motif_idx < s_b + k_b) & valid_mask
    linker_pos_mask = linker_pos_mask & apply_linker.unsqueeze(-1)

    # Build keys
    rand_keys = torch.rand(bsz, seqlen, device=device)
    invalid_tail = 5.0 + torch.arange(seqlen, device=device, dtype=rand_keys.dtype) / float(max(1, seqlen))
    keys = torch.where(valid_mask, rand_keys, invalid_tail.unsqueeze(0).expand(bsz, -1))
    # Linker positions get +2 so they sort after non-linker middle
    keys = torch.where(linker_pos_mask, rand_keys + 2.0, keys)

    # Pin first valid → -1.0, last valid → 4.0
    first_idx = valid_mask.to(torch.int32).argmax(dim=-1)
    last_idx = (seqlen - 1) - valid_mask.flip(-1).to(torch.int32).argmax(dim=-1)
    rows = torch.arange(bsz, device=device)
    keys[rows, first_idx] = -1.0
    keys[rows, last_idx] = 4.0
    # If first/last are inside linker_pos_mask, we still want them as anchors,
    # so the override above takes precedence (already done after the +2 step).

    orders = keys.argsort(dim=-1).to(torch.long)

    weights = torch.where(linker_pos_mask, torch.full_like(rand_keys, 5.0), torch.ones_like(rand_keys))
    return orders, weights


def _compute_ao_gpt_loss_linker(
    model, clean_ids, labels, cfg,
    structural_id_set, structural_id_tensor,
    linker_prob=0.7, linker_loss_weight=5.0,
    min_motifs=1, max_motifs=3,
):
    bsz, seqlen = clean_ids.shape
    device = clean_ids.device
    orders, weights = _sample_ao_orders_linker_biased(
        clean_ids, labels, structural_id_set, structural_id_tensor,
        linker_prob=linker_prob, min_motifs=min_motifs, max_motifs=max_motifs,
    )
    # Apply user-specified linker weight (overrides default 5.0 in helper)
    weights = torch.where(weights > 1.0, torch.full_like(weights, linker_loss_weight), weights)
    logits = model(clean_ids, orders=orders)
    batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, seqlen)
    targets = labels[batch_idx, orders]
    weights_at_orders = weights[batch_idx, orders]
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100, reduction="none",
    ).reshape(bsz, seqlen)
    valid_target = (targets != -100).float()
    num = (ce * weights_at_orders * valid_target).sum()
    den = (weights_at_orders * valid_target).sum().clamp(min=1.0)
    return num / den


# --- copy minimal scan / dataset / training-loop scaffolding from train_ao_gpt_parquet ---
def _scan_parquet(parquet_path, vocab, text_col, split_col, train_split, valid_split,
                    train_max, valid_max, require_all_pass, max_seq_len):
    train_seqs, valid_seqs = [], []
    pf = pq.ParquetFile(str(parquet_path))
    cols = [text_col, split_col]
    if require_all_pass: cols.append("all_pass")
    for batch in pf.iter_batches(batch_size=200_000, columns=cols):
        df = batch.to_pandas()
        if require_all_pass and "all_pass" in df.columns:
            df = df[df["all_pass"] == True]
        for _, row in df.iterrows():
            text = row[text_col]
            if text is None: continue
            split = row.get(split_col, train_split)
            tokens = text.split()
            ids = [vocab.bos_id]
            for t in tokens:
                if t in vocab.stoi: ids.append(vocab.stoi[t])
                elif t in vocab.atomic_motifs: ids.append(vocab.atomic_motifs[t])
                else: ids.extend(vocab._expand_motif_to_chars(t))
            ids.append(vocab.eos_id)
            if len(ids) > max_seq_len: continue
            if split == train_split:
                if train_max <= 0 or len(train_seqs) < train_max:
                    train_seqs.append(ids)
            elif split == valid_split:
                if len(valid_seqs) < valid_max:
                    valid_seqs.append(ids)
        if train_max > 0 and len(train_seqs) >= train_max and len(valid_seqs) >= valid_max:
            break
    return train_seqs, valid_seqs


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--stamp-vocab", required=True)
    p.add_argument("--hybrid-vocab", required=True, help="Path to hybrid_vocab.json")
    p.add_argument("--init-ckpt", required=True, help="Pretrain checkpoint to fine-tune from")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--text-column", default="stamp_smiles")
    p.add_argument("--split-column", default="split_random")
    p.add_argument("--train-split", default="train")
    p.add_argument("--valid-split", default="val")
    p.add_argument("--train-max-rows", type=int, default=0)
    p.add_argument("--valid-max-rows", type=int, default=20000)
    p.add_argument("--require-all-pass", action="store_true", default=True)
    p.add_argument("--max-seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-data-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--amp", default="auto", choices=["auto","bf16","fp16","none"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch-compile", action="store_true", default=False)
    # Linker-bias specific
    p.add_argument("--linker-prob", type=float, default=0.7,
                    help="Per-batch row probability to apply linker-biased order")
    p.add_argument("--linker-loss-weight", type=float, default=5.0,
                    help="Cross-entropy weight at linker positions vs 1.0 elsewhere")
    p.add_argument("--linker-min-motifs", type=int, default=1)
    p.add_argument("--linker-max-motifs", type=int, default=3)
    return p


def main():
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")
    _set_seed(args.seed)

    dev = _resolve_torch_device(args.device)
    _setup_dist(dev)
    if dev.type == "cuda" and _is_dist():
        torch.cuda.set_device(_dist_local_rank())
        dev = torch.device(f"cuda:{_dist_local_rank()}")

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if _is_main(): print(f"[ft] device={dev} world={_dist_world_size()}", flush=True)

    # Load hybrid vocab (provides char-expansion for OOV motifs)
    vocab = HybridVocab.load(args.hybrid_vocab)
    structural_ids = {vocab.stoi[t] for t in vocab.itos if is_stamp_structural(t)}
    structural_id_tensor = torch.tensor(sorted(structural_ids), dtype=torch.long, device=dev)
    if _is_main(): print(f"[ft] vocab={len(vocab.itos)}  n_structural={len(structural_ids)}", flush=True)

    # Scan parquet
    if _is_main(): print(f"[ft] scanning parquet {args.parquet}...", flush=True)
    train_seqs, valid_seqs = _scan_parquet(
        args.parquet, vocab,
        args.text_column, args.split_column, args.train_split, args.valid_split,
        args.train_max_rows, args.valid_max_rows, args.require_all_pass,
        args.max_seq_len,
    )
    if _is_main():
        print(f"[ft] train={len(train_seqs):,}  valid={len(valid_seqs):,}", flush=True)

    train_ds = NextTokenDataset(train_seqs, args.max_seq_len, vocab.pad_id)
    valid_ds = NextTokenDataset(valid_seqs, args.max_seq_len, vocab.pad_id)
    collate = build_dynamic_lm_collate_fn(pad_id=vocab.pad_id)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_data_workers, collate_fn=collate,
        pin_memory=(dev.type == "cuda"),
        prefetch_factor=args.prefetch_factor if args.num_data_workers > 0 else None,
        drop_last=True, persistent_workers=(args.num_data_workers > 0),
    )

    # Build model + load checkpoint
    ckpt = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
    cfg = LMConfig(**ckpt["cfg"]); cfg.use_adaln = True
    model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg, bidirectional=False).to(dev)
    state = ckpt["model_state"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.","",1):v for k,v in state.items()}
    model.load_state_dict(state)
    if _is_main():
        print(f"[ft] loaded ckpt {args.init_ckpt}", flush=True)

    if _is_dist():
        model = DDP(model, device_ids=[dev.index] if dev.type == "cuda" else None)
    if args.torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_mode = _resolve_amp_mode(args.amp, dev)
    if amp_mode in ("bf16", "fp16"):
        scaler = torch.amp.GradScaler(enabled=(amp_mode=="fp16"))
        amp_dtype = torch.bfloat16 if amp_mode=="bf16" else torch.float16
        amp_ctx = torch.amp.autocast(device_type="cuda" if dev.type=="cuda" else "cpu", dtype=amp_dtype)
    else:
        scaler = None
        amp_ctx = nullcontext()

    log_every = 50
    t_start = time.time()
    for ep in range(args.epochs):
        model.train()
        total_loss = 0.0; total_tok = 0; nstep = 0
        for step, (x, y) in enumerate(iterable_with_progress(train_loader, total=len(train_loader),
                                                                desc=f"ep{ep}", show_progress=_is_main())):
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with amp_ctx:
                loss = _compute_ao_gpt_loss_linker(
                    model, x, y, cfg,
                    structural_id_set=structural_ids,
                    structural_id_tensor=structural_id_tensor,
                    linker_prob=args.linker_prob,
                    linker_loss_weight=args.linker_loss_weight,
                    min_motifs=args.linker_min_motifs,
                    max_motifs=args.linker_max_motifs,
                )
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            total_loss += float(loss.item()); nstep += 1
            if _is_main() and (step % log_every == 0):
                elapsed = time.time() - t_start
                print(f"[ft ep{ep} step {step}/{len(train_loader)}] loss={loss.item():.4f}  avg={total_loss/nstep:.4f}  ({elapsed:.0f}s)", flush=True)

        # Save checkpoint each epoch
        if _is_main():
            ckpt_dir = out_dir / f"checkpoints/epoch_{ep:04d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            sd = _unwrap(model).state_dict()
            torch.save({"cfg": cfg.__dict__, "model_state": sd, "epoch": ep}, ckpt_dir / "model.pt")
            print(f"[ft] saved {ckpt_dir / 'model.pt'}  loss={total_loss/max(1,nstep):.4f}", flush=True)

    if _is_dist(): _cleanup_dist()
    if _is_main(): print(f"[ft] done. {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    raise SystemExit(main())
