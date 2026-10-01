"""ESO-LM training with HybridVocab + random char fallback.

Combines:
- ESO sequential branch loss (from train_stamp_eso_v2_parquet.py)
- HybridVocab data pipeline with random char-fallback (from train_stamp_hybrid_parquet.py)

Goal: align ESO training with the hybrid pretrain checkpoint (random motif↔char
expansion at training time), so we can compare apples-to-apples.
"""
from __future__ import annotations
import argparse, json, math, os, random, shutil, sys, time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from stamp.benchmark.common import iterable_with_progress
from stamp.benchmark.lm import (
    LMConfig,
    _resolve_amp_mode, _resolve_torch_device, _set_seed,
)
from stamp.benchmark.eso_v3 import TinyEsoDiTV3, compute_eso_v3_seq_loss
from stamp.benchmark.hybrid_vocab import HybridVocab


def _is_main(): return int(os.environ.get("RANK", "0")) == 0
def _world_size(): return int(os.environ.get("WORLD_SIZE", "1"))
def _is_dist(): return _world_size() > 1
def _setup_dist(dev):
    if not _is_dist() or dist.is_initialized(): return
    dist.init_process_group("nccl" if dev.type == "cuda" else "gloo")
def _cleanup_dist():
    if dist.is_initialized(): dist.destroy_process_group()
def _unwrap(model): return model.module if hasattr(model, "module") else model


# --- Dataset: HybridVocab + random char fallback ---
class _EsoHybridDataset(IterableDataset):
    """Streams parquet rows, encodes with HybridVocab.encode_stream(training=True)
    so atomic motifs are randomly char-expanded per fallback_prob."""
    def __init__(self, parquet_path, *, vocab, text_column, split_column, split_value,
                  require_all_pass, max_seq_len, max_rows, seed,
                  fallback_p_low=0.02, fallback_p_high=0.15, training_fallback=True,
                  rank=0, world_size=1):
        super().__init__()
        self.parquet_path = str(parquet_path)
        self.vocab = vocab
        self.text_column = text_column
        self.split_column = split_column
        self.split_value = split_value
        self.require_all_pass = bool(require_all_pass)
        self.max_seq_len = int(max_seq_len)
        self.max_rows = max_rows
        self.seed = int(seed)
        self.fallback_p_low = float(fallback_p_low)
        self.fallback_p_high = float(fallback_p_high)
        self.training_fallback = bool(training_fallback)
        self.epoch = 0
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))

    def set_epoch(self, ep): self.epoch = int(ep)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        wid = 0 if worker is None else int(worker.id)
        nw = 1 if worker is None else int(worker.num_workers)
        shard_id = self.rank * nw + wid
        nshards = self.world_size * nw

        pf = pq.ParquetFile(self.parquet_path)
        rg_indices = list(range(pf.num_row_groups))
        shuffle_rng = random.Random(self.seed + self.epoch)
        shuffle_rng.shuffle(rg_indices)
        rg_indices = rg_indices[shard_id::nshards]
        fallback_rng = random.Random(self.seed * 131 + self.epoch * 17 + shard_id)

        if self.max_rows is not None:
            base = self.max_rows // nshards
            rem = self.max_rows % nshards
            worker_limit = base + (1 if shard_id < rem else 0)
        else:
            worker_limit = None

        columns = [self.text_column, self.split_column]
        if self.require_all_pass:
            columns.append("all_pass")

        yielded = 0
        for rg in rg_indices:
            tab = pf.read_row_group(rg, columns=columns)
            cols = {name: tab.column(i).to_pylist() for i, name in enumerate(columns)}
            order = list(range(len(cols[self.text_column])))
            shuffle_rng.shuffle(order)
            for idx in order:
                if cols[self.split_column][idx] != self.split_value: continue
                if self.require_all_pass and not bool(cols["all_pass"][idx]): continue
                text = cols[self.text_column][idx]
                if text is None: continue
                toks = str(text).split()
                if not toks: continue
                ids = (
                    [self.vocab.bos_id]
                    + self.vocab.encode_stream(
                        toks,
                        training=self.training_fallback,
                        rng=fallback_rng,
                        p_low=self.fallback_p_low,
                        p_high=self.fallback_p_high,
                    )
                    + [self.vocab.eos_id]
                )
                if len(ids) < 2: continue
                ids = ids[: self.max_seq_len]
                yield ids
                yielded += 1
                if worker_limit is not None and yielded >= worker_limit: return


def _collate(batch, pad_id, fixed_len=None):
    """Pad to fixed_len if given, else to max in batch.
    Fixed length avoids torch.compile recompilation on variable shapes."""
    if fixed_len is None:
        max_len = max(len(ids) for ids in batch)
    else:
        max_len = fixed_len
    bsz = len(batch)
    x = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    y = torch.full((bsz, max_len), -100, dtype=torch.long)
    for i, ids in enumerate(batch):
        L = min(len(ids), max_len)
        x[i, :L] = torch.tensor(ids[:L], dtype=torch.long)
        y[i, :L] = torch.tensor(ids[:L], dtype=torch.long)
    return x, y


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--hybrid-vocab", required=True, help="Path to hybrid_vocab.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--text-column", default="stamp_smiles")
    p.add_argument("--split-column", default="split_random")
    p.add_argument("--train-split", default="train")
    p.add_argument("--valid-split", default="val")
    p.add_argument("--require-all-pass", action="store_true", default=True)
    p.add_argument("--train-max-rows", type=int, default=0)
    p.add_argument("--valid-max-rows", type=int, default=20000)
    p.add_argument("--max-seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-data-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--amp", default="auto", choices=["auto","bf16","fp16","none"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch-compile", action="store_true", default=False)

    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--alpha-0", type=float, default=0.0)
    p.add_argument("--shuffle-unmasked", action="store_true", default=False)

    p.add_argument("--fallback-p-low", type=float, default=0.02)
    p.add_argument("--fallback-p-high", type=float, default=0.15)
    p.add_argument("--no-fallback", dest="training_fallback", action="store_false", default=True)
    return p


def main():
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")
    _set_seed(args.seed)

    dev = _resolve_torch_device(args.device)
    _setup_dist(dev)
    if dev.type == "cuda" and _is_dist():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dev = torch.device(f"cuda:{local_rank}")

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if _is_main(): print(f"[eso-hybrid] device={dev} world={_world_size()}", flush=True)

    vocab = HybridVocab.load(args.hybrid_vocab)
    if _is_main(): print(f"[eso-hybrid] vocab={len(vocab.itos)} mask_id={vocab.mask_id} fallback={args.training_fallback}", flush=True)

    train_ds = _EsoHybridDataset(
        args.parquet, vocab=vocab, text_column=args.text_column,
        split_column=args.split_column, split_value=args.train_split,
        require_all_pass=args.require_all_pass, max_seq_len=args.max_seq_len,
        max_rows=args.train_max_rows if args.train_max_rows > 0 else None,
        seed=args.seed,
        fallback_p_low=args.fallback_p_low, fallback_p_high=args.fallback_p_high,
        training_fallback=args.training_fallback,
        rank=int(os.environ.get("RANK", "0")), world_size=_world_size(),
    )
    valid_ds = _EsoHybridDataset(
        args.parquet, vocab=vocab, text_column=args.text_column,
        split_column=args.split_column, split_value=args.valid_split,
        require_all_pass=args.require_all_pass, max_seq_len=args.max_seq_len,
        max_rows=args.valid_max_rows if args.valid_max_rows > 0 else None,
        seed=args.seed + 1,
        training_fallback=False,  # No fallback at eval
        rank=int(os.environ.get("RANK", "0")), world_size=_world_size(),
    )

    from functools import partial
    # Fixed-length pad: required because flex_attention compiled with dynamic=False
    # would recompile on each new shape and hit cache_size_limit.
    coll = partial(_collate, pad_id=vocab.pad_id, fixed_len=args.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_data_workers,
                                collate_fn=coll, pin_memory=(dev.type == "cuda"),
                                prefetch_factor=args.prefetch_factor if args.num_data_workers > 0 else None,
                                persistent_workers=(args.num_data_workers > 0))
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, num_workers=2, collate_fn=coll,
                                pin_memory=(dev.type == "cuda"))

    cfg = LMConfig(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff,
        dropout=args.dropout, max_seq_len=args.max_seq_len,
    )
    model = TinyEsoDiTV3(vocab_size=len(vocab.itos), cfg=cfg, mask_index=vocab.mask_id).to(dev)
    if _is_main():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[eso-hybrid] model params={n_params/1e6:.1f}M", flush=True)

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
        scaler = None; amp_ctx = nullcontext()

    log_every = 50
    t_start = time.time()
    for ep in range(args.epochs):
        train_ds.set_epoch(ep)
        model.train()
        total_loss = 0.0; nstep = 0
        for step, (x, y) in enumerate(iterable_with_progress(train_loader, total=None,
                                                                desc=f"ep{ep}", show_progress=_is_main())):
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with amp_ctx:
                loss = compute_eso_v3_seq_loss(
                    _unwrap(model), x, y, cfg,
                    mask_id=vocab.mask_id,
                    alpha_0=args.alpha_0,
                    shuffle_unmasked=args.shuffle_unmasked,
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
                print(f"[eso-hybrid ep{ep} step {step}] loss={loss.item():.4f}  avg={total_loss/nstep:.4f}  ({time.time()-t_start:.0f}s)", flush=True)

        if _is_main():
            ck = out_dir / f"checkpoints/epoch_{ep:04d}"; ck.mkdir(parents=True, exist_ok=True)
            sd = _unwrap(model).state_dict()
            torch.save({"cfg": cfg.__dict__, "model_state": sd, "epoch": ep}, ck / "model.pt")
            print(f"[eso-hybrid] saved {ck/'model.pt'}  loss={total_loss/max(1,nstep):.4f}", flush=True)

    if _is_dist(): _cleanup_dist()
    if _is_main(): print(f"[eso-hybrid] done. {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    raise SystemExit(main())
