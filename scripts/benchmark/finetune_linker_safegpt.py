"""SAFE-GPT-style prefix fine-tune for linker_design.

Format: BOS  pieceA_STAMP  [SEP]  pieceB_STAMP  [SEP]  linker_STAMP  EOS
        (loss only on linker_STAMP positions)

Uses HybridVocab.encode_stream(training=True) for random char fallback (matches
AO-GPT pretrain's data pipeline).

Init from existing AO-GPT pretrain ckpt (ep7).
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F
from rdkit import RDLogger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from stamp.benchmark.common import iterable_with_progress
from stamp.benchmark.lm import (
    LMConfig, TinyDecoderLM,
    _resolve_amp_mode, _resolve_torch_device, _set_seed,
)
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


SEP_TOKEN = "[B_8]"  # rare structural token, repurposed as fragment separator


# --- Dataset ---
class LinkerTriplesDataset(IterableDataset):
    """Streams triples from parquet. Each sample yields (token_ids, linker_mask)
    where linker_mask is 1 at linker positions (where loss applies)."""
    def __init__(self, parquet_path, *, vocab, max_seq_len, max_rows=None,
                  fallback_p_low=0.02, fallback_p_high=0.15, training_fallback=True,
                  seed=42, rank=0, world_size=1, sep_token=SEP_TOKEN):
        super().__init__()
        self.parquet_path = str(parquet_path)
        self.vocab = vocab
        self.max_seq_len = int(max_seq_len)
        self.max_rows = max_rows
        self.fallback_p_low = float(fallback_p_low)
        self.fallback_p_high = float(fallback_p_high)
        self.training_fallback = bool(training_fallback)
        self.seed = int(seed)
        self.epoch = 0
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.sep_token = sep_token
        self.sep_id = self.vocab.stoi.get(sep_token, self.vocab.unk_id)

    def set_epoch(self, ep): self.epoch = int(ep)

    def _encode_tokens(self, toks_str, fallback_rng):
        """Apply random char fallback per token via HybridVocab.encode_stream."""
        toks = toks_str.split()
        return self.vocab.encode_stream(toks, training=self.training_fallback, rng=fallback_rng,
                                         p_low=self.fallback_p_low, p_high=self.fallback_p_high)

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

        yielded = 0
        for rg in rg_indices:
            tab = pf.read_row_group(rg, columns=["linker_triples"])
            triples_col = tab.column(0).to_pylist()
            order = list(range(len(triples_col)))
            shuffle_rng.shuffle(order)
            for idx in order:
                tr_str = triples_col[idx]
                if not tr_str or tr_str == "[]": continue
                try:
                    triples = json.loads(tr_str)
                except: continue
                if not triples: continue
                # Pick one triple per drug per iter (random)
                tr = triples[shuffle_rng.randint(0, len(triples) - 1)]
                pa_ids = self._encode_tokens(tr["pieceA"], fallback_rng)
                pb_ids = self._encode_tokens(tr["pieceB"], fallback_rng)
                lk_ids = self._encode_tokens(tr["linker"], fallback_rng)
                # Build: BOS pa SEP pb SEP linker EOS
                ids = [self.vocab.bos_id] + pa_ids + [self.sep_id] + pb_ids + [self.sep_id] + lk_ids + [self.vocab.eos_id]
                # Linker mask: 1 only at linker token positions (and the trailing EOS)
                mask = [0] * (1 + len(pa_ids) + 1 + len(pb_ids) + 1) + [1] * len(lk_ids) + [1]
                if len(ids) > self.max_seq_len:
                    # Truncate from the linker side (keep pieces intact, drop overflow linker tail)
                    overflow = len(ids) - self.max_seq_len
                    if overflow >= len(lk_ids) + 1:
                        continue  # too much overflow, skip
                    ids = ids[:self.max_seq_len]
                    mask = mask[:self.max_seq_len]
                yield ids, mask
                yielded += 1
                if worker_limit is not None and yielded >= worker_limit: return


def _collate(batch, pad_id, fixed_len):
    bsz = len(batch)
    x = torch.full((bsz, fixed_len), pad_id, dtype=torch.long)
    y = torch.full((bsz, fixed_len), -100, dtype=torch.long)
    m = torch.zeros((bsz, fixed_len), dtype=torch.float)
    for i, (ids, mask) in enumerate(batch):
        L = min(len(ids), fixed_len)
        x[i, :L] = torch.tensor(ids[:L], dtype=torch.long)
        # Standard AR: target = next token. So y[i, j] = x[i, j+1]
        # We mask only at positions where loss applies (linker tokens)
        y[i, :L-1] = torch.tensor(ids[1:L], dtype=torch.long)
        # Loss mask: at position j, predict x[j+1]. Apply loss if x[j+1] is linker (mask[j+1] == 1)
        m[i, :L-1] = torch.tensor(mask[1:L], dtype=torch.float)
    return x, y, m


def compute_ar_loss_masked(model, x, y, m):
    """Standard AR loss but only at masked positions (linker)."""
    bsz, seqlen = x.shape
    device = x.device
    # Causal AR via standard left-to-right order
    orders = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
    logits = model(x, orders=orders)  # uses identity order = pure AR
    # Cross-entropy at masked positions
    valid = (y != -100).float()
    weight = m * valid
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                          ignore_index=-100, reduction="none").reshape(bsz, seqlen)
    den = weight.sum().clamp(min=1.0)
    return (ce * weight).sum() / den


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--hybrid-vocab", required=True)
    p.add_argument("--init-ckpt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-data-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--amp", default="auto", choices=["auto","bf16","fp16","none"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch-compile", action="store_true", default=False)
    p.add_argument("--max-rows", type=int, default=0)
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
    if _is_main(): print(f"[safegpt-ft] device={dev} world={_world_size()}", flush=True)

    vocab = HybridVocab.load(args.hybrid_vocab)
    sep_id = vocab.stoi.get(SEP_TOKEN, vocab.unk_id)
    if _is_main():
        print(f"[safegpt-ft] vocab={len(vocab.itos)}  SEP={SEP_TOKEN} id={sep_id}  fallback={args.training_fallback}", flush=True)

    train_ds = LinkerTriplesDataset(
        args.parquet, vocab=vocab, max_seq_len=args.max_seq_len,
        max_rows=args.max_rows if args.max_rows > 0 else None,
        fallback_p_low=args.fallback_p_low, fallback_p_high=args.fallback_p_high,
        training_fallback=args.training_fallback, seed=args.seed,
        rank=int(os.environ.get("RANK", "0")), world_size=_world_size(),
    )

    coll = partial(_collate, pad_id=vocab.pad_id, fixed_len=args.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_data_workers,
                                collate_fn=coll, pin_memory=(dev.type == "cuda"),
                                prefetch_factor=args.prefetch_factor if args.num_data_workers > 0 else None,
                                persistent_workers=(args.num_data_workers > 0))

    ckpt = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
    cfg = LMConfig(**ckpt["cfg"]); cfg.use_adaln = True
    model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg, bidirectional=False).to(dev)
    state = ckpt["model_state"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.","",1):v for k,v in state.items()}
    model.load_state_dict(state)
    if _is_main(): print(f"[safegpt-ft] loaded ckpt {args.init_ckpt}", flush=True)

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
        for step, (x, y, m) in enumerate(iterable_with_progress(train_loader, total=None,
                                                                  desc=f"ep{ep}", show_progress=_is_main())):
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True); m = m.to(dev, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with amp_ctx:
                loss = compute_ar_loss_masked(_unwrap(model), x, y, m)
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
                print(f"[safegpt-ft ep{ep} step {step}] loss={loss.item():.4f}  avg={total_loss/nstep:.4f}  ({time.time()-t_start:.0f}s)", flush=True)

        if _is_main():
            ck = out_dir / f"checkpoints/epoch_{ep:04d}"; ck.mkdir(parents=True, exist_ok=True)
            sd = _unwrap(model).state_dict()
            torch.save({"cfg": cfg.__dict__, "model_state": sd, "epoch": ep, "sep_token": SEP_TOKEN},
                        ck / "model.pt")
            print(f"[safegpt-ft] saved {ck/'model.pt'}  loss={total_loss/max(1,nstep):.4f}", flush=True)

    if _is_dist(): _cleanup_dist()
    if _is_main(): print(f"[safegpt-ft] done. {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    raise SystemExit(main())
