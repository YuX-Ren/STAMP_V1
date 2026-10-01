"""Fine-tune AO-GPT on linker subtree mask (BRICS-derived linker positions).

Data: parquet with `stamp_smiles` + `linker_subtrees` (JSON list of motif idx lists)
Format: At each step, randomly pick one subtree from the list, mask those motifs'
full token blocks (motif token + joints), pin everything else, AO-GPT loss
weighted on mask positions.

Char-fallback: applied per pretrain (random char-expand for atomic motifs based
on freq).
"""
from __future__ import annotations
import argparse, math, os, random, sys, time, json, shutil
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F
from rdkit import RDLogger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from stamp.benchmark.common import iterable_with_progress
from stamp.benchmark.lm import LMConfig, TinyDecoderLM, _resolve_amp_mode, _resolve_torch_device, _set_seed
from stamp.benchmark.hybrid_vocab import HybridVocab, is_stamp_structural


def _is_main(): return int(os.environ.get("RANK", "0")) == 0
def _world_size(): return int(os.environ.get("WORLD_SIZE", "1"))
def _is_dist(): return _world_size() > 1
def _setup_dist(dev):
    if not _is_dist() or dist.is_initialized(): return
    dist.init_process_group("nccl" if dev.type == "cuda" else "gloo")
def _cleanup_dist():
    if dist.is_initialized(): dist.destroy_process_group()
def _unwrap(model): return model.module if hasattr(model, "module") else model


# --- Dataset ---
class LinkerSubtreeDataset(Dataset):
    """Each sample: (stamp_text, list of valid subtree motif index lists).
    At __getitem__, pick a random subtree and return (token_ids, linker_mask)."""
    def __init__(self, samples: List[Tuple[str, List[List[int]]]], vocab: HybridVocab,
                  max_seq_len: int, training: bool = True,
                  fallback_p_low: float = 0.02, fallback_p_high: float = 0.15):
        self.samples = samples
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.training = training
        self.p_low = fallback_p_low
        self.p_high = fallback_p_high

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        stamp_text, subtrees_list = self.samples[idx]
        # Pick random subtree
        subtree = subtrees_list[random.randrange(len(subtrees_list))]
        subtree_set = set(subtree)

        # Walk tokens, build (token_ids, mask) with char-fallback applied per pretrain
        tokens = stamp_text.split()
        # First identify motif blocks: each motif token + its trailing joints/structural
        # Collect (block_tokens, motif_idx) pairs interleaved with [B_n] tokens
        ids_out = [self.vocab.bos_id]
        mask_out = [0]  # 0 = pinned, 1 = mask (loss)
        rng = random.Random()
        motif_count = 0
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if not is_stamp_structural(t):
                # Motif token + its trailing joint/stereo/ring tokens
                this_motif = motif_count
                is_mask = (this_motif in subtree_set)
                block = [t]; i += 1
                while i < len(tokens) and (tokens[i].startswith("[J_") or
                                            tokens[i].startswith("[S_") or
                                            tokens[i] == "[Ring]"):
                    block.append(tokens[i]); i += 1
                # Encode block tokens with char-fallback (only motif token can fallback,
                # joints/structural always single id)
                for j, b_tok in enumerate(block):
                    if is_stamp_structural(b_tok):
                        # Joint/stereo/ring — use direct id
                        if b_tok in self.vocab.stoi:
                            ids = [self.vocab.stoi[b_tok]]
                        else:
                            ids = [self.vocab.unk_id]
                    elif b_tok in self.vocab.atomic_motifs:
                        # Atomic motif: maybe char-expand based on freq fallback prob
                        atomic_id = self.vocab.atomic_motifs[b_tok]
                        freq = self.vocab.motif_freq.get(b_tok, 1)
                        p_fallback = self.vocab.fallback_prob(freq, p_low=self.p_low, p_high=self.p_high)
                        if self.training and rng.random() < p_fallback:
                            ids = list(self.vocab._expand_motif_to_chars(b_tok))
                        else:
                            ids = [atomic_id]
                    elif b_tok in self.vocab.stoi:
                        ids = [self.vocab.stoi[b_tok]]
                    else:
                        ids = list(self.vocab._expand_motif_to_chars(b_tok))
                    for x in ids:
                        ids_out.append(x)
                        mask_out.append(1 if is_mask else 0)
                motif_count += 1
            else:
                # [B_n] structural — always pin
                if t in self.vocab.stoi:
                    ids_out.append(self.vocab.stoi[t]); mask_out.append(0)
                i += 1
        ids_out.append(self.vocab.eos_id); mask_out.append(0)

        # Truncate
        if len(ids_out) > self.max_seq_len:
            ids_out = ids_out[:self.max_seq_len]
            mask_out = mask_out[:self.max_seq_len]
        return ids_out, mask_out


def collate_fn(batch, pad_id):
    max_len = max(len(ids) for ids, _ in batch)
    bsz = len(batch)
    x = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    y = torch.full((bsz, max_len), -100, dtype=torch.long)  # labels
    m = torch.zeros((bsz, max_len), dtype=torch.float)  # mask positions for loss weighting
    for i, (ids, mask) in enumerate(batch):
        L = len(ids)
        x[i, :L] = torch.tensor(ids, dtype=torch.long)
        # AO-GPT next-token: predict ids[1:] given ids[:-1]; here we use full label = ids
        y[i, :L] = torch.tensor(ids, dtype=torch.long)
        m[i, :L] = torch.tensor(mask, dtype=torch.float)
    return x, y, m


# --- AO-GPT ordering with subtree-mask priority ---
def _sample_orders_subtree_biased(clean_ids, labels, mask_weights):
    """Place mask positions LAST in AO-GPT permutation (predict them after seeing all others).
    Non-mask valid positions go first (random shuffled). BOS first, EOS last."""
    bsz, seqlen = labels.shape
    device = clean_ids.device
    valid_mask = labels != -100
    # mask_weights[i,j] > 0 means position j is in linker mask region
    rand_keys = torch.rand(bsz, seqlen, device=device)
    invalid_tail = 5.0 + torch.arange(seqlen, device=device, dtype=rand_keys.dtype) / float(seqlen)
    keys = torch.where(valid_mask, rand_keys, invalid_tail.unsqueeze(0).expand(bsz, -1))
    # Linker positions sort AFTER non-linker (offset by +2)
    keys = torch.where(mask_weights > 0, rand_keys + 2.0, keys)
    # Pin first valid → -1, last valid → 4
    first_idx = valid_mask.to(torch.int32).argmax(dim=-1)
    last_idx = (seqlen - 1) - valid_mask.flip(-1).to(torch.int32).argmax(dim=-1)
    rows = torch.arange(bsz, device=device)
    keys[rows, first_idx] = -1.0
    keys[rows, last_idx] = 4.0
    return keys.argsort(dim=-1).to(torch.long)


def compute_loss(model, x, y, m, cfg, mask_weight=5.0):
    bsz, seqlen = x.shape
    device = x.device
    orders = _sample_orders_subtree_biased(x, y, m)
    logits = model(x, orders=orders)
    batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, seqlen)
    targets = y[batch_idx, orders]
    weights = m[batch_idx, orders]  # 1 at mask, 0 elsewhere
    # Apply mask_weight; non-mask gets weight 1
    final_w = torch.where(weights > 0, torch.full_like(weights, mask_weight), torch.ones_like(weights))
    valid = (targets != -100).float()
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                         targets.reshape(-1),
                         ignore_index=-100, reduction="none").reshape(bsz, seqlen)
    num = (ce * final_w * valid).sum()
    den = (final_w * valid).sum().clamp(min=1.0)
    return num / den


def _scan(parquet_path, vocab, max_seq_len, max_rows, train_split, valid_split, valid_max_rows):
    train_samples, valid_samples = [], []
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=200_000,
                                  columns=['stamp_smiles', 'split_random', 'linker_subtrees']):
        df = batch.to_pandas()
        for _, row in df.iterrows():
            stamp = row['stamp_smiles']
            sub_str = row.get('linker_subtrees', '[]')
            if not stamp or sub_str == '[]': continue
            try:
                subtrees = json.loads(sub_str)
            except: continue
            if not subtrees: continue
            split = row.get('split_random', 'train')
            sample = (stamp, subtrees)
            if split == train_split:
                if max_rows <= 0 or len(train_samples) < max_rows:
                    train_samples.append(sample)
            elif split == valid_split:
                if len(valid_samples) < valid_max_rows:
                    valid_samples.append(sample)
        if max_rows > 0 and len(train_samples) >= max_rows and len(valid_samples) >= valid_max_rows: break
    return train_samples, valid_samples


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--stamp-vocab", required=True)
    p.add_argument("--hybrid-vocab", required=True)
    p.add_argument("--init-ckpt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-split", default="train")
    p.add_argument("--valid-split", default="val")
    p.add_argument("--train-max-rows", type=int, default=0)
    p.add_argument("--valid-max-rows", type=int, default=20000)
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
    p.add_argument("--mask-loss-weight", type=float, default=5.0)
    p.add_argument("--fallback-p-low", type=float, default=0.02)
    p.add_argument("--fallback-p-high", type=float, default=0.15)
    return p


def main():
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")
    _set_seed(args.seed)

    dev = _resolve_torch_device(args.device)
    _setup_dist(dev)
    if dev.type == "cuda" and _is_dist():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if _is_main(): print(f"[ft-subtree] device={dev} world={_world_size()}", flush=True)

    vocab = HybridVocab.load(args.hybrid_vocab)
    if _is_main(): print(f"[ft-subtree] vocab={len(vocab.itos)}", flush=True)

    if _is_main(): print(f"[ft-subtree] scanning {args.parquet}...", flush=True)
    train_samples, valid_samples = _scan(args.parquet, vocab, args.max_seq_len,
                                          args.train_max_rows, args.train_split, args.valid_split,
                                          args.valid_max_rows)
    if _is_main(): print(f"[ft-subtree] train={len(train_samples):,}  valid={len(valid_samples):,}", flush=True)

    train_ds = LinkerSubtreeDataset(train_samples, vocab, args.max_seq_len, training=True,
                                      fallback_p_low=args.fallback_p_low,
                                      fallback_p_high=args.fallback_p_high)
    valid_ds = LinkerSubtreeDataset(valid_samples, vocab, args.max_seq_len, training=False)

    from functools import partial
    coll = partial(collate_fn, pad_id=vocab.pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_data_workers, collate_fn=coll,
                                pin_memory=(dev.type == "cuda"),
                                prefetch_factor=args.prefetch_factor if args.num_data_workers > 0 else None,
                                drop_last=True, persistent_workers=(args.num_data_workers > 0))

    ckpt = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
    cfg = LMConfig(**ckpt["cfg"]); cfg.use_adaln = True
    model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg, bidirectional=False).to(dev)
    state = ckpt["model_state"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.","",1):v for k,v in state.items()}
    model.load_state_dict(state)
    if _is_main(): print(f"[ft-subtree] loaded ckpt {args.init_ckpt}", flush=True)

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
        model.train()
        total_loss = 0.0; nstep = 0
        for step, (x, y, m) in enumerate(iterable_with_progress(train_loader, total=len(train_loader),
                                                                 desc=f"ep{ep}", show_progress=_is_main())):
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True); m = m.to(dev, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with amp_ctx:
                loss = compute_loss(model, x, y, m, cfg, mask_weight=args.mask_loss_weight)
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
                print(f"[ft-subtree ep{ep} step {step}/{len(train_loader)}] loss={loss.item():.4f}  avg={total_loss/nstep:.4f}  ({elapsed:.0f}s)", flush=True)

        if _is_main():
            ckpt_dir = out_dir / f"checkpoints/epoch_{ep:04d}"; ckpt_dir.mkdir(parents=True, exist_ok=True)
            sd = _unwrap(model).state_dict()
            torch.save({"cfg": cfg.__dict__, "model_state": sd, "epoch": ep}, ckpt_dir / "model.pt")
            print(f"[ft-subtree] saved {ckpt_dir / 'model.pt'}  loss={total_loss/max(1,nstep):.4f}", flush=True)

    if _is_dist(): _cleanup_dist()
    if _is_main(): print(f"[ft-subtree] done. {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    raise SystemExit(main())
