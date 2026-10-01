#!/usr/bin/env python3
"""MolFormer-style MLM pretraining on sharded STAMP parquet data.

This is the STAMP adaptation of IBM MolFormer:
  - encoder-only MLM objective
  - linear attention with rotary embeddings
  - large-batch LAMB/AdamW training
  - streaming over sharded PubChem/ZINC-style parquet data

The script intentionally avoids the original IBM runtime dependencies
(`pytorch_lightning`, `fast_transformers`, `apex`) because they are not present
in the current STAMP training environment.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.lm import TokenVocab, _resolve_amp_mode, _resolve_torch_device, _set_seed  # noqa: E402
from stamp.benchmark.molformer_stamp import (  # noqa: E402
    Lamb,
    MolFormerEncoder,
    MolFormerMLMHead,
    MolFormerSTAMPConfig,
    build_stamp_molformer_vocab,
    expand_stamp_views,
    ids_from_stamp_tokens,
    save_molformer_checkpoint,
    smiles_char_tokens,
    split_concat_views,
    token_ids_are_structural,
)


def _dist_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _dist_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _dist_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _is_dist() -> bool:
    return _dist_world_size() > 1


def _is_main() -> bool:
    return _dist_rank() == 0


def _setup_dist(dev: torch.device) -> None:
    if not _is_dist() or dist.is_initialized():
        return
    dist.init_process_group(backend="nccl" if dev.type == "cuda" else "gloo")


def _barrier(dev: torch.device) -> None:
    if not dist.is_initialized():
        return
    if dev.type == "cuda" and dev.index is not None:
        dist.barrier(device_ids=[dev.index])
    else:
        dist.barrier()


def _cleanup_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _sorted_paths(pattern: str) -> list[str]:
    paths: list[str] = []
    for part in str(pattern).split(","):
        part = part.strip()
        if part:
            paths.extend(glob.glob(part))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No files matched glob: {pattern}")
    return paths


def _count_parquet_rows(paths: Sequence[str]) -> int:
    total = 0
    for path in paths:
        meta = pq.ParquetFile(path).metadata
        total += int(meta.num_rows) if meta is not None else 0
    return total


def _split_train_valid_paths(
    train_paths: Sequence[str],
    *,
    valid_glob: str,
    valid_file_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    train = list(train_paths)
    if valid_glob.strip():
        valid = _sorted_paths(valid_glob)
        valid_set = set(valid)
        train = [p for p in train if p not in valid_set]
        if not train:
            raise ValueError("Validation glob overlaps all training files.")
        return train, valid
    frac = float(valid_file_fraction)
    if frac <= 0.0:
        return train, []
    if frac >= 1.0:
        raise ValueError("--valid-file-fraction must be < 1.0")
    n_valid = max(1, int(math.ceil(len(train) * frac)))
    if n_valid >= len(train):
        raise ValueError("Validation file holdout leaves no training files.")
    shuffled = list(train)
    random.Random(int(seed)).shuffle(shuffled)
    valid = set(shuffled[:n_valid])
    return [p for p in train if p not in valid], [p for p in train if p in valid]


def _iter_text_rows(paths: Sequence[str], text_column: str) -> Iterator[str]:
    for path in paths:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            tab = pf.read_row_group(rg, columns=[text_column])
            for text in tab.column(0).to_pylist():
                if text is None:
                    continue
                text = str(text).strip()
                if text:
                    yield text


def _parquet_row_groups(paths: Sequence[str]) -> list[tuple[str, int]]:
    row_groups: list[tuple[str, int]] = []
    for path in paths:
        pf = pq.ParquetFile(path)
        row_groups.extend((str(path), rg) for rg in range(pf.num_row_groups))
    return row_groups


@dataclass
class ScanSummary:
    raw_rows: int
    scanned_rows: int
    examples_per_raw_row: float
    kept_examples: int
    dropped_overlength: int
    lengths: dict[str, float]
    oov_token_occurrences: int
    oov_unique_tokens: int


def _summarize_lengths(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {}
    arr = np.asarray(lengths, dtype=np.float64)
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def _select_views(tokens: Sequence[str], view_mode: str, view_index: int) -> list[list[str]]:
    if view_index >= 0:
        views = split_concat_views(tokens)
        if not views:
            views = [list(tokens)]
        if view_index >= len(views):
            raise ValueError(f"view_index={view_index} out of range for {len(views)} views")
        return [views[view_index]]
    return expand_stamp_views(tokens, view_mode)


def _select_text_views(text: str, view_mode: str, view_index: int) -> list[list[str]]:
    raw = str(text).strip()
    if view_index >= 0:
        parts = raw.split(" [CONCAT] ")
        if len(parts) > 1:
            if view_index >= len(parts):
                raise ValueError(f"view_index={view_index} out of range for {len(parts)} views")
            return [parts[view_index].split()]
    return _select_views(raw.split(), view_mode, view_index)


def _scan_paths(
    *,
    paths: Sequence[str],
    text_column: str,
    vocab: TokenVocab,
    view_mode: str,
    view_index: int,
    max_seq_len: int,
    scan_max_rows: int,
    drop_overlength: bool,
    fallback_oov_to_smiles: bool,
) -> ScanSummary:
    lengths: list[int] = []
    oov_counter: dict[str, int] = {}
    raw_seen = 0
    examples = 0
    dropped = 0
    for text in _iter_text_rows(paths, text_column):
        if scan_max_rows > 0 and raw_seen >= scan_max_rows:
            break
        raw_seen += 1
        for view in _select_text_views(str(text), view_mode, view_index):
            ids = ids_from_stamp_tokens(
                view,
                vocab,
                max_seq_len=10**9,
                fallback_oov_to_smiles=fallback_oov_to_smiles,
            )
            if drop_overlength and len(ids) > max_seq_len:
                dropped += 1
                continue
            lengths.append(min(len(ids), max_seq_len))
            examples += 1
            for tok in view:
                if tok not in {"[BOS]", "[EOS]"} and tok not in vocab.stoi:
                    if fallback_oov_to_smiles:
                        try:
                            if all(t in vocab.stoi for t in smiles_char_tokens(tok)):
                                continue
                        except ValueError:
                            pass
                    oov_counter[tok] = oov_counter.get(tok, 0) + 1
    return ScanSummary(
        raw_rows=0,
        scanned_rows=raw_seen,
        examples_per_raw_row=float(examples / max(1, raw_seen)),
        kept_examples=examples,
        dropped_overlength=dropped,
        lengths=_summarize_lengths(lengths),
        oov_token_occurrences=int(sum(oov_counter.values())),
        oov_unique_tokens=int(len(oov_counter)),
    )


class ShardedSTAMPDataset(IterableDataset):
    def __init__(
        self,
        paths: Sequence[str],
        *,
        text_column: str,
        vocab: TokenVocab,
        view_mode: str,
        view_index: int,
        max_seq_len: int,
        drop_overlength: bool,
        fallback_oov_to_smiles: bool = False,
        motif_smiles_fallback_prob: float = 0.0,
        length_bucket_size: int = 0,
        epoch: int = 0,
        shuffle_files: bool = True,
    ) -> None:
        self.paths = list(paths)
        self.text_column = text_column
        self.vocab = vocab
        self.view_mode = view_mode
        self.view_index = int(view_index)
        self.max_seq_len = int(max_seq_len)
        self.drop_overlength = bool(drop_overlength)
        self.fallback_oov_to_smiles = bool(fallback_oov_to_smiles)
        self.motif_smiles_fallback_prob = float(motif_smiles_fallback_prob)
        self.length_bucket_size = max(0, int(length_bucket_size))
        self.epoch = int(epoch)
        self.shuffle_files = bool(shuffle_files)

    def _encode_text(self, text: str, rng: random.Random) -> Iterator[list[int]]:
        for view in _select_text_views(str(text), self.view_mode, self.view_index):
            ids_full = ids_from_stamp_tokens(
                view,
                self.vocab,
                max_seq_len=10**9,
                fallback_oov_to_smiles=self.fallback_oov_to_smiles,
                motif_smiles_fallback_prob=self.motif_smiles_fallback_prob,
                rng=rng,
            )
            if self.drop_overlength and len(ids_full) > self.max_seq_len:
                continue
            ids = ids_full[: self.max_seq_len]
            if len(ids) >= 3:
                yield ids

    def __iter__(self) -> Iterator[list[int]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)
        rank = _dist_rank()
        world = _dist_world_size()
        shard_id = rank * num_workers + worker_id
        num_shards = max(1, world * num_workers)

        rnd = random.Random(10_000 * self.epoch + rank * 997 + worker_id)
        paths = list(self.paths)
        if self.shuffle_files:
            rnd.shuffle(paths)
        paths = paths[shard_id::num_shards]

        for path in paths:
            pf = pq.ParquetFile(path)
            rg_indices = list(range(pf.num_row_groups))
            rnd.shuffle(rg_indices)
            for rg in rg_indices:
                tab = pf.read_row_group(rg, columns=[self.text_column])
                rows = tab.column(0).to_pylist()
                order = list(range(len(rows)))
                rnd.shuffle(order)
                if self.length_bucket_size > 0:
                    buckets: dict[int, list[list[int]]] = {}
                    for idx in order:
                        text = rows[idx]
                        if text is None:
                            continue
                        for ids in self._encode_text(str(text), rnd):
                            key = int(math.ceil(len(ids) / self.length_bucket_size) * self.length_bucket_size)
                            buckets.setdefault(key, []).append(ids)
                    keys = list(buckets)
                    rnd.shuffle(keys)
                    for key in keys:
                        items = buckets[key]
                        rnd.shuffle(items)
                        for ids in items:
                            yield ids
                    continue
                for idx in order:
                    text = rows[idx]
                    if text is None:
                        continue
                    for ids in self._encode_text(str(text), rnd):
                        yield ids


class MLMCollator:
    def __init__(
        self,
        vocab: TokenVocab,
        *,
        mask_prob: float,
        type_aware_masking: bool,
        frag_mask_prob: float,
        topo_mask_prob: float,
        topology_ids: set[int],
        pad_to_multiple_of: int = 8,
    ) -> None:
        self.vocab = vocab
        self.mask_prob = float(mask_prob)
        self.type_aware_masking = bool(type_aware_masking)
        self.frag_mask_prob = float(frag_mask_prob)
        self.topo_mask_prob = float(topo_mask_prob)
        self.topology_ids = set(int(x) for x in topology_ids)
        self.pad_to_multiple_of = int(pad_to_multiple_of)
        self.special_ids = {vocab.pad_id, vocab.bos_id, vocab.eos_id}

    def __call__(self, batch: Sequence[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(x) for x in batch)
        if self.pad_to_multiple_of > 1:
            rem = max_len % self.pad_to_multiple_of
            if rem:
                max_len += self.pad_to_multiple_of - rem
        x = torch.full((len(batch), max_len), self.vocab.pad_id, dtype=torch.long)
        for i, ids in enumerate(batch):
            x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        labels = x.clone()
        candidate = torch.ones_like(x, dtype=torch.bool)
        for sid in self.special_ids:
            candidate &= x.ne(int(sid))

        if self.type_aware_masking:
            probs = torch.full(x.shape, self.frag_mask_prob, dtype=torch.float32)
            if self.topology_ids:
                topo = torch.zeros_like(candidate)
                for tid in self.topology_ids:
                    topo |= x.eq(int(tid))
                probs = torch.where(topo, torch.full_like(probs, self.topo_mask_prob), probs)
        else:
            probs = torch.full(x.shape, self.mask_prob, dtype=torch.float32)
        probs.masked_fill_(~candidate, 0.0)
        masked = torch.bernoulli(probs).bool()
        if not bool(masked.any()):
            valid = candidate.nonzero(as_tuple=False)
            if valid.numel() > 0:
                row = valid[torch.randint(valid.shape[0], (1,)).item()]
                masked[row[0], row[1]] = True

        labels[~masked] = -100
        replace = torch.bernoulli(torch.full(x.shape, 0.8)).bool() & masked
        randomize = torch.bernoulli(torch.full(x.shape, 0.5)).bool() & masked & ~replace
        x[replace] = self.vocab.mask_id
        random_words = torch.randint(0, len(self.vocab.itos), x.shape, dtype=torch.long)
        x[randomize] = random_words[randomize]
        return x, labels


def _evaluate_mlm_loss(
    *,
    encoder: torch.nn.Module,
    mlm_head: torch.nn.Module,
    valid_paths: Sequence[str],
    text_column: str,
    vocab: TokenVocab,
    view_mode: str,
    view_index: int,
    max_seq_len: int,
    drop_overlength: bool,
    fallback_oov_to_smiles: bool,
    collate: MLMCollator,
    batch_size: int,
    max_raw_rows: int,
    dev: torch.device,
    non_blocking: bool,
    use_amp: bool,
    amp_dtype: torch.dtype,
    seed: int,
) -> float:
    if not valid_paths:
        return float("nan")
    enc_was_training = encoder.training
    head_was_training = mlm_head.training
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if dev.type == "cuda" else None
    torch.manual_seed(int(seed))
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    encoder.eval()
    mlm_head.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=dev)
    batch: list[list[int]] = []
    raw_seen = 0
    rank = _dist_rank()
    world = _dist_world_size()

    def flush() -> None:
        nonlocal batch
        if not batch:
            return
        x, y = collate(batch)
        batch = []
        x = x.to(dev, non_blocking=non_blocking)
        y = y.to(dev, non_blocking=non_blocking)
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp) if use_amp else nullcontext()
        with torch.no_grad(), amp_ctx:
            h = encoder(x)
            masked = y.ne(-100)
            logits = mlm_head(h[masked])
            loss_sum = F.cross_entropy(
                logits,
                y[masked],
                reduction="sum",
            )
            denom = masked.sum()
        totals[0] += loss_sum.detach().to(dtype=torch.float64)
        totals[1] += denom.detach().to(dtype=torch.float64)

    for text in _iter_text_rows(valid_paths, text_column):
        if max_raw_rows > 0 and raw_seen >= max_raw_rows:
            break
        take = (raw_seen % max(1, world)) == rank
        raw_seen += 1
        if not take:
            continue
        for view in _select_text_views(str(text), view_mode, view_index):
            ids_full = ids_from_stamp_tokens(
                view,
                vocab,
                max_seq_len=10**9,
                fallback_oov_to_smiles=fallback_oov_to_smiles,
                motif_smiles_fallback_prob=0.0,
            )
            if drop_overlength and len(ids_full) > max_seq_len:
                continue
            ids = ids_full[:max_seq_len]
            if len(ids) >= 3:
                batch.append(ids)
            if len(batch) >= batch_size:
                flush()
    flush()

    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if enc_was_training:
        encoder.train()
    if head_was_training:
        mlm_head.train()
    torch.random.set_rng_state(cpu_rng_state)
    if cuda_rng_states is not None:
        torch.cuda.set_rng_state_all(cuda_rng_states)
    return float((totals[0] / totals[1].clamp_min(1.0)).item())


def _scheduled_lr(base_lr: float, min_lr: float, step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (base_lr - min_lr)


def _make_optimizer(params, optimizer: str, lr: float, weight_decay: float):
    if optimizer == "lamb":
        return Lamb(params, lr=lr, betas=(0.9, 0.99), eps=1e-6, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-glob", required=True)
    p.add_argument("--valid-glob", default="", help="Optional validation parquet glob. If omitted, --valid-file-fraction can hold out train files.")
    p.add_argument("--valid-file-fraction", type=float, default=0.0, help="Hold out this fraction of sorted train files for validation.")
    p.add_argument("--valid-max-rows", type=int, default=200000, help="Max raw validation rows to scan per validation check.")
    p.add_argument("--eval-every-steps", type=int, default=0, help="Run validation loss every N optimizer steps. 0 disables validation.")
    p.add_argument("--early-stop-patience", type=int, default=0, help="Stop after this many non-improving validation checks. 0 disables early stop.")
    p.add_argument("--early-stop-min-delta", type=float, default=0.0, help="Minimum validation loss improvement required to reset patience.")
    p.add_argument("--text-column", default="stamp")
    p.add_argument("--stamp-vocabs", required=True, help="Comma-separated STAMP vocab files.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--view-mode", choices=["concat_only", "single_views", "cams_multiview"], default="cams_multiview")
    p.add_argument("--view-index", type=int, default=-1, help="If >=0, select one [CONCAT]-separated view before applying view-mode; use 1 for msring s2000.")
    p.add_argument("--include-smiles-vocab", action="store_true", help="Add regex SMILES tokens from the motif vocabs to the model vocabulary.")
    p.add_argument("--fallback-oov-to-smiles", action="store_true", help="Encode non-vocab STAMP motif tokens as SMILES regex tokens instead of <unk>.")
    p.add_argument("--motif-smiles-fallback-prob", type=float, default=0.0, help="Training-time probability of expanding an in-vocab motif token to SMILES regex tokens.")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--drop-overlength", action="store_true", help="Drop sequences longer than max length instead of truncating.")
    p.add_argument("--length-bucket-size", type=int, default=0, help="0 disables; otherwise bucket row-group examples by encoded length to reduce padding.")
    p.add_argument("--scan-max-rows", type=int, default=200000)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--d-ff", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--num-random-features", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4, help="Micro-batch per rank.")
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=2.4e-4)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=10000)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--optimizer", choices=["lamb", "adamw"], default="lamb")
    p.add_argument("--mask-prob", type=float, default=0.15)
    p.add_argument("--type-aware-masking", action="store_true")
    p.add_argument("--frag-mask-prob", type=float, default=0.20)
    p.add_argument("--topo-mask-prob", type=float, default=0.05)
    p.add_argument("--num-data-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="fp16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every-steps", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--resume", default="")
    p.add_argument("--no-tf32", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_paths = _sorted_paths(args.train_glob)
    train_paths, valid_paths = _split_train_valid_paths(
        train_paths,
        valid_glob=str(args.valid_glob),
        valid_file_fraction=float(args.valid_file_fraction),
        seed=int(args.seed),
    )
    stamp_vocab_paths = [x.strip() for x in str(args.stamp_vocabs).split(",") if x.strip()]
    if not stamp_vocab_paths:
        raise ValueError("--stamp-vocabs is empty.")

    local_rank = _dist_local_rank()
    dev = _resolve_torch_device(f"cuda:{local_rank}" if _is_dist() and str(args.device).startswith("cuda") else args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index if dev.index is not None else 0)
        torch.backends.cuda.matmul.allow_tf32 = not bool(args.no_tf32)
        torch.backends.cudnn.allow_tf32 = not bool(args.no_tf32)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    _setup_dist(dev)
    _set_seed(args.seed + _dist_rank())

    include_smiles_vocab = bool(args.include_smiles_vocab or args.fallback_oov_to_smiles or args.motif_smiles_fallback_prob > 0.0)
    vocab = build_stamp_molformer_vocab(stamp_vocab_paths, include_smiles_tokens=include_smiles_vocab)
    topology_ids = token_ids_are_structural(vocab)
    raw_rows = _count_parquet_rows(train_paths)
    scan = _scan_paths(
        paths=train_paths,
        text_column=args.text_column,
        vocab=vocab,
        view_mode=args.view_mode,
        view_index=int(args.view_index),
        max_seq_len=args.max_seq_len,
        scan_max_rows=args.scan_max_rows,
        drop_overlength=bool(args.drop_overlength),
        fallback_oov_to_smiles=bool(args.fallback_oov_to_smiles),
    ) if _is_main() else None
    if _is_main():
        assert scan is not None
        scan.raw_rows = raw_rows
        est_examples = int(math.ceil(raw_rows * max(1e-9, scan.examples_per_raw_row)))
        est_steps_per_epoch = int(math.ceil(est_examples / max(1, args.global_batch_size)))
        total_steps = int(args.max_steps) if args.max_steps > 0 else est_steps_per_epoch * max(1, int(args.epochs))
        meta = {
            "train_files": len(train_paths),
            "valid_files": len(valid_paths),
            "raw_rows": raw_rows,
            "estimated_examples": est_examples,
            "estimated_steps_per_epoch": est_steps_per_epoch,
            "total_steps": total_steps,
            "scan": asdict(scan),
            "vocab_size": len(vocab.itos),
            "stamp_vocabs": stamp_vocab_paths,
            "view_mode": args.view_mode,
            "view_index": int(args.view_index),
            "include_smiles_vocab": include_smiles_vocab,
            "fallback_oov_to_smiles": bool(args.fallback_oov_to_smiles),
            "motif_smiles_fallback_prob": float(args.motif_smiles_fallback_prob),
            "length_bucket_size": int(args.length_bucket_size),
            "valid_file_fraction": float(args.valid_file_fraction),
            "valid_split_seed": int(args.seed),
            "valid_max_rows": int(args.valid_max_rows),
            "eval_every_steps": int(args.eval_every_steps),
            "early_stop_patience": int(args.early_stop_patience),
            "early_stop_min_delta": float(args.early_stop_min_delta),
        }
        (out_dir / "pretrain_scan_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2), flush=True)
    else:
        total_steps = 0

    if dist.is_initialized():
        ts = torch.tensor([total_steps], dtype=torch.long, device=dev)
        dist.broadcast(ts, src=0)
        total_steps = int(ts.item())

    cfg = MolFormerSTAMPConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        num_random_features=args.num_random_features,
        vocab_size=len(vocab.itos),
        pad_id=vocab.pad_id,
        seed=args.seed,
    )
    encoder = MolFormerEncoder(cfg).to(dev)
    mlm_head = MolFormerMLMHead(cfg.d_model, len(vocab.itos)).to(dev)
    start_epoch = 0
    global_step = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        encoder.load_state_dict(payload["model_state"], strict=True)
        if payload.get("mlm_state") is not None:
            mlm_head.load_state_dict(payload["mlm_state"], strict=True)
        start_epoch = int(payload.get("epoch", 0))
        global_step = int(payload.get("global_step", 0))
        if _is_main():
            print(f"[resume] {args.resume} global_step={global_step} epoch={start_epoch}", flush=True)

    if dist.is_initialized():
        encoder = DDP(encoder, device_ids=[dev.index] if dev.type == "cuda" else None)
        mlm_head = DDP(mlm_head, device_ids=[dev.index] if dev.type == "cuda" else None)

    params = list(encoder.parameters()) + list(mlm_head.parameters())
    opt = _make_optimizer(params, args.optimizer, lr=args.lr, weight_decay=args.weight_decay)
    micro_global = max(1, args.batch_size * _dist_world_size())
    grad_accum = max(1, int(math.ceil(args.global_batch_size / micro_global)))
    use_amp, amp_dtype, use_scaler, amp_label = _resolve_amp_mode(args.amp, dev)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=use_scaler)

    collate = MLMCollator(
        vocab,
        mask_prob=args.mask_prob,
        type_aware_masking=bool(args.type_aware_masking),
        frag_mask_prob=args.frag_mask_prob,
        topo_mask_prob=args.topo_mask_prob,
        topology_ids=topology_ids,
    )
    pin = dev.type == "cuda"
    non_blocking = dev.type == "cuda"
    n_params = sum(p.numel() for p in params)
    if _is_main():
        print(
            f"[model] params={n_params/1e6:.2f}M vocab={len(vocab.itos)} amp={amp_label} "
            f"micro_batch={args.batch_size} world={_dist_world_size()} grad_accum={grad_accum}",
            flush=True,
        )

    t0 = time.time()
    last_log = time.time()
    running = 0.0
    running_steps = 0
    stop = False
    valid_history: list[dict[str, Any]] = []
    best_valid_loss = float("inf")
    bad_valid_checks = 0
    for epoch in range(start_epoch, start_epoch + max(1, args.epochs)):
        ds = ShardedSTAMPDataset(
            train_paths,
            text_column=args.text_column,
            vocab=vocab,
            view_mode=args.view_mode,
            view_index=int(args.view_index),
            max_seq_len=args.max_seq_len,
            drop_overlength=bool(args.drop_overlength),
            fallback_oov_to_smiles=bool(args.fallback_oov_to_smiles),
            motif_smiles_fallback_prob=float(args.motif_smiles_fallback_prob),
            length_bucket_size=int(args.length_bucket_size),
            epoch=epoch,
        )
        loader_kwargs: dict[str, Any] = {
            "batch_size": args.batch_size,
            "collate_fn": collate,
            "num_workers": max(0, args.num_data_workers),
            "pin_memory": pin,
            "drop_last": True,
        }
        if args.num_data_workers > 0:
            loader_kwargs["prefetch_factor"] = max(2, args.prefetch_factor)
            loader_kwargs["persistent_workers"] = True
        loader = DataLoader(ds, **loader_kwargs)

        accum_idx = 0
        opt.zero_grad(set_to_none=True)
        for x, y in loader:
            if global_step >= total_steps:
                stop = True
                break
            x = x.to(dev, non_blocking=non_blocking)
            y = y.to(dev, non_blocking=non_blocking)
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp) if use_amp else nullcontext()
            with amp_ctx:
                h = encoder(x)
                masked = y.ne(-100)
                logits = mlm_head(h[masked])
                loss = F.cross_entropy(logits, y[masked])
                loss = loss / grad_accum
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_idx += 1
            running += float(loss.detach().item()) * grad_accum
            running_steps += 1
            if accum_idx % grad_accum != 0:
                continue

            lr_now = _scheduled_lr(args.lr, args.min_lr, global_step, total_steps, args.warmup_steps)
            for group in opt.param_groups:
                group["lr"] = lr_now
            if use_scaler:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1

            if _is_main() and (global_step % args.log_every == 0 or time.time() - last_log > 120):
                dt = max(1e-9, time.time() - last_log)
                avg_loss = running / max(1, running_steps)
                print(
                    f"[train] step={global_step}/{total_steps} epoch={epoch+1} "
                    f"loss={avg_loss:.4f} lr={lr_now:.3e} micro_steps_per_s={running_steps/dt:.2f}",
                    flush=True,
                )
                running = 0.0
                running_steps = 0
                last_log = time.time()

            if _is_main() and args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                ckpt = out_dir / f"checkpoint_step_{global_step}.pt"
                save_molformer_checkpoint(
                    ckpt,
                    encoder=encoder,
                    mlm_head=mlm_head,
                    cfg=cfg,
                    vocab=vocab,
                    global_step=global_step,
                    epoch=epoch + 1,
                    extra={"args": vars(args)},
                )
                save_molformer_checkpoint(
                    out_dir / "last.pt",
                    encoder=encoder,
                    mlm_head=mlm_head,
                    cfg=cfg,
                    vocab=vocab,
                    global_step=global_step,
                    epoch=epoch + 1,
                    extra={"args": vars(args)},
                )
            if valid_paths and args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                valid_loss = _evaluate_mlm_loss(
                    encoder=encoder,
                    mlm_head=mlm_head,
                    valid_paths=valid_paths,
                    text_column=args.text_column,
                    vocab=vocab,
                    view_mode=args.view_mode,
                    view_index=int(args.view_index),
                    max_seq_len=args.max_seq_len,
                    drop_overlength=bool(args.drop_overlength),
                    fallback_oov_to_smiles=bool(args.fallback_oov_to_smiles),
                    collate=collate,
                    batch_size=args.batch_size,
                    max_raw_rows=int(args.valid_max_rows),
                    dev=dev,
                    non_blocking=non_blocking,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    seed=int(args.seed) + 104729,
                )
                should_stop = False
                if _is_main():
                    improved = valid_loss < (best_valid_loss - float(args.early_stop_min_delta))
                    if improved:
                        best_valid_loss = valid_loss
                        bad_valid_checks = 0
                        save_molformer_checkpoint(
                            out_dir / "best_valid.pt",
                            encoder=encoder,
                            mlm_head=mlm_head,
                            cfg=cfg,
                            vocab=vocab,
                            global_step=global_step,
                            epoch=epoch + 1,
                            extra={"args": vars(args), "best_valid_loss": best_valid_loss},
                        )
                    else:
                        bad_valid_checks += 1
                    row = {
                        "step": int(global_step),
                        "epoch": int(epoch + 1),
                        "valid_loss": float(valid_loss),
                        "best_valid_loss": float(best_valid_loss),
                        "improved": bool(improved),
                        "bad_valid_checks": int(bad_valid_checks),
                        "elapsed_sec": float(time.time() - t0),
                    }
                    valid_history.append(row)
                    (out_dir / "valid_history.json").write_text(json.dumps(valid_history, indent=2), encoding="utf-8")
                    print(
                        f"[valid] step={global_step}/{total_steps} epoch={epoch+1} "
                        f"valid_loss={valid_loss:.4f} best={best_valid_loss:.4f} "
                        f"bad={bad_valid_checks}/{args.early_stop_patience}",
                        flush=True,
                    )
                    should_stop = bool(args.early_stop_patience > 0 and bad_valid_checks >= args.early_stop_patience)
                if dist.is_initialized():
                    flag = torch.tensor([1 if should_stop else 0], dtype=torch.long, device=dev)
                    dist.broadcast(flag, src=0)
                    should_stop = bool(flag.item())
                if should_stop:
                    if _is_main():
                        print(f"[early-stop] step={global_step} best_valid_loss={best_valid_loss:.4f}", flush=True)
                    stop = True
                    break
        if stop:
            break

    if _is_main():
        save_molformer_checkpoint(
            out_dir / "last.pt",
            encoder=encoder,
            mlm_head=mlm_head,
            cfg=cfg,
            vocab=vocab,
            global_step=global_step,
            epoch=epoch + 1,
            extra={
                "args": vars(args),
                "elapsed_sec": time.time() - t0,
                "best_valid_loss": best_valid_loss if valid_history else None,
                "early_stopped": bool(stop and valid_history),
            },
        )
        print(f"[done] global_step={global_step} elapsed_sec={time.time()-t0:.1f}", flush=True)
    _barrier(dev)
    _cleanup_dist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
