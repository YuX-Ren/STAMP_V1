#!/usr/bin/env python3
"""AO-GPT pretraining with the hybrid motif+char STAMP vocab.

Vocab is built from the static motif-frequency file (``v3_cm_union.txt``),
NOT counted from parquet. Atomic ids cover motifs with freq >= cutoff
(default 5000, ~2387 motifs, 91% of occurrences); rare motifs are always
expanded to SMILES char tokens. During training, atomic motifs are also
expanded to chars with a log-interpolated probability (~2% at the most
frequent motif, ~15% at the cutoff) so the model sees both forms.

Everything else — model, loss, training loop, checkpoint layout — matches
``train_ao_gpt_parquet.py``; only the vocab and dataset differ.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from rdkit import RDLogger
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.common import iterable_with_progress  # noqa: E402
from stamp.benchmark.hybrid_vocab import HybridVocab  # noqa: E402
from stamp.benchmark.lm import (  # noqa: E402
    LMConfig,
    TinyDecoderLM,
    _compute_ao_gpt_loss,
    _resolve_amp_mode,
    _resolve_torch_device,
    _set_seed,
)
from stamp.vocab.branch_paths import resolve_branch_output_dir  # noqa: E402

from scripts.benchmark.train_ao_gpt_parquet import (  # noqa: E402
    ScanSummary,
    _cleanup_distributed,
    _dist_rank,
    _dist_world_size,
    _distributed_barrier,
    _distributed_mean,
    _histogram,
    _is_dist,
    _is_main_process,
    _setup_distributed,
    _unwrap_model,
    build_dynamic_lm_collate_fn,
)
from stamp.benchmark.lm import NextTokenDataset, build_dynamic_lm_collate_fn as _unused  # noqa: E402,F401


_SCAN_CACHE_NAME = "_scan_cache_hybrid.pt"


def _scan_cache_path(out_dir: Path) -> Path:
    return out_dir / _SCAN_CACHE_NAME


def _save_scan_cache(out_dir: Path, summary: ScanSummary, vocab: HybridVocab) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "summary": {
                "train_rows": summary.train_rows,
                "valid_rows_total": summary.valid_rows_total,
                "valid_rows_sampled": summary.valid_rows_sampled,
                "train_lengths": summary.train_lengths,
                "valid_lengths": summary.valid_lengths,
                "valid_sequences": summary.valid_sequences,
            },
            "vocab": vocab.to_dict(),
        },
        _scan_cache_path(out_dir),
    )


def _load_scan_cache(out_dir: Path) -> tuple[ScanSummary, HybridVocab]:
    payload = torch.load(_scan_cache_path(out_dir), map_location="cpu", weights_only=False)
    s = payload["summary"]
    summary = ScanSummary(
        train_rows=s["train_rows"],
        valid_rows_total=s["valid_rows_total"],
        valid_rows_sampled=s["valid_rows_sampled"],
        train_lengths=s["train_lengths"],
        valid_lengths=s["valid_lengths"],
        train_hist=_histogram(s["train_lengths"]),
        valid_hist=_histogram(s["valid_lengths"]),
        token_counter=None,
        valid_sequences=s["valid_sequences"],
    )
    vocab = HybridVocab.from_dict(payload["vocab"])
    return summary, vocab


def _scan_parquet_for_valid(
    parquet_path: str | Path,
    vocab: HybridVocab,
    *,
    text_column: str,
    split_column: str,
    train_split: str,
    valid_split: str,
    require_all_pass: bool,
    train_max_rows: Optional[int],
    valid_max_rows: int,
    seed: int,
    show_progress: bool,
) -> ScanSummary:
    """Same flow as the AO-GPT scan but without counting motifs — vocab is
    fixed by the static motif file. We still materialize a valid_sequences
    list (encoded without fallback, i.e., in the "clean" atomic form)."""
    train_rows = 0
    valid_rows_total = 0
    train_lengths: list[int] = []
    valid_lengths: list[int] = []
    valid_sample_tokens: list[list[str]] = []
    rng = random.Random(seed + 17)

    pf = pq.ParquetFile(str(parquet_path))
    columns = [text_column, split_column]
    if require_all_pass:
        columns.append("all_pass")
    row_group_iter = iterable_with_progress(
        range(pf.num_row_groups),
        show_progress=show_progress,
        total=pf.num_row_groups,
        desc="Scan parquet (hybrid)",
    )
    for rg in row_group_iter:
        tab = pf.read_row_group(rg, columns=columns)
        cols = {name: tab.column(i).to_pylist() for i, name in enumerate(columns)}
        n = len(cols[text_column])
        for i in range(n):
            if require_all_pass and not bool(cols["all_pass"][i]):
                continue
            text = cols[text_column][i]
            if text is None:
                continue
            text = str(text).strip()
            if not text:
                continue
            split = cols[split_column][i]
            toks = text.split()
            if not toks:
                continue
            if split == train_split:
                train_rows += 1
                train_lengths.append(len(toks))
                if train_max_rows is not None and train_rows >= train_max_rows:
                    break
            elif split == valid_split:
                valid_rows_total += 1
                valid_lengths.append(len(toks))
                if len(valid_sample_tokens) < valid_max_rows:
                    valid_sample_tokens.append(toks)
                else:
                    j = rng.randint(0, valid_rows_total - 1)
                    if j < valid_max_rows:
                        valid_sample_tokens[j] = toks
        if train_max_rows is not None and train_rows >= train_max_rows:
            break

    # Valid sequences encoded WITHOUT fallback expansion (clean atomic form).
    valid_sequences = [
        [vocab.bos_id] + vocab.encode_stream(toks, training=False) + [vocab.eos_id]
        for toks in valid_sample_tokens
    ]
    return ScanSummary(
        train_rows=train_rows,
        valid_rows_total=valid_rows_total,
        valid_rows_sampled=len(valid_sequences),
        train_lengths=train_lengths,
        valid_lengths=valid_lengths[: len(valid_sample_tokens)],
        train_hist=_histogram(train_lengths),
        valid_hist=_histogram(valid_lengths[: len(valid_sample_tokens)]),
        token_counter=None,
        valid_sequences=valid_sequences,
    )


class _HybridParquetDataset(IterableDataset[tuple[list[int], list[int]]]):
    """Streams parquet rows, encodes with HybridVocab (training fallback on)."""

    def __init__(
        self,
        parquet_path: str | Path,
        *,
        vocab: HybridVocab,
        text_column: str,
        split_column: str,
        split_value: str,
        require_all_pass: bool,
        max_seq_len: int,
        max_rows: Optional[int],
        seed: int,
        fallback_p_low: float = 0.02,
        fallback_p_high: float = 0.15,
        training_fallback: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
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

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[tuple[list[int], list[int]]]:
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)
        shard_id = self.rank * num_workers + worker_id
        num_shards = self.world_size * num_workers

        pf = pq.ParquetFile(self.parquet_path)
        rg_indices = list(range(pf.num_row_groups))
        # Deterministic-per-epoch shuffle; per-row stochastic fallback uses a
        # fresh RNG derived from seed + epoch + worker so the fallback pattern
        # varies across epochs without leaking between workers.
        shuffle_rng = random.Random(self.seed + self.epoch)
        shuffle_rng.shuffle(rg_indices)
        rg_indices = rg_indices[shard_id::num_shards]
        fallback_rng = random.Random(self.seed * 131 + self.epoch * 17 + shard_id)

        if self.max_rows is not None:
            base = self.max_rows // num_shards
            rem = self.max_rows % num_shards
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
                if cols[self.split_column][idx] != self.split_value:
                    continue
                if self.require_all_pass and not bool(cols["all_pass"][idx]):
                    continue
                text = cols[self.text_column][idx]
                if text is None:
                    continue
                toks = str(text).split()
                if not toks:
                    continue
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
                if len(ids) < 2:
                    continue
                x = ids[:-1][: self.max_seq_len]
                y = ids[1:][: self.max_seq_len]
                if not x:
                    continue
                yield x, y
                yielded += 1
                if worker_limit is not None and yielded >= worker_limit:
                    return


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AO-GPT pretraining with hybrid motif+char STAMP vocab.")
    p.add_argument("--parquet", required=True)
    p.add_argument("--motif-vocab", required=True, help="Path to v3_cm_union.txt-style motif freq file.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--text-column", default="stamp_smiles")
    p.add_argument("--split-column", default="split_random")
    p.add_argument("--train-split", default="train")
    p.add_argument("--valid-split", default="val")
    p.add_argument("--require-all-pass", action="store_true", default=True)
    p.add_argument("--include-failures", dest="require_all_pass", action="store_false")
    p.add_argument("--train-max-rows", type=int, default=0, help="0 = all.")
    p.add_argument("--valid-max-rows", type=int, default=20000)
    p.add_argument("--scan-only", action="store_true")
    p.add_argument("--no-tqdm", action="store_true")

    p.add_argument("--freq-cutoff", type=int, default=5000)
    p.add_argument("--fallback-p-low", type=float, default=0.02)
    p.add_argument("--fallback-p-high", type=float, default=0.15)
    p.add_argument("--no-fallback", dest="training_fallback", action="store_false", default=True)

    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--global-batch-size", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--amp", default="bf16")
    p.add_argument("--num-data-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=6)
    p.add_argument("--torch-compile", dest="torch_compile", action="store_true")
    p.add_argument("--no-torch-compile", dest="torch_compile", action="store_false")
    p.set_defaults(torch_compile=False)
    p.add_argument("--data-parallel", action="store_true", default=False)
    p.add_argument("--reuse-scan-cache", action="store_true", default=False)
    p.add_argument("--tf32", action="store_true", default=True)
    p.add_argument("--save-every-epoch", action="store_true", default=True)
    p.add_argument("--random-ratio", type=float, default=0.9,
                   help="AO-GPT random-order ratio (fraction of batches with random permutation).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from", default="")
    return p


def main() -> int:
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    motif_vocab_path = Path(args.motif_vocab)
    if not motif_vocab_path.exists():
        raise FileNotFoundError(motif_vocab_path)

    output_dir = resolve_branch_output_dir(args.output_dir, stamp_vocab=str(motif_vocab_path))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = _resolve_torch_device(args.device)
    _setup_distributed(dev)
    _set_seed(args.seed + _dist_rank())

    cfg = LMConfig(
        model_type="ao_gpt",
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        amp=args.amp,
        num_data_workers=args.num_data_workers,
        prefetch_factor=args.prefetch_factor,
        torch_compile=args.torch_compile,
        tf32=args.tf32,
        seed=args.seed,
    )

    # Build HybridVocab from the static motif-frequency file. No parquet
    # counting needed; the vocab is fully determined by the motif list.
    vocab = HybridVocab.build(str(motif_vocab_path), freq_cutoff=args.freq_cutoff)

    train_max_rows = int(args.train_max_rows) if int(args.train_max_rows) > 0 else None

    if _is_main_process():
        cache_path = _scan_cache_path(out_dir)
        if args.reuse_scan_cache and cache_path.exists():
            scan_summary, vocab = _load_scan_cache(out_dir)
        else:
            scan_summary = _scan_parquet_for_valid(
                parquet_path,
                vocab=vocab,
                text_column=args.text_column,
                split_column=args.split_column,
                train_split=args.train_split,
                valid_split=args.valid_split,
                require_all_pass=bool(args.require_all_pass),
                train_max_rows=train_max_rows,
                valid_max_rows=int(args.valid_max_rows),
                seed=cfg.seed,
                show_progress=not args.no_tqdm,
            )
            _save_scan_cache(out_dir, scan_summary, vocab)
        print(
            json.dumps(
                {
                    "train_rows": scan_summary.train_rows,
                    "valid_rows_total": scan_summary.valid_rows_total,
                    "valid_rows_sampled": scan_summary.valid_rows_sampled,
                    "vocab_size": len(vocab.itos),
                    "atomic_motifs": len(vocab.atomic_motifs),
                    "mask_id": vocab.mask_id,
                    "freq_cutoff": vocab.freq_cutoff,
                    "world_size": _dist_world_size(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    _distributed_barrier(device=dev)
    if not _is_main_process():
        cache_path = _scan_cache_path(out_dir)
        while not cache_path.exists():
            time.sleep(0.5)
        scan_summary, vocab = _load_scan_cache(out_dir)

    if args.scan_only:
        _cleanup_distributed()
        return 0

    train_ds = _HybridParquetDataset(
        parquet_path=parquet_path,
        vocab=vocab,
        text_column=args.text_column,
        split_column=args.split_column,
        split_value=args.train_split,
        require_all_pass=bool(args.require_all_pass),
        max_seq_len=cfg.max_seq_len,
        max_rows=train_max_rows,
        seed=cfg.seed,
        fallback_p_low=args.fallback_p_low,
        fallback_p_high=args.fallback_p_high,
        training_fallback=bool(args.training_fallback),
        rank=_dist_rank(),
        world_size=_dist_world_size(),
    )
    valid_seqs_full = list(scan_summary.valid_sequences)
    world = max(1, _dist_world_size())
    valid_seqs_rank = valid_seqs_full[_dist_rank() :: world]
    valid_ds = NextTokenDataset(valid_seqs_rank, cfg.max_seq_len, vocab.pad_id)

    collate_fn = build_dynamic_lm_collate_fn(vocab.pad_id)

    pin_memory = dev.type == "cuda"
    train_loader_kwargs: dict[str, Any] = dict(
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=max(0, int(cfg.num_data_workers)),
    )
    if cfg.num_data_workers > 0:
        train_loader_kwargs["prefetch_factor"] = max(2, int(cfg.prefetch_factor))
        train_loader_kwargs["persistent_workers"] = True
    loader_train = DataLoader(train_ds, **train_loader_kwargs)
    loader_valid = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=0,
    )

    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    use_amp, amp_dtype, use_scaler, _ = _resolve_amp_mode(cfg.amp, dev)

    # AO-GPT architecture: causal + target-pos AdaLN (use_adaln=True).
    cfg.use_adaln = True
    model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg, bidirectional=False)
    if cfg.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass
    model.to(dev)

    if _is_dist():
        model = DDP(
            model,
            device_ids=[dev.index] if dev.type == "cuda" and dev.index is not None else None,
            output_device=(dev.index if dev.type == "cuda" else None),
        )
    elif bool(args.data_parallel):
        if dev.type != "cuda" or torch.cuda.device_count() <= 1:
            raise ValueError("--data-parallel requires multiple visible CUDA devices.")
        model = DataParallel(model)

    n_params = int(sum(p.numel() for p in model.parameters()))
    if _is_main_process():
        print(f"[train-hybrid] n_model_params={n_params} (~{n_params/1e6:.2f}M)", flush=True)

    adamw_kwargs: dict[str, Any] = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    if dev.type == "cuda":
        adamw_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    except TypeError:
        adamw_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)  # pragma: no cover

    start_epoch = 1
    history: list[dict[str, float]] = []
    if args.resume_from:
        resume = Path(args.resume_from)
        if resume.is_dir():
            resume = resume / "model.pt"
        if not resume.exists():
            raise FileNotFoundError(f"--resume-from path does not exist: {resume}")
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        sd = ck["model_state"]
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
        _unwrap_model(model).load_state_dict(sd)
        if ck.get("optimizer_state") is not None:
            optimizer.load_state_dict(ck["optimizer_state"])
        if ck.get("scaler_state") is not None and scaler is not None:
            try:
                scaler.load_state_dict(ck["scaler_state"])
            except Exception:
                pass
        start_epoch = int(ck.get("epoch", 0)) + 1
        hist_path = out_dir / "history.json"
        if hist_path.exists():
            try:
                history = json.loads(hist_path.read_text(encoding="utf-8"))
            except Exception:
                history = []
        if _is_main_process():
            print(f"[train-hybrid] resumed from {resume} (epoch {ck.get('epoch','?')}) → start_epoch={start_epoch}", flush=True)

    non_blocking = dev.type == "cuda"
    effective_global_batch = int(args.global_batch_size) if int(args.global_batch_size) > 0 else int(cfg.batch_size)
    min_global_batch = int(cfg.batch_size) * max(1, _dist_world_size())
    if effective_global_batch < min_global_batch:
        raise ValueError("global_batch_size must be >= batch_size * world_size when provided.")
    grad_accum_steps = max(1, math.ceil(effective_global_batch / max(1, int(cfg.batch_size) * max(1, _dist_world_size()))))

    train_rows_rank = math.ceil(scan_summary.train_rows / max(1, _dist_world_size()))
    train_steps = max(1, math.ceil(train_rows_rank / max(1, cfg.batch_size)))
    ckpt_dir = out_dir / "checkpoints"
    if args.save_every_epoch and _is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    if _is_main_process():
        # Copy vocab + motif list alongside checkpoints
        (out_dir / "hybrid_vocab.json").write_text(
            json.dumps(vocab.to_dict(), ensure_ascii=False), encoding="utf-8"
        )
        shutil.copy2(motif_vocab_path, out_dir / "motif_vocab.txt")

    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_t0 = time.time()
        train_ds.set_epoch(epoch)
        model.train()
        train_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        iterator = iterable_with_progress(
            loader_train,
            show_progress=(not args.no_tqdm) and _is_main_process(),
            total=train_steps,
            desc=f"Train ao-gpt[hybrid] ep{epoch}",
        )
        micro_step = 0
        for x, y in iterator:
            micro_step += 1
            x = x.to(dev, non_blocking=non_blocking)
            y = y.to(dev, non_blocking=non_blocking)
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True) if use_amp else nullcontext()
            with amp_ctx:
                loss = _compute_ao_gpt_loss(model, x, y, cfg, random_ratio=args.random_ratio)
                loss_for_backward = loss / float(grad_accum_steps)
            if use_scaler:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()
            should_step = (micro_step % grad_accum_steps == 0) or (micro_step == train_steps)
            if should_step:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            train_losses.append(float(loss.item()))

        train_loss_mean = float(np.mean(train_losses)) if train_losses else float("nan")
        train_loss_mean = _distributed_mean(train_loss_mean, device=dev)

        model.eval()
        valid_losses: list[float] = []
        with torch.no_grad():
            for x, y in loader_valid:
                x = x.to(dev, non_blocking=non_blocking)
                y = y.to(dev, non_blocking=non_blocking)
                amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True) if use_amp else nullcontext()
                with amp_ctx:
                    loss = _compute_ao_gpt_loss(model, x, y, cfg, random_ratio=args.random_ratio)
                valid_losses.append(float(loss.item()))
        va = float(np.mean(valid_losses)) if valid_losses else float("nan")
        va = _distributed_mean(va, device=dev)

        if not _is_main_process():
            continue
        epoch_wall_seconds = float(time.time() - epoch_t0)
        history.append(
            {"epoch": epoch, "train_loss": train_loss_mean, "valid_loss": va, "epoch_wall_seconds": epoch_wall_seconds}
        )
        print(
            f"[train-hybrid] epoch={epoch} train_loss={train_loss_mean:.4f} "
            f"valid_loss={va:.4f} epoch_wall_s={epoch_wall_seconds:.1f}",
            flush=True,
        )
        (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        model_state = _unwrap_model(model).state_dict()
        ckpt_payload = {
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": (scaler.state_dict() if scaler is not None else None),
            "cfg": vars(cfg),
            "representation": "stamp",
            "model_type": "ao_gpt_hybrid",
            "epoch": epoch,
        }
        torch.save(ckpt_payload, out_dir / "model.pt")
        if args.save_every_epoch:
            epoch_dir = ckpt_dir / f"epoch_{epoch:04d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {k: v for k, v in ckpt_payload.items() if k != "optimizer_state"},
                epoch_dir / "model.pt",
            )

    _cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
