#!/usr/bin/env python3
"""Eso-LM (arXiv 2506.01928) pretraining at alpha_0=1 from parquet.

Same data pipeline as the MDLM trainer, but the model and loss combine
AO-GPT's shuffled causal attention with MDM-style masked denoising:

  * model: ``TinyDecoderLM(bidirectional=False, use_adaln=True)`` — causal
    attention on a shuffled order, with target-pos AdaLN conditioning.
  * loss: ``_compute_eso_loss`` — sample a permutation per row, mask some
    positions at random time t, and compute masked CE under causal attention
    along the shuffled order (Remark 2 of the paper: train at alpha_0=1 and
    vary alpha_0 at inference).
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
from stamp.benchmark.lm import (  # noqa: E402
    LMConfig,
    TinyEsoDiT,
    TokenVocab,
    _compute_eso_loss_v2,
    _resolve_amp_mode,
    _resolve_torch_device,
    _set_seed,
)
from stamp.vocab.branch_paths import resolve_branch_output_dir  # noqa: E402

from scripts.benchmark.train_ao_gpt_parquet import (  # noqa: E402
    ScanSummary,
    _cleanup_distributed,
    _dist_local_rank,
    _dist_rank,
    _dist_world_size,
    _distributed_barrier,
    _distributed_mean,
    _is_dist,
    _is_main_process,
    _load_scan_cache,
    _load_vocab_seed_tokens,
    _save_scan_cache,
    _scan_cache_path,
    _scan_parquet_for_vocab_and_valid,
    _setup_distributed,
    _unwrap_model,
    _debug_print,
)


class _ParquetFullSeqDataset(IterableDataset[list[int]]):
    """Yields the full ``[BOS, tok1, ..., tokN, EOS]`` id list for each row."""

    def __init__(
        self,
        parquet_path: str | Path,
        *,
        vocab: TokenVocab,
        text_column: str,
        split_column: str,
        split_value: str,
        require_all_pass: bool,
        max_seq_len: int,
        max_rows: Optional[int],
        seed: int,
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
        self.epoch = 0
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)
        shard_id = self.rank * num_workers + worker_id
        num_shards = self.world_size * num_workers

        pf = pq.ParquetFile(self.parquet_path)
        rg_indices = list(range(pf.num_row_groups))
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(rg_indices)
        rg_indices = rg_indices[shard_id::num_shards]

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
            rng.shuffle(order)
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
                ids = [self.vocab.bos_id] + self.vocab.encode_tokens(toks) + [self.vocab.eos_id]
                if len(ids) < 2:
                    continue
                ids = ids[: self.max_seq_len]
                yield ids
                yielded += 1
                if worker_limit is not None and yielded >= worker_limit:
                    return


def _build_denoise_collate(pad_id: int):
    def _collate(batch: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(x) for x in batch)
        bsz = len(batch)
        x_out = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        y_out = torch.full((bsz, max_len), -100, dtype=torch.long)
        for i, ids in enumerate(batch):
            n = len(ids)
            t = torch.tensor(ids, dtype=torch.long)
            x_out[i, :n] = t
            y_out[i, :n] = t
            # Keep BOS fixed (never masked) by marking label as -100.
            y_out[i, 0] = -100
        return x_out, y_out

    return _collate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Large-scale MDLM pretraining directly from parquet stamp sequences.")
    p.add_argument("--parquet", required=True)
    p.add_argument("--stamp-vocab", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--text-column", default="stamp_smiles")
    p.add_argument("--split-column", default="split_random")
    p.add_argument("--train-split", default="train")
    p.add_argument("--valid-split", default="val")
    p.add_argument("--require-all-pass", action="store_true", default=True)
    p.add_argument("--include-failures", dest="require_all_pass", action="store_false")
    p.add_argument("--train-max-rows", type=int, default=0, help="0 = all train rows.")
    p.add_argument("--valid-max-rows", type=int, default=20000)
    p.add_argument("--scan-only", action="store_true")
    p.add_argument("--no-tqdm", action="store_true")

    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--d-ff", type=int, default=3072)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--global-batch-size", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-data-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "none"])
    p.add_argument("--torch-compile", action="store_true", default=True)
    p.add_argument("--no-torch-compile", dest="torch_compile", action="store_false")
    p.add_argument("--no-tf32", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-parallel", action="store_true")
    p.add_argument("--save-every-epoch", action="store_true", default=True)
    p.add_argument("--no-save-every-epoch", dest="save_every_epoch", action="store_false")
    p.add_argument("--reuse-scan-cache", action="store_true")
    p.add_argument("--debug-phase-prints", action="store_true")
    p.add_argument("--diffusion-time-epsilon", type=float, default=1e-3)
    p.add_argument("--diffusion-scheduler", default="linear", choices=["linear", "cosine"])
    p.add_argument("--diffusion-loss-weight", default="scheduler", choices=["scheduler", "uniform"])
    p.add_argument("--resume-from", default="",
                   help="Path to a previous ckpt (model.pt) or run dir. Loads model/optimizer/scaler "
                        "and continues from ckpt['epoch'] + 1 up to --epochs.")
    p.add_argument("--alpha-0", type=float, default=0.0,
                   help="Eso-LM sequential-branch mask boundary (0=pure AR, 1=pure MDM). "
                        "Mask rate during training is (1 - alpha_0).")
    p.add_argument("--shuffle-unmasked", action="store_true", default=False,
                   help="Within the unmasked prefix of the sorted sequence, shuffle randomly "
                        "(matches sequential_shuffle=True in the paper). Default: keep original order.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")
    debug = bool(args.debug_phase_prints)

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    stamp_vocab = Path(args.stamp_vocab)
    if not stamp_vocab.exists():
        raise FileNotFoundError(stamp_vocab)

    output_dir = resolve_branch_output_dir(args.output_dir, stamp_vocab=str(stamp_vocab))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    local_rank = _dist_local_rank()
    cfg = LMConfig(
        model_type="eso_v2",
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
        seed=args.seed,
        num_data_workers=args.num_data_workers,
        prefetch_factor=args.prefetch_factor,
        amp=args.amp,
        torch_compile=bool(args.torch_compile),
        tf32=not bool(args.no_tf32),
        diffusion_time_epsilon=args.diffusion_time_epsilon,
        diffusion_scheduler=args.diffusion_scheduler,
        diffusion_loss_weight=args.diffusion_loss_weight,
        use_adaln=False,
    )
    _set_seed(cfg.seed)
    if args.data_parallel and _is_dist():
        raise ValueError("Cannot combine --data-parallel with torchrun/DDP.")
    if _is_dist() and str(args.device).startswith("cuda"):
        dev = _resolve_torch_device(f"cuda:{local_rank}")
    else:
        dev = _resolve_torch_device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    _setup_distributed(dev)

    vocab_header, seed_tokens = _load_vocab_seed_tokens(stamp_vocab)
    # Ensure <mask> is in the vocab for diffusion training.
    if TokenVocab.MASK not in seed_tokens:
        seed_tokens = list(seed_tokens) + [TokenVocab.MASK]
    train_max_rows = (None if int(args.train_max_rows) <= 0 else int(args.train_max_rows))

    if _is_main_process():
        cache_path = _scan_cache_path(out_dir)
        if bool(args.reuse_scan_cache) and cache_path.exists():
            scan_summary, vocab = _load_scan_cache(out_dir)
        else:
            scan_summary, vocab = _scan_parquet_for_vocab_and_valid(
                parquet_path=parquet_path,
                text_column=args.text_column,
                split_column=args.split_column,
                train_split=args.train_split,
                valid_split=args.valid_split,
                seed_tokens=seed_tokens,
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
                    "mask_id": vocab.mask_id,
                    "world_size": _dist_world_size(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    if not _is_main_process():
        cache_path = _scan_cache_path(out_dir)
        while not cache_path.exists():
            time.sleep(0.5)
        scan_summary, vocab = _load_scan_cache(out_dir)

    # Verify mask token exists in the rebuilt vocab.
    if vocab.mask_id == vocab.unk_id and TokenVocab.MASK not in vocab.stoi:
        _cleanup_distributed()
        raise RuntimeError(
            "Vocab cache does not contain <mask>; delete the scan cache and re-run so "
            "the seed tokens include <mask> for diffusion training."
        )

    if args.scan_only:
        _cleanup_distributed()
        return 0

    train_ds = _ParquetFullSeqDataset(
        parquet_path=parquet_path,
        vocab=vocab,
        text_column=args.text_column,
        split_column=args.split_column,
        split_value=args.train_split,
        require_all_pass=bool(args.require_all_pass),
        max_seq_len=cfg.max_seq_len,
        max_rows=train_max_rows,
        seed=cfg.seed,
        rank=_dist_rank(),
        world_size=_dist_world_size(),
    )
    # Valid is a deterministic list materialized during scan.
    valid_seqs_full = list(scan_summary.valid_sequences)
    # Shard valid across ranks.
    world = max(1, _dist_world_size())
    valid_seqs_rank = valid_seqs_full[_dist_rank() :: world]
    collate_fn = _build_denoise_collate(vocab.pad_id)

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
        valid_seqs_rank,
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

    use_amp, amp_dtype, use_scaler, _amp_label = _resolve_amp_mode(cfg.amp, dev)

    # Full Eso-LM (arXiv 2506.01928) sequential branch with block_diff_mask
    # over [zt, x0] concatenation. use_adaln is unused in this backbone.
    cfg.use_adaln = False
    model = TinyEsoDiT(vocab_size=len(vocab.itos), cfg=cfg)
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

    n_model_params = int(sum(p.numel() for p in model.parameters()))
    if _is_main_process():
        print(f"[train-eso-v2] n_model_params={n_model_params} (~{n_model_params/1e6:.2f}M)", flush=True)

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
            print(f"[train-eso-v2] resumed from {resume} (epoch {ck.get('epoch','?')}) → start_epoch={start_epoch}", flush=True)
    non_blocking = dev.type == "cuda"
    dp_world_size = torch.cuda.device_count() if bool(args.data_parallel) and dev.type == "cuda" else 1
    effective_global_batch = int(args.global_batch_size) if int(args.global_batch_size) > 0 else int(cfg.batch_size)
    min_global_batch = int(cfg.batch_size) * max(1, _dist_world_size())
    if effective_global_batch < min_global_batch:
        raise ValueError("global_batch_size must be >= batch_size when provided.")
    grad_accum_steps = max(1, math.ceil(effective_global_batch / max(1, int(cfg.batch_size) * max(1, _dist_world_size()))))

    train_rows_rank = math.ceil(scan_summary.train_rows / max(1, _dist_world_size()))
    train_steps = max(1, math.ceil(train_rows_rank / max(1, cfg.batch_size)))
    train_opt_steps = max(1, math.ceil(train_rows_rank / max(1, int(cfg.batch_size) * grad_accum_steps)))
    ckpt_dir = out_dir / "checkpoints"
    if args.save_every_epoch and _is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    if _is_main_process():
        shutil.copy2(stamp_vocab, out_dir / "stamp_vocab.txt")

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
            desc=f"Train mdlm[stamp] ep{epoch}",
        )
        micro_step = 0
        for x, y in iterator:
            micro_step += 1
            x = x.to(dev, non_blocking=non_blocking)
            y = y.to(dev, non_blocking=non_blocking)
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True) if use_amp else nullcontext()
            with amp_ctx:
                loss = _compute_eso_loss_v2(
                    model, x, y, cfg,
                    mask_id=vocab.mask_id,
                    alpha_0=args.alpha_0,
                    shuffle_unmasked=args.shuffle_unmasked,
                )
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
                    loss = _compute_eso_loss_v2(
                    model, x, y, cfg,
                    mask_id=vocab.mask_id,
                    alpha_0=args.alpha_0,
                    shuffle_unmasked=args.shuffle_unmasked,
                )
                valid_losses.append(float(loss.item()))
        va = float(np.mean(valid_losses)) if valid_losses else float("nan")
        va = _distributed_mean(va, device=dev)

        if not _is_main_process():
            continue
        epoch_wall_seconds = float(time.time() - epoch_t0)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_mean,
                "valid_loss": va,
                "epoch_wall_seconds": epoch_wall_seconds,
            }
        )
        print(
            f"[train-eso-v2] epoch={epoch} train_loss={train_loss_mean:.4f} "
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
            "model_type": "eso_v2",
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
