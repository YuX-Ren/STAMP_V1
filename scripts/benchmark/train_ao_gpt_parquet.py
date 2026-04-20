#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
import sys
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from rdkit import RDLogger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from torch.utils.data import DataLoader, IterableDataset, Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.common import iterable_with_progress  # noqa: E402
from stamp.benchmark.lm import (  # noqa: E402
    LMConfig,
    NextTokenDataset,
    TinyDecoderLM,
    TokenVocab,
    _compute_ao_gpt_loss,
    _resolve_amp_mode,
    _resolve_torch_device,
    _set_seed,
    build_dynamic_lm_collate_fn,
)
from stamp.vocab.branch_paths import resolve_branch_output_dir  # noqa: E402


def _dist_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _dist_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _dist_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _is_dist() -> bool:
    return _dist_world_size() > 1


def _is_main_process() -> bool:
    return _dist_rank() == 0


def _setup_distributed(dev: torch.device) -> None:
    if not _is_dist():
        return
    if dist.is_initialized():
        return
    backend = "nccl" if dev.type == "cuda" else "gloo"
    dist.init_process_group(backend=backend)


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _distributed_mean(value: float, *, device: torch.device) -> float:
    if not dist.is_initialized():
        return float(value)
    tensor = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(_dist_world_size())
    return float(tensor.item())


def _distributed_barrier(device: torch.device) -> None:
    if not dist.is_initialized():
        return
    if device.type == "cuda" and device.index is not None:
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        return model.module  # type: ignore[return-value]
    if hasattr(model, "_orig_mod"):
        return model._orig_mod  # type: ignore[return-value]
    return model


def _debug_print(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, flush=True)
        debug_log = os.environ.get("AO_DEBUG_LOG", "").strip()
        if debug_log:
            try:
                with Path(debug_log).open("a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass


def _summarize_lengths(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {
            "count": 0,
            "min": 0,
            "mean": 0.0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
        }
    vals = sorted(int(x) for x in lengths)
    def _pct(q: float) -> int:
        idx = int(math.ceil((len(vals) - 1) * q))
        idx = min(len(vals) - 1, max(0, idx))
        return int(vals[idx])
    return {
        "count": len(vals),
        "min": int(vals[0]),
        "mean": float(sum(vals) / len(vals)),
        "p50": _pct(0.50),
        "p90": _pct(0.90),
        "p95": _pct(0.95),
        "p99": _pct(0.99),
        "max": int(vals[-1]),
    }


def _histogram(lengths: Iterable[int]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for x in lengths:
        key = str(int(x))
        hist[key] = hist.get(key, 0) + 1
    return hist


def _load_vocab_seed_tokens(vocab_path: str | Path) -> tuple[dict[str, Any], list[str]]:
    path = Path(vocab_path)
    with path.open("r", encoding="utf-8") as f:
        header = json.loads(f.readline())
        toks = [line.split("\t")[0] for line in f if line.strip()]
    return header, toks


def _build_vocab_from_counter(
    counter: Counter[str],
    *,
    seed_tokens: list[str],
) -> TokenVocab:
    vocab = [TokenVocab.PAD, TokenVocab.BOS, TokenVocab.EOS, TokenVocab.UNK]
    added = set(vocab)
    for tok in seed_tokens:
        if tok not in added:
            vocab.append(tok)
            added.add(tok)
    for tok, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if freq <= 0:
            continue
        if tok not in added:
            vocab.append(tok)
            added.add(tok)
    stoi = {tok: i for i, tok in enumerate(vocab)}
    return TokenVocab(stoi=stoi, itos=vocab)


def _iter_parquet_rows(
    parquet_path: str | Path,
    *,
    text_column: str,
    split_column: str,
    split_value: str,
    require_all_pass: bool,
) -> Iterator[str]:
    pf = pq.ParquetFile(str(parquet_path))
    columns = [text_column, split_column]
    if require_all_pass:
        columns.append("all_pass")
    for rg in range(pf.num_row_groups):
        tab = pf.read_row_group(rg, columns=columns)
        cols = {name: tab.column(i).to_pylist() for i, name in enumerate(columns)}
        n = len(cols[text_column])
        for i in range(n):
            if cols[split_column][i] != split_value:
                continue
            if require_all_pass and not bool(cols["all_pass"][i]):
                continue
            text = cols[text_column][i]
            if text is None:
                continue
            text = str(text).strip()
            if not text:
                continue
            yield text


@dataclass
class ScanSummary:
    train_rows: int
    valid_rows_total: int
    valid_rows_sampled: int
    train_lengths: list[int]
    valid_lengths: list[int]
    train_hist: dict[str, int]
    valid_hist: dict[str, int]
    token_counter: Counter[str]
    valid_sequences: list[list[int]]


def _scan_cache_path(out_dir: Path) -> Path:
    return out_dir / "_scan_cache.pt"


def _save_scan_cache(out_dir: Path, summary: ScanSummary, vocab: TokenVocab) -> None:
    payload = {
        "summary": {
            "train_rows": summary.train_rows,
            "valid_rows_total": summary.valid_rows_total,
            "valid_rows_sampled": summary.valid_rows_sampled,
            "train_lengths": summary.train_lengths,
            "valid_lengths": summary.valid_lengths,
            "train_hist": summary.train_hist,
            "valid_hist": summary.valid_hist,
            "valid_sequences": summary.valid_sequences,
        },
        "vocab_itos": vocab.itos,
    }
    torch.save(payload, _scan_cache_path(out_dir))


def _load_scan_cache(out_dir: Path) -> tuple[ScanSummary, TokenVocab]:
    payload = torch.load(_scan_cache_path(out_dir), map_location="cpu")
    summary_payload = payload["summary"]
    vocab_itos = list(payload["vocab_itos"])
    vocab = TokenVocab(stoi={tok: i for i, tok in enumerate(vocab_itos)}, itos=vocab_itos)
    summary = ScanSummary(
        train_rows=int(summary_payload["train_rows"]),
        valid_rows_total=int(summary_payload["valid_rows_total"]),
        valid_rows_sampled=int(summary_payload["valid_rows_sampled"]),
        train_lengths=[int(x) for x in summary_payload["train_lengths"]],
        valid_lengths=[int(x) for x in summary_payload["valid_lengths"]],
        train_hist={str(k): int(v) for k, v in dict(summary_payload["train_hist"]).items()},
        valid_hist={str(k): int(v) for k, v in dict(summary_payload["valid_hist"]).items()},
        token_counter=Counter(),
        valid_sequences=[[int(y) for y in seq] for seq in summary_payload["valid_sequences"]],
    )
    return summary, vocab


def _scan_parquet_for_vocab_and_valid(
    parquet_path: str | Path,
    *,
    text_column: str,
    split_column: str,
    train_split: str,
    valid_split: str,
    seed_tokens: list[str],
    require_all_pass: bool,
    train_max_rows: Optional[int],
    valid_max_rows: int,
    seed: int,
    show_progress: bool,
) -> tuple[ScanSummary, TokenVocab]:
    counter: Counter[str] = Counter()
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
        desc="Scan parquet",
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
                counter.update(toks)
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

    vocab = _build_vocab_from_counter(counter, seed_tokens=seed_tokens)
    valid_sequences = [
        [vocab.bos_id] + vocab.encode_tokens(toks) + [vocab.eos_id]
        for toks in valid_sample_tokens
    ]
    summary = ScanSummary(
        train_rows=train_rows,
        valid_rows_total=valid_rows_total,
        valid_rows_sampled=len(valid_sequences),
        train_lengths=train_lengths,
        valid_lengths=valid_lengths[: len(valid_sample_tokens)],
        train_hist=_histogram(train_lengths),
        valid_hist=_histogram(valid_lengths[: len(valid_sample_tokens)]),
        token_counter=counter,
        valid_sequences=valid_sequences,
    )
    return summary, vocab


class _ParquetTokenPairDataset(IterableDataset[tuple[list[int], list[int]]]):
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

    def __iter__(self) -> Iterator[tuple[list[int], list[int]]]:
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
                x = ids[:-1][: self.max_seq_len]
                y = ids[1:][: self.max_seq_len]
                if not x:
                    continue
                yield x, y
                yielded += 1
                if worker_limit is not None and yielded >= worker_limit:
                    return


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Large-scale AO-GPT pretraining directly from parquet stamp sequences.")
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

    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--global-batch-size", type=int, default=0, help="0 = use batch-size. If > batch-size, enable gradient accumulation to keep this effective batch size.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-data-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "none"])
    p.add_argument("--torch-compile", action="store_true", default=True)
    p.add_argument("--no-torch-compile", dest="torch_compile", action="store_false")
    p.add_argument("--no-tf32", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-parallel", action="store_true", help="Use single-process torch.nn.DataParallel across all visible CUDA devices.")
    p.add_argument("--save-every-epoch", action="store_true", default=True)
    p.add_argument("--no-save-every-epoch", dest="save_every_epoch", action="store_false")
    p.add_argument(
        "--reuse-scan-cache",
        action="store_true",
        help="Reuse existing output_dir/_scan_cache.pt instead of rescanning parquet on rank 0.",
    )
    p.add_argument("--debug-phase-prints", action="store_true", help="Print rank/phase markers for debugging distributed hangs.")
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
        seed=args.seed,
        num_data_workers=args.num_data_workers,
        prefetch_factor=args.prefetch_factor,
        amp=args.amp,
        torch_compile=bool(args.torch_compile),
        tf32=not bool(args.no_tf32),
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
    _debug_print(debug, f"[debug rank={_dist_rank()}] device={dev} world_size={_dist_world_size()} local_rank={local_rank}")
    _setup_distributed(dev)
    _debug_print(debug, f"[debug rank={_dist_rank()}] distributed_initialized={dist.is_initialized()}")

    vocab_header, seed_tokens = _load_vocab_seed_tokens(stamp_vocab)
    train_max_rows = (None if int(args.train_max_rows) <= 0 else int(args.train_max_rows))
    if _is_main_process():
        cache_path = _scan_cache_path(out_dir)
        if bool(args.reuse_scan_cache) and cache_path.exists():
            _debug_print(debug, f"[debug rank={_dist_rank()}] reuse_scan_cache path={cache_path}")
            scan_summary, vocab = _load_scan_cache(out_dir)
        else:
            _debug_print(debug, f"[debug rank={_dist_rank()}] begin_scan")
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
            _debug_print(debug, f"[debug rank={_dist_rank()}] end_scan saved_cache")
        print(
            json.dumps(
                {
                    "train_rows": scan_summary.train_rows,
                    "valid_rows_total": scan_summary.valid_rows_total,
                    "valid_rows_sampled": scan_summary.valid_rows_sampled,
                    "vocab_size": len(vocab.itos),
                    "seed_vocab_size": len(seed_tokens),
                    "world_size": _dist_world_size(),
                    "reused_scan_cache": bool(args.reuse_scan_cache and cache_path.exists()),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    if not _is_main_process():
        cache_path = _scan_cache_path(out_dir)
        _debug_print(debug, f"[debug rank={_dist_rank()}] wait_cache path={cache_path}")
        while not cache_path.exists():
            time.sleep(0.5)
        _debug_print(debug, f"[debug rank={_dist_rank()}] load_cache")
        scan_summary, vocab = _load_scan_cache(out_dir)
        _debug_print(debug, f"[debug rank={_dist_rank()}] cache_loaded train_rows={scan_summary.train_rows}")

    if args.scan_only:
        _cleanup_distributed()
        return 0
    if scan_summary.train_rows <= 0:
        _cleanup_distributed()
        raise ValueError("No train rows found after parquet scan.")
    if not scan_summary.valid_sequences:
        _cleanup_distributed()
        raise ValueError("No validation rows sampled after parquet scan.")

    train_ds = _ParquetTokenPairDataset(
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
    valid_sequences_rank = scan_summary.valid_sequences[_dist_rank():: max(1, _dist_world_size())]
    valid_ds = NextTokenDataset(valid_sequences_rank, cfg.max_seq_len, vocab.pad_id)
    _debug_print(debug, f"[debug rank={_dist_rank()}] dataset_ready valid_local={len(valid_sequences_rank)}")

    collate_fn = build_dynamic_lm_collate_fn(vocab.pad_id, pad_to_multiple_of=(8 if dev.type == "cuda" else None))
    pin_memory = dev.type == "cuda"
    train_loader_kwargs: dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
        "num_workers": max(0, int(cfg.num_data_workers)),
    }
    if cfg.num_data_workers > 0:
        train_loader_kwargs["prefetch_factor"] = max(2, int(cfg.prefetch_factor))
        train_loader_kwargs["persistent_workers"] = True
    loader_train = DataLoader(train_ds, **train_loader_kwargs)
    valid_loader_kwargs: dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "drop_last": False,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
        "num_workers": max(0, int(cfg.num_data_workers)),
    }
    if cfg.num_data_workers > 0:
        valid_loader_kwargs["prefetch_factor"] = max(2, int(cfg.prefetch_factor))
        valid_loader_kwargs["persistent_workers"] = True
    loader_valid = DataLoader(valid_ds, **valid_loader_kwargs)

    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    use_amp, amp_dtype, use_scaler, _amp_label = _resolve_amp_mode(cfg.amp, dev)
    _debug_print(debug, f"[debug rank={_dist_rank()}] amp={args.amp} resolved_dtype={amp_dtype}")
    model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg)
    _debug_print(debug, f"[debug rank={_dist_rank()}] model_created")
    if cfg.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[misc]
        except Exception:
            pass
    model.to(dev)
    _debug_print(debug, f"[debug rank={_dist_rank()}] model_to_device_done")
    if _is_dist():
        _debug_print(debug, f"[debug rank={_dist_rank()}] before_ddp_wrap")
        model = DDP(
            model,
            device_ids=[dev.index] if dev.type == "cuda" and dev.index is not None else None,
            output_device=(dev.index if dev.type == "cuda" else None),
        )
    elif bool(args.data_parallel):
        if dev.type != "cuda" or torch.cuda.device_count() <= 1:
            raise ValueError("--data-parallel requires multiple visible CUDA devices.")
        model = DataParallel(model)
    _debug_print(debug, f"[debug rank={_dist_rank()}] model_wrapped")

    adamw_kwargs: dict[str, Any] = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    if dev.type == "cuda":
        try:
            adamw_kwargs["fused"] = True
        except Exception:
            pass
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    except TypeError:
        adamw_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)  # pragma: no cover

    history: list[dict[str, float]] = []
    non_blocking = dev.type == "cuda"
    dp_world_size = torch.cuda.device_count() if bool(args.data_parallel) and dev.type == "cuda" else 1
    effective_global_batch = int(args.global_batch_size) if int(args.global_batch_size) > 0 else int(cfg.batch_size)
    min_global_batch = int(cfg.batch_size) * max(1, _dist_world_size())
    if effective_global_batch < min_global_batch:
        raise ValueError("global_batch_size must be >= batch_size when provided.")
    grad_accum_steps = max(1, math.ceil(effective_global_batch / max(1, int(cfg.batch_size) * max(1, _dist_world_size()))))
    if _is_dist():
        effective_global_batch = int(cfg.batch_size) * max(1, _dist_world_size()) * grad_accum_steps
        micro_batch_report = int(cfg.batch_size)
    elif bool(args.data_parallel):
        effective_global_batch = int(cfg.batch_size) * grad_accum_steps
        micro_batch_report = max(1, math.ceil(int(cfg.batch_size) / max(1, dp_world_size)))
    else:
        effective_global_batch = int(cfg.batch_size) * grad_accum_steps
        micro_batch_report = int(cfg.batch_size)
    train_rows_rank = math.ceil(scan_summary.train_rows / max(1, _dist_world_size()))
    train_steps = max(1, math.ceil(train_rows_rank / max(1, cfg.batch_size)))
    train_opt_steps = max(1, math.ceil(train_rows_rank / max(1, int(cfg.batch_size) * grad_accum_steps)))
    ckpt_dir = out_dir / "checkpoints"
    if args.save_every_epoch and _is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    if _is_main_process():
        shutil.copy2(stamp_vocab, out_dir / "stamp_vocab.txt")
        print(
            json.dumps(
                {
                    "world_size": int(_dist_world_size()),
                    "data_parallel_world_size": int(dp_world_size),
                    "micro_batch_size_per_rank": int(micro_batch_report),
                    "global_batch_size": int(effective_global_batch),
                    "grad_accum_steps": int(grad_accum_steps),
                    "train_micro_steps_per_epoch_per_rank": int(train_steps),
                    "train_optimizer_steps_per_epoch_per_rank": int(train_opt_steps),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    for epoch in range(1, cfg.epochs + 1):
        _debug_print(debug, f"[debug rank={_dist_rank()}] epoch={epoch} train_start")
        epoch_t0 = time.time()
        train_ds.set_epoch(epoch)
        model.train()
        train_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        iterator = iterable_with_progress(
            loader_train,
            show_progress=(not args.no_tqdm) and _is_main_process(),
            total=train_steps,
            desc=f"Train ao-pretrain[stamp] ep{epoch}",
        )
        micro_step = 0
        for x, y in iterator:
            micro_step += 1
            x = x.to(dev, non_blocking=non_blocking)
            y = y.to(dev, non_blocking=non_blocking)
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True) if use_amp else nullcontext()
            with amp_ctx:
                loss = _compute_ao_gpt_loss(model, x, y, cfg, random_ratio=0.9)
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
        _debug_print(debug, f"[debug rank={_dist_rank()}] epoch={epoch} train_done n_losses={len(train_losses)}")

        train_loss_mean = float(np.mean(train_losses)) if train_losses else float("nan")
        train_loss_mean = _distributed_mean(train_loss_mean, device=dev)

        model.eval()
        valid_losses: list[float] = []
        _debug_print(debug, f"[debug rank={_dist_rank()}] epoch={epoch} valid_start")
        with torch.no_grad():
            for x, y in loader_valid:
                x = x.to(dev, non_blocking=non_blocking)
                y = y.to(dev, non_blocking=non_blocking)
                amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True) if use_amp else nullcontext()
                with amp_ctx:
                    loss = _compute_ao_gpt_loss(model, x, y, cfg, random_ratio=0.9)
                valid_losses.append(float(loss.item()))
        _debug_print(debug, f"[debug rank={_dist_rank()}] epoch={epoch} valid_done n_losses={len(valid_losses)}")

        tr = train_loss_mean
        va = float(np.mean(valid_losses)) if valid_losses else float("nan")
        va = _distributed_mean(va, device=dev)
        if not _is_main_process():
            continue
        epoch_wall_seconds = float(time.time() - epoch_t0)
        train_rows_per_sec = float(scan_summary.train_rows) / max(1e-9, epoch_wall_seconds)
        optimizer_steps_per_sec = float(train_opt_steps) / max(1e-9, epoch_wall_seconds)
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr,
                "valid_loss": va,
                "train_ppl": float(math.exp(tr)) if not math.isnan(tr) else float("nan"),
                "valid_ppl": float(math.exp(va)) if not math.isnan(va) else float("nan"),
                "epoch_wall_seconds": epoch_wall_seconds,
                "train_rows_per_sec": train_rows_per_sec,
                "optimizer_steps_per_sec": optimizer_steps_per_sec,
            }
        )
        print(
            f"[train-ao-gpt-parquet] epoch={epoch} train_loss={tr:.4f} valid_loss={va:.4f} "
            f"train_ppl={history[-1]['train_ppl']:.2f} valid_ppl={history[-1]['valid_ppl']:.2f} "
            f"epoch_wall_s={epoch_wall_seconds:.1f} train_rows_per_s={train_rows_per_sec:.1f} "
            f"opt_steps_per_s={optimizer_steps_per_sec:.3f}",
            flush=True,
        )
        (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        model_state = _unwrap_model(model).state_dict()
        checkpoint_payload = {
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": (scaler.state_dict() if scaler is not None else None),
            "cfg": vars(cfg),
            "representation": "stamp",
            "model_type": "ao_gpt",
            "stamp_vocab": "stamp_vocab.txt",
            "epoch": int(epoch),
            "history": history,
            "world_size": int(_dist_world_size()),
        }
        torch.save(checkpoint_payload, out_dir / "model.pt")
        if args.save_every_epoch:
            torch.save(checkpoint_payload, ckpt_dir / f"epoch_{epoch:04d}.pt")
        _debug_print(debug, f"[debug rank={_dist_rank()}] epoch={epoch} checkpoint_written")
    if _is_main_process():
        (out_dir / "vocab.json").write_text(json.dumps(vocab.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        param_iter = _unwrap_model(model).parameters()
        meta = {
            "representation": "stamp",
            "train_size": scan_summary.train_rows,
            "valid_size": scan_summary.valid_rows_total,
            "n_train_encoded": scan_summary.train_rows,
            "n_valid_encoded": scan_summary.valid_rows_sampled,
            "vocab_size": len(vocab.itos),
            "model_type": "ao_gpt",
            "cfg": vars(cfg),
            "device": str(dev),
            "world_size": int(_dist_world_size()),
            "data_parallel_world_size": int(dp_world_size),
            "n_model_params": int(sum(p.numel() for p in param_iter)),
            "micro_batch_size_per_rank": int(micro_batch_report),
            "global_batch_size": int(effective_global_batch),
            "grad_accum_steps": int(grad_accum_steps),
            "train_optimizer_steps_per_epoch_per_rank": int(train_opt_steps),
            "padding_mode": "dynamic_batch_max",
            "train_effective_max_seq_len": max(scan_summary.train_lengths) if scan_summary.train_lengths else 0,
            "valid_effective_max_seq_len": max(scan_summary.valid_lengths) if scan_summary.valid_lengths else 0,
            "train_token_length_stats": _summarize_lengths(scan_summary.train_lengths),
            "valid_token_length_stats": _summarize_lengths(scan_summary.valid_lengths),
            "train_token_length_histogram": scan_summary.train_hist,
            "valid_token_length_histogram": scan_summary.valid_hist,
            "vocab_fit_size": scan_summary.train_rows,
            "n_vocab_fit_encoded": scan_summary.train_rows,
            "tokenize_workers": 0,
            "tokenize_chunksize": 0,
            "stamp_vocab": "stamp_vocab.txt",
            "seed_vocab_path": str(stamp_vocab),
            "seed_vocab_size": len(seed_tokens),
            "seed_vocab_header": vocab_header,
            "parquet_path": str(parquet_path),
            "text_column": args.text_column,
            "split_column": args.split_column,
            "train_split": args.train_split,
            "valid_split": args.valid_split,
            "require_all_pass": bool(args.require_all_pass),
            "save_every_epoch": bool(args.save_every_epoch),
        }
        (out_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output_dir": str(out_dir), "history": history, "device": str(dev)}, ensure_ascii=False, indent=2))
        _debug_print(debug, f"[debug rank={_dist_rank()}] main_finalize_done")
    _cleanup_distributed()
    _debug_print(debug, f"[debug rank={_dist_rank()}] cleanup_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
