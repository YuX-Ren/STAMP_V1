#!/usr/bin/env python3
"""Pretrain MolT5 on the original-style C4 + ZINC SMILES denoising mix.

The MolT5 paper pretrains T5 on a 50/50 mixture of C4 text and 100M ZINC
SMILES. This script keeps that split explicit: every optimization micro-batch
is assembled from separate C4 and ZINC dataloaders before T5 span corruption.
"""
from __future__ import annotations

import argparse
import gzip
import glob
import json
import math
import os
import random
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator, Sequence

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from transformers import AddedToken

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.hybrid_vocab import FCSTAMP_CONTROL_TOKENS  # noqa: E402


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


def _setup_dist(device: torch.device) -> None:
    if not _is_dist() or dist.is_initialized():
        return
    backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(backend=backend)


def _cleanup_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _all_reduce_sums(values: Sequence[float], device: torch.device) -> list[float]:
    tensor = torch.tensor([float(x) for x in values], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return [float(x) for x in tensor.detach().cpu().tolist()]


def _atomic_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix=".tmp", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _resolve_paths(spec: str) -> list[str]:
    paths: list[str] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        matches = sorted(glob.glob(part))
        paths.extend(matches if matches else [part])
    out = [str(Path(p).resolve()) for p in paths if Path(p).exists()]
    return sorted(dict.fromkeys(out))


def _read_stamp_vocab_tokens(path: str | Path, limit: int = 0) -> list[str]:
    tokens: list[str] = []
    if not path:
        return tokens
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            raw = line.rstrip("\n")
            if not raw or raw.startswith("#"):
                continue
            if line_no == 0:
                try:
                    json.loads(raw)
                    continue
                except json.JSONDecodeError:
                    pass
            parts = raw.split("\t")
            tok = parts[0] if parts else raw
            if tok:
                tokens.append(tok)
                if limit > 0 and len(tokens) >= limit:
                    break
    return tokens


def _stamp_structural_tokens(max_index: int = 255) -> list[str]:
    toks = ["[BOS]", "[EOS]", "[CONCAT]", "[END]", "[BACK]", "[Ring]", ".", "[S_R]", "[S_S]", "[S_E]", "[S_Z]"]
    toks.extend(f"[J_{i}]" for i in range(max_index + 1))
    toks.extend(f"[B_{i}]" for i in range(max_index + 1))
    toks.extend(FCSTAMP_CONTROL_TOKENS)
    return toks


def _add_stamp_tokens(tokenizer: Any, *, stamp_vocab: str, token_limit: int, structural_max_index: int) -> int:
    tokens = _stamp_structural_tokens(structural_max_index) + _read_stamp_vocab_tokens(stamp_vocab, token_limit)
    seen: set[str] = set()
    add: list[AddedToken] = []
    for tok in tokens:
        if tok in seen:
            continue
        seen.add(tok)
        if tokenizer.convert_tokens_to_ids(tok) != tokenizer.unk_token_id:
            continue
        add.append(AddedToken(tok, single_word=False, lstrip=False, rstrip=False, normalized=False, special=False))
    if not add:
        return 0
    return int(tokenizer.add_tokens(add, special_tokens=False))


def _worker_shard() -> tuple[int, int]:
    worker = torch.utils.data.get_worker_info()
    worker_id = 0 if worker is None else int(worker.id)
    num_workers = 1 if worker is None else int(worker.num_workers)
    shard_id = _dist_rank() * num_workers + worker_id
    num_shards = _dist_world_size() * num_workers
    return shard_id, num_shards


def _worker_limit(total_limit: int, shard_id: int, num_shards: int) -> int | None:
    if total_limit <= 0:
        return None
    base = total_limit // max(1, num_shards)
    rem = total_limit % max(1, num_shards)
    return base + (1 if shard_id < rem else 0)


def _random_noise_mask(length: int, noise_density: float, mean_span_length: float, rng: random.Random) -> list[bool]:
    if length <= 1:
        return [False] * max(0, length)
    n_noise = int(round(float(length) * float(noise_density)))
    n_noise = max(1, min(length - 1, n_noise))
    max_span = max(1, int(round(float(mean_span_length) * 2)))
    mask = [False] * length
    remaining = n_noise
    attempts = 0
    while remaining > 0 and attempts < length * 20:
        attempts += 1
        start = rng.randrange(0, length)
        if mask[start]:
            continue
        span_len = max(1, min(remaining, rng.randint(1, max_span)))
        for pos in range(start, min(length, start + span_len)):
            if mask[pos]:
                continue
            mask[pos] = True
            remaining -= 1
            if remaining <= 0:
                break
    if remaining > 0:
        positions = [i for i, v in enumerate(mask) if not v]
        rng.shuffle(positions)
        for pos in positions[:remaining]:
            mask[pos] = True
    return mask


def _sentinel_id(tokenizer: Any, index: int) -> int:
    token = f"<extra_id_{min(index, 99)}>"
    sid = tokenizer.convert_tokens_to_ids(token)
    if sid is None or sid == tokenizer.unk_token_id:
        raise ValueError(f"Tokenizer does not provide required sentinel token {token!r}.")
    return int(sid)


def _span_corrupt(
    ids: Sequence[int],
    *,
    tokenizer: Any,
    rng: random.Random,
    noise_density: float,
    mean_span_length: float,
    max_source_len: int,
    max_target_len: int,
) -> tuple[list[int], list[int]]:
    body = [int(x) for x in ids if int(x) >= 0]
    if len(body) > max(2, max_source_len - 1):
        start = rng.randrange(0, len(body) - (max_source_len - 1) + 1)
        body = body[start : start + max_source_len - 1]
    mask = _random_noise_mask(len(body), noise_density, mean_span_length, rng)

    source: list[int] = []
    target: list[int] = []
    span_idx = 0
    i = 0
    while i < len(body):
        if mask[i]:
            sentinel = _sentinel_id(tokenizer, span_idx)
            source.append(sentinel)
            target.append(sentinel)
            while i < len(body) and mask[i]:
                target.append(body[i])
                i += 1
            span_idx += 1
        else:
            source.append(body[i])
            i += 1
    source.append(int(tokenizer.eos_token_id))
    target.append(int(tokenizer.eos_token_id))
    source = source[:max_source_len]
    target = target[:max_target_len]
    source[-1] = int(tokenizer.eos_token_id)
    target[-1] = int(tokenizer.eos_token_id)
    return source, target


class _DenoiseMixin:
    tokenizer: Any
    seed: int
    epoch: int
    max_source_len: int
    max_target_len: int
    noise_density: float
    mean_span_length: float
    max_raw_chars: int

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _make_example(self, text: str, rng: random.Random) -> dict[str, list[int]] | None:
        text = str(text).strip()
        if not text:
            return None
        if self.max_raw_chars > 0 and len(text) > self.max_raw_chars:
            start = rng.randrange(0, len(text) - self.max_raw_chars + 1)
            text = text[start : start + self.max_raw_chars]
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 2:
            return None
        source, target = _span_corrupt(
            ids,
            tokenizer=self.tokenizer,
            rng=rng,
            noise_density=self.noise_density,
            mean_span_length=self.mean_span_length,
            max_source_len=self.max_source_len,
            max_target_len=self.max_target_len,
        )
        return {"input_ids": source, "labels": target}


class C4JsonlDataset(_DenoiseMixin, IterableDataset):
    def __init__(
        self,
        paths: Sequence[str],
        *,
        tokenizer: Any,
        field: str,
        max_rows: int,
        seed: int,
        max_source_len: int,
        max_target_len: int,
        noise_density: float,
        mean_span_length: float,
        max_raw_chars: int,
    ) -> None:
        super().__init__()
        self.paths = list(paths)
        self.tokenizer = tokenizer
        self.field = str(field)
        self.max_rows = int(max_rows)
        self.seed = int(seed)
        self.epoch = 0
        self.max_source_len = int(max_source_len)
        self.max_target_len = int(max_target_len)
        self.noise_density = float(noise_density)
        self.mean_span_length = float(mean_span_length)
        self.max_raw_chars = int(max_raw_chars)

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        shard_id, num_shards = _worker_shard()
        rng = random.Random(self.seed + self.epoch * 1009 + shard_id)
        paths = list(self.paths)
        rng.shuffle(paths)
        shard_by_line = len(paths) < num_shards
        if not shard_by_line:
            paths = paths[shard_id::num_shards]
        limit = _worker_limit(self.max_rows, shard_id, num_shards)
        yielded = 0
        line_idx = 0
        for path in paths:
            opener = gzip.open if str(path).endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if shard_by_line:
                        current_idx = line_idx
                        line_idx += 1
                        if current_idx % max(1, num_shards) != shard_id:
                            continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = row.get(self.field)
                    if text is None:
                        continue
                    ex = self._make_example(str(text), rng)
                    if ex is None:
                        continue
                    yield ex
                    yielded += 1
                    if limit is not None and yielded >= limit:
                        return


class C4StreamingDataset(_DenoiseMixin, IterableDataset):
    def __init__(
        self,
        *,
        dataset_name: str,
        config_name: str,
        split: str,
        tokenizer: Any,
        field: str,
        max_rows: int,
        seed: int,
        max_source_len: int,
        max_target_len: int,
        noise_density: float,
        mean_span_length: float,
        max_raw_chars: int,
        shuffle_buffer: int,
        shard_by_row: bool,
    ) -> None:
        super().__init__()
        self.dataset_name = str(dataset_name)
        self.config_name = str(config_name)
        self.split = str(split)
        self.tokenizer = tokenizer
        self.field = str(field)
        self.max_rows = int(max_rows)
        self.seed = int(seed)
        self.epoch = 0
        self.max_source_len = int(max_source_len)
        self.max_target_len = int(max_target_len)
        self.noise_density = float(noise_density)
        self.mean_span_length = float(mean_span_length)
        self.max_raw_chars = int(max_raw_chars)
        self.shuffle_buffer = int(shuffle_buffer)
        self.shard_by_row = bool(shard_by_row)

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        from datasets import load_dataset

        shard_id, num_shards = _worker_shard()
        rng = random.Random(self.seed + self.epoch * 1009 + shard_id)
        ds = load_dataset(self.dataset_name, self.config_name, split=self.split, streaming=True)
        if self.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed + self.epoch)
        if not self.shard_by_row:
            ds = ds.shard(num_shards=max(1, num_shards), index=shard_id)
        limit = _worker_limit(self.max_rows, shard_id, num_shards)
        yielded = 0
        row_idx = 0
        for row in ds:
            if self.shard_by_row:
                current_idx = row_idx
                row_idx += 1
                if current_idx % max(1, num_shards) != shard_id:
                    continue
            text = row.get(self.field)
            if text is None:
                continue
            ex = self._make_example(str(text), rng)
            if ex is None:
                continue
            yield ex
            yielded += 1
            if limit is not None and yielded >= limit:
                return


class ZincParquetDataset(_DenoiseMixin, IterableDataset):
    def __init__(
        self,
        paths: Sequence[str],
        *,
        tokenizer: Any,
        column: str,
        max_rows: int,
        seed: int,
        max_source_len: int,
        max_target_len: int,
        noise_density: float,
        mean_span_length: float,
        max_raw_chars: int,
        pretokenized: bool,
    ) -> None:
        super().__init__()
        self.paths = list(paths)
        self.tokenizer = tokenizer
        self.column = str(column)
        self.max_rows = int(max_rows)
        self.seed = int(seed)
        self.epoch = 0
        self.max_source_len = int(max_source_len)
        self.max_target_len = int(max_target_len)
        self.noise_density = float(noise_density)
        self.mean_span_length = float(mean_span_length)
        self.max_raw_chars = int(max_raw_chars)
        self.pretokenized = bool(pretokenized)

    def _make_zinc_example(self, text: str, rng: random.Random) -> dict[str, list[int]] | None:
        text = str(text).strip()
        if not text:
            return None
        if not self.pretokenized:
            return self._make_example(text, rng)
        ids: list[int] = []
        for tok in text.split():
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is None or tid == self.tokenizer.unk_token_id:
                ids.extend(self.tokenizer.encode(tok, add_special_tokens=False))
            else:
                ids.append(int(tid))
        if len(ids) < 2:
            return None
        source, target = _span_corrupt(
            ids,
            tokenizer=self.tokenizer,
            rng=rng,
            noise_density=self.noise_density,
            mean_span_length=self.mean_span_length,
            max_source_len=self.max_source_len,
            max_target_len=self.max_target_len,
        )
        return {"input_ids": source, "labels": target}

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        shard_id, num_shards = _worker_shard()
        row_groups: list[tuple[str, int]] = []
        for path in self.paths:
            pf = pq.ParquetFile(path)
            row_groups.extend((path, rg) for rg in range(pf.num_row_groups))
        rng = random.Random(self.seed + self.epoch * 1009 + shard_id)
        rng.shuffle(row_groups)
        row_groups = row_groups[shard_id::num_shards]
        limit = _worker_limit(self.max_rows, shard_id, num_shards)
        yielded = 0
        for path, rg in row_groups:
            pf = pq.ParquetFile(path)
            names = set(pf.schema_arrow.names)
            if self.column not in names:
                raise KeyError(f"{path} missing column {self.column!r}; available={sorted(names)}")
            table = pf.read_row_group(rg, columns=[self.column])
            values = table.column(self.column).to_pylist()
            order = list(range(len(values)))
            rng.shuffle(order)
            for idx in order:
                text = values[idx]
                if text is None:
                    continue
                ex = self._make_zinc_example(str(text), rng)
                if ex is None:
                    continue
                yield ex
                yielded += 1
                if limit is not None and yielded >= limit:
                    return


def _collate_examples(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    max_in = max(len(x["input_ids"]) for x in batch)
    max_lab = max(len(x["labels"]) for x in batch)
    input_ids = torch.full((len(batch), max_in), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_in), dtype=torch.long)
    labels = torch.full((len(batch), max_lab), -100, dtype=torch.long)
    for i, row in enumerate(batch):
        src = torch.tensor(row["input_ids"], dtype=torch.long)
        tgt = torch.tensor(row["labels"], dtype=torch.long)
        input_ids[i, : len(src)] = src
        attention_mask[i, : len(src)] = 1
        labels[i, : len(tgt)] = tgt
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _cycle(loader: DataLoader, dataset: Any) -> Iterator[list[dict[str, list[int]]]]:
    epoch = 0
    while True:
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        produced = False
        for batch in loader:
            produced = True
            yield batch
        if not produced:
            raise RuntimeError("DataLoader produced no examples.")
        epoch += 1


def _autocast(device: torch.device, amp: str):
    mode = str(amp).lower()
    if device.type != "cuda" or mode == "none":
        return nullcontext()
    dtype = torch.bfloat16 if mode in {"auto", "bf16"} else torch.float16
    return torch.amp.autocast("cuda", dtype=dtype)


def _make_loaders(args: argparse.Namespace, tokenizer: Any, *, train: bool) -> tuple[DataLoader, DataLoader, Any, Any]:
    if train:
        zinc_paths = _resolve_paths(args.zinc_train_parquet)
        c4_rows = int(args.c4_train_max_rows)
        zinc_rows = int(args.zinc_train_max_rows)
        seed = int(args.seed)
        c4_split = str(args.c4_train_split)
        c4_glob = str(args.c4_train_glob)
    else:
        zinc_paths = _resolve_paths(args.zinc_valid_parquet)
        c4_rows = int(args.c4_valid_max_rows)
        zinc_rows = int(args.zinc_valid_max_rows)
        seed = int(args.seed) + 999
        c4_split = str(args.c4_valid_split)
        c4_glob = str(args.c4_valid_glob)
    if not zinc_paths:
        raise FileNotFoundError(args.zinc_train_parquet if train else args.zinc_valid_parquet)

    if bool(args.c4_streaming):
        c4_dataset = C4StreamingDataset(
            dataset_name=args.c4_dataset_name,
            config_name=args.c4_config_name,
            split=c4_split,
            tokenizer=tokenizer,
            field=args.c4_text_field,
            max_rows=c4_rows,
            seed=seed,
            max_source_len=int(args.max_source_len),
            max_target_len=int(args.max_target_len),
            noise_density=float(args.noise_density),
            mean_span_length=float(args.mean_noise_span_length),
            max_raw_chars=int(args.max_c4_chars),
            shuffle_buffer=int(args.c4_stream_shuffle_buffer),
            shard_by_row=not train,
        )
    else:
        c4_paths = _resolve_paths(c4_glob)
        if not c4_paths:
            raise FileNotFoundError(c4_glob)
        c4_dataset = C4JsonlDataset(
            c4_paths,
            tokenizer=tokenizer,
            field=args.c4_text_field,
            max_rows=c4_rows,
            seed=seed,
            max_source_len=int(args.max_source_len),
            max_target_len=int(args.max_target_len),
            noise_density=float(args.noise_density),
            mean_span_length=float(args.mean_noise_span_length),
            max_raw_chars=int(args.max_c4_chars),
        )
    zinc_dataset = ZincParquetDataset(
        zinc_paths,
        tokenizer=tokenizer,
        column=args.zinc_column,
        max_rows=zinc_rows,
        seed=seed + 131,
        max_source_len=int(args.max_source_len),
        max_target_len=int(args.max_target_len),
        noise_density=float(args.noise_density),
        mean_span_length=float(args.mean_noise_span_length),
        max_raw_chars=int(args.max_smiles_chars),
        pretokenized=str(args.molecule_format).lower() == "stamp",
    )

    c4_bs = max(1, int(args.batch_size) // 2)
    zinc_bs = max(1, int(args.batch_size) - c4_bs)
    if int(args.batch_size) < 2:
        raise ValueError("--batch-size must be >=2 so each micro-batch contains C4 and ZINC examples.")
    kwargs = {
        "num_workers": int(args.num_data_workers),
        "collate_fn": lambda b: b,
        "pin_memory": torch.cuda.is_available(),
    }
    if int(args.num_data_workers) > 0:
        kwargs["prefetch_factor"] = 2
    c4_loader = DataLoader(c4_dataset, batch_size=c4_bs, **kwargs)
    zinc_loader = DataLoader(zinc_dataset, batch_size=zinc_bs, **kwargs)
    return c4_loader, zinc_loader, c4_dataset, zinc_dataset


def _make_model(args: argparse.Namespace):
    from transformers import T5Config, T5ForConditionalGeneration

    if args.resume_from:
        return T5ForConditionalGeneration.from_pretrained(args.resume_from, local_files_only=bool(args.local_files_only))
    if args.init_model:
        return T5ForConditionalGeneration.from_pretrained(args.init_model, local_files_only=bool(args.local_files_only))
    cfg = T5Config(
        vocab_size=int(args.vocab_size),
        d_model=int(args.d_model),
        d_kv=max(1, int(args.d_model) // int(args.n_heads)),
        d_ff=int(args.d_ff),
        num_layers=int(args.num_layers),
        num_decoder_layers=int(args.num_decoder_layers),
        num_heads=int(args.n_heads),
        dropout_rate=float(args.dropout),
        layer_norm_epsilon=1e-6,
        feed_forward_proj=str(args.feed_forward_proj),
        pad_token_id=int(args.pad_token_id),
        eos_token_id=int(args.eos_token_id),
        decoder_start_token_id=int(args.pad_token_id),
    )
    return T5ForConditionalGeneration(cfg)


def _save_checkpoint(
    *,
    ckpt_dir: Path,
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    args: argparse.Namespace,
) -> None:
    raw = model.module if hasattr(model, "module") else model
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    raw.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    _atomic_save(
        {
            "step": int(step),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "args": vars(args),
        },
        ckpt_dir / "training_state.pt",
    )


def _load_training_state(
    *,
    path: str,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> int:
    state_path = Path(path) / "training_state.pt"
    if not state_path.exists():
        return 0
    state = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    return int(state.get("step", 0))


@torch.no_grad()
def _evaluate(
    *,
    model: torch.nn.Module,
    c4_loader: DataLoader,
    zinc_loader: DataLoader,
    c4_dataset: Any,
    zinc_dataset: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
    step: int,
) -> dict[str, Any]:
    if hasattr(c4_dataset, "set_epoch"):
        c4_dataset.set_epoch(10_000 + int(step))
    if hasattr(zinc_dataset, "set_epoch"):
        zinc_dataset.set_epoch(10_000 + int(step))
    c4_batches = _cycle(c4_loader, c4_dataset)
    zinc_batches = _cycle(zinc_loader, zinc_dataset)
    model.eval()
    c4_loss = 0.0
    zinc_loss = 0.0
    c4_count = 0
    zinc_count = 0
    max_batches = int(args.eval_max_batches)
    for batch_idx in range(max_batches):
        for name, source_iter in (("c4", c4_batches), ("zinc", zinc_batches)):
            examples = next(source_iter)
            batch = _collate_examples(examples, int(tokenizer.pad_token_id))
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with _autocast(device, args.amp):
                out = model(**batch)
            loss = float(out.loss.detach().item())
            if name == "c4":
                c4_loss += loss
                c4_count += 1
            else:
                zinc_loss += loss
                zinc_count += 1
    sums = _all_reduce_sums([c4_loss, c4_count, zinc_loss, zinc_count], device)
    c4_loss_total, c4_count_total, zinc_loss_total, zinc_count_total = sums
    c4_mean = c4_loss_total / max(1.0, c4_count_total)
    zinc_mean = zinc_loss_total / max(1.0, zinc_count_total)
    mixed = 0.5 * (c4_mean + zinc_mean)
    model.train()
    return {
        "step": int(step),
        "valid_loss": float(mixed),
        "valid_ppl": float(math.exp(min(20.0, mixed))),
        "c4_valid_loss": float(c4_mean),
        "zinc_valid_loss": float(zinc_mean),
        "c4_valid_ppl": float(math.exp(min(20.0, c4_mean))),
        "zinc_valid_ppl": float(math.exp(min(20.0, zinc_mean))),
        "valid_batches_per_source": int(max_batches),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c4-train-glob", default="")
    p.add_argument("--c4-valid-glob", default="")
    p.add_argument("--c4-streaming", action="store_true", help="Stream C4 through datasets instead of reading local json.gz shards.")
    p.add_argument("--c4-dataset-name", default="allenai/c4")
    p.add_argument("--c4-config-name", default="en")
    p.add_argument("--c4-train-split", default="train")
    p.add_argument("--c4-valid-split", default="validation")
    p.add_argument("--c4-stream-shuffle-buffer", type=int, default=10_000)
    p.add_argument("--zinc-train-parquet", required=True)
    p.add_argument("--zinc-valid-parquet", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tokenizer", default="/scratch/yuxuan.ren/models/t5-small")
    p.add_argument("--init-model", default="/scratch/yuxuan.ren/models/t5-small")
    p.add_argument("--local-files-only", action="store_true", default=True)
    p.add_argument("--c4-text-field", default="text")
    p.add_argument("--zinc-column", default="smiles")
    p.add_argument("--molecule-format", choices=["smiles", "stamp"], default="smiles")
    p.add_argument("--stamp-vocab", default="/Project_Storage/CFP-03/CFP03-CF-130/data/zinc_pretrain/vocab_merged/motif_vocab.txt")
    p.add_argument("--stamp-token-limit", type=int, default=0, help="0 means add all motif tokens from --stamp-vocab.")
    p.add_argument("--stamp-structural-max-index", type=int, default=255)
    p.add_argument("--c4-train-max-rows", type=int, default=0)
    p.add_argument("--zinc-train-max-rows", type=int, default=100_000_000)
    p.add_argument("--c4-valid-max-rows", type=int, default=20_000)
    p.add_argument("--zinc-valid-max-rows", type=int, default=20_000)
    p.add_argument("--max-c4-chars", type=int, default=4096)
    p.add_argument("--max-smiles-chars", type=int, default=1024)

    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-decoder-layers", type=int, default=12)
    p.add_argument("--d-ff", type=int, default=3072)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--feed-forward-proj", default="gated-gelu")
    p.add_argument("--vocab-size", type=int, default=32128)
    p.add_argument("--pad-token-id", type=int, default=0)
    p.add_argument("--eos-token-id", type=int, default=1)

    p.add_argument("--max-source-len", type=int, default=512)
    p.add_argument("--max-target-len", type=int, default=128)
    p.add_argument("--noise-density", type=float, default=0.15)
    p.add_argument("--mean-noise-span-length", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=32, help="Per-GPU micro batch; split 50/50 C4/ZINC.")
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=1_000_000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=10_000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every-steps", type=int, default=10_000)
    p.add_argument("--save-every-steps", type=int, default=10_000)
    p.add_argument("--keep-last-checkpoints", type=int, default=10)
    p.add_argument("--eval-max-batches", type=int, default=100)
    p.add_argument("--num-data-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="bf16")
    p.add_argument("--no-tf32", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume-from", default="")
    return p


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    if torch.cuda.is_available() and not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    local_rank = _dist_local_rank()
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if _is_dist() else args.device)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    _setup_dist(device)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=bool(args.local_files_only))
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("T5 tokenizer must define pad/eos token ids.")
    # We truncate after span corruption, so tokenizer-level max length warnings are noise here.
    tokenizer.model_max_length = max(int(getattr(tokenizer, "model_max_length", 0) or 0), 1_000_000)
    added_stamp_tokens = 0
    if str(args.molecule_format).lower() == "stamp":
        added_stamp_tokens = _add_stamp_tokens(
            tokenizer,
            stamp_vocab=args.stamp_vocab,
            token_limit=int(args.stamp_token_limit),
            structural_max_index=int(args.stamp_structural_max_index),
        )
    if _sentinel_id(tokenizer, 0) < 0:
        raise ValueError("T5 tokenizer sentinel token check failed.")

    out_dir = Path(args.output_dir)
    if _is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(out_dir)
        (out_dir / "train_args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
        c4_desc = (
            f"stream:{args.c4_dataset_name}/{args.c4_config_name}"
            if bool(args.c4_streaming)
            else f"files={len(_resolve_paths(args.c4_train_glob))}"
        )
        print(
            "[molt5-mix] "
            f"c4_train={c4_desc} "
            f"zinc_train_files={len(_resolve_paths(args.zinc_train_parquet))} "
            f"molecule_format={args.molecule_format} zinc_column={args.zinc_column} "
            f"tokenizer_vocab={tokenizer.vocab_size} tokenizer_len={len(tokenizer)} "
            f"added_stamp_tokens={added_stamp_tokens} batch={args.batch_size} global_batch={args.global_batch_size}",
            flush=True,
        )
    _barrier()

    model = _make_model(args)
    if len(tokenizer) != int(model.get_input_embeddings().num_embeddings):
        model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    def lr_lambda(step: int) -> float:
        step = max(1, int(step))
        warmup = max(1, int(args.warmup_steps))
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, int(args.max_steps) - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    global_step = 0
    if args.resume_from:
        global_step = _load_training_state(path=args.resume_from, optimizer=optimizer, scheduler=scheduler)

    if _is_dist():
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    c4_loader, zinc_loader, c4_dataset, zinc_dataset = _make_loaders(args, tokenizer, train=True)
    c4_valid_loader, zinc_valid_loader, c4_valid_dataset, zinc_valid_dataset = _make_loaders(args, tokenizer, train=False)
    c4_batches = _cycle(c4_loader, c4_dataset)
    zinc_batches = _cycle(zinc_loader, zinc_dataset)

    per_step_global = int(args.batch_size) * max(1, _dist_world_size())
    accum_steps = max(1, int(math.ceil(int(args.global_batch_size) / max(1, per_step_global))))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp == "fp16"))
    model.train()

    history: list[dict[str, Any]] = []
    t0 = time.time()
    running_loss = 0.0
    running_tokens = 0
    optimizer.zero_grad(set_to_none=True)
    while global_step < int(args.max_steps):
        step_loss = 0.0
        # Bucket-load: c4 and zinc go through separate forward/backward passes so
        # each source pads only to its own batch max, not the joint max. Two
        # backward calls per micro-step; each loss is scaled by 1/(accum_steps*2)
        # so the accumulated gradient equals the mean of one "combined micro-batch"
        # per accum unit, matching the semantics of the old joint-collate path.
        divisor = accum_steps * 2
        for _ in range(accum_steps):
            for source_iter in (c4_batches, zinc_batches):
                examples = next(source_iter)
                batch = _collate_examples(examples, int(tokenizer.pad_token_id))
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                labels = batch["labels"]
                nonpad = int((labels != -100).sum().item())
                with _autocast(device, args.amp):
                    out = model(**batch)
                    loss = out.loss / divisor
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                # Log semantics: sum of per-source mean CE per micro-batch,
                # so step_loss over accum_steps has the same magnitude as before.
                step_loss += float(loss.detach().item()) * accum_steps
                running_tokens += nonpad
        if float(args.grad_clip) > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        running_loss += step_loss

        if _is_main() and (global_step == 1 or global_step % 100 == 0):
            elapsed = max(1e-6, time.time() - t0)
            denom = 1 if global_step == 1 else min(100, global_step)
            print(
                f"[molt5-mix] step={global_step} loss={step_loss:.4f} "
                f"avg_loss={running_loss / denom:.4f} lr={scheduler.get_last_lr()[0]:.3e} "
                f"toks/s={running_tokens / elapsed:.0f}",
                flush=True,
            )
            if global_step % 100 == 0:
                running_loss = 0.0
                running_tokens = 0
                t0 = time.time()

        if int(args.eval_every_steps) > 0 and global_step % int(args.eval_every_steps) == 0:
            metrics = _evaluate(
                model=model,
                c4_loader=c4_valid_loader,
                zinc_loader=zinc_valid_loader,
                c4_dataset=c4_valid_dataset,
                zinc_dataset=zinc_valid_dataset,
                tokenizer=tokenizer,
                args=args,
                device=device,
                step=global_step,
            )
            if _is_main():
                history.append(metrics)
                (out_dir / "eval_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
                print(
                    f"[molt5-mix] eval step={global_step} valid_ppl={metrics['valid_ppl']:.3f} "
                    f"c4_ppl={metrics['c4_valid_ppl']:.3f} zinc_ppl={metrics['zinc_valid_ppl']:.3f}",
                    flush=True,
                )

        if _is_main() and int(args.save_every_steps) > 0 and global_step % int(args.save_every_steps) == 0:
            ckpt_dir = out_dir / "checkpoints" / f"step_{global_step:08d}"
            _save_checkpoint(
                ckpt_dir=ckpt_dir,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                step=global_step,
                args=args,
            )
            latest = ckpt_dir.parent / "latest_step.txt"
            latest.write_text(ckpt_dir.name + "\n", encoding="utf-8")
            keep = int(args.keep_last_checkpoints)
            if keep > 0:
                ckpts = sorted(p for p in ckpt_dir.parent.glob("step_*") if p.is_dir())
                for old in ckpts[:-keep]:
                    for child in old.iterdir():
                        child.unlink()
                    old.rmdir()
            print(f"[molt5-mix] saved {ckpt_dir}", flush=True)

    if _is_main():
        final_dir = out_dir / "model"
        _save_checkpoint(
            ckpt_dir=final_dir,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            step=global_step,
            args=args,
        )
        print(f"[molt5-mix] final {final_dir}", flush=True)
    _barrier()
    _cleanup_dist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
