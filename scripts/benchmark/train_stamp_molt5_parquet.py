#!/usr/bin/env python3
"""Train a MolT5-style encoder-decoder model on STAMP span corruption.

This is intentionally not the original SMILES MolT5 tokenizer. It uses the
project's HybridVocab over STAMP sequences and trains a T5 denoising objective:

    STAMP tokens -> span-corrupted STAMP tokens

The decoder target is the standard T5 sentinel stream containing the removed
spans.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.hybrid_vocab import FCSTAMP_CONTROL_TOKENS, HybridVocab  # noqa: E402


QED_CONDITION_TOKENS = ["[C_QED_LOW]", "[C_QED_MID]", "[C_QED_HIGH]", "[C_QED_VHIGH]"]
SA_CONDITION_TOKENS = ["[C_SA_LOW]", "[C_SA_MID]", "[C_SA_HIGH]", "[C_SA_UNK]"]


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


def _all_reduce_mean(value: float, count: int, device: torch.device) -> tuple[float, int]:
    tensor = torch.tensor([float(value), float(count)], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total, n = float(tensor[0].item()), int(round(float(tensor[1].item())))
    return total / max(1, n), n


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


def _stamp_structural_seed_tokens(max_index: int = 255) -> list[str]:
    toks = ["[END]", "[BACK]", "[Ring]", "[S_R]", "[S_S]", "[S_E]", "[S_Z]"]
    toks.extend(f"[J_{i}]" for i in range(max_index + 1))
    toks.extend(f"[B_{i}]" for i in range(max_index + 1))
    toks.extend(FCSTAMP_CONTROL_TOKENS)
    toks.extend(QED_CONDITION_TOKENS)
    toks.extend(SA_CONDITION_TOKENS)
    return toks


@dataclass
class StampT5Vocab:
    """Hybrid STAMP vocab plus T5 sentinel ids."""

    hybrid: HybridVocab
    num_extra_ids: int = 100

    def __post_init__(self) -> None:
        self.itos = list(self.hybrid.itos)
        self.sentinel_tokens = [f"<extra_id_{i}>" for i in range(int(self.num_extra_ids))]
        for tok in self.sentinel_tokens:
            if tok not in self.hybrid.stoi:
                self.itos.append(tok)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.pad_id = int(self.stoi["<pad>"])
        self.eos_id = int(self.stoi["<eos>"])
        self.unk_id = int(self.stoi["<unk>"])
        self.sentinel_start_id = int(self.stoi["<extra_id_0>"])

    def encode_stamp(
        self,
        tokens: Sequence[str],
        *,
        training: bool,
        rng: random.Random,
        p_low: float,
        p_high: float,
    ) -> list[int]:
        return self.hybrid.encode_stream(tokens, training=training, rng=rng, p_low=p_low, p_high=p_high)

    def sentinel_id(self, index: int) -> int:
        if index >= self.num_extra_ids:
            index = self.num_extra_ids - 1
        return self.sentinel_start_id + int(index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "itos": self.itos,
            "num_extra_ids": int(self.num_extra_ids),
            "hybrid": self.hybrid.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StampT5Vocab":
        return cls(HybridVocab.from_dict(payload["hybrid"]), num_extra_ids=int(payload.get("num_extra_ids", 100)))


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
        wrote = 0
        for pos in range(start, min(length, start + span_len)):
            if not mask[pos]:
                mask[pos] = True
                wrote += 1
                remaining -= 1
                if remaining <= 0:
                    break
        if wrote == 0:
            continue
    if remaining > 0:
        positions = [i for i, v in enumerate(mask) if not v]
        rng.shuffle(positions)
        for pos in positions[:remaining]:
            mask[pos] = True
    return mask


def _span_corrupt(
    ids: Sequence[int],
    *,
    vocab: StampT5Vocab,
    rng: random.Random,
    noise_density: float,
    mean_span_length: float,
    max_source_len: int,
    max_target_len: int,
) -> tuple[list[int], list[int]]:
    body = list(ids)
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
            sentinel = vocab.sentinel_id(span_idx)
            source.append(sentinel)
            target.append(sentinel)
            while i < len(body) and mask[i]:
                target.append(body[i])
                i += 1
            span_idx += 1
        else:
            source.append(body[i])
            i += 1
    source.append(vocab.eos_id)
    target.append(vocab.eos_id)
    source = source[:max_source_len]
    target = target[:max_target_len]
    source[-1] = vocab.eos_id
    target[-1] = vocab.eos_id
    return source, target


class StampT5ParquetDataset(IterableDataset):
    def __init__(
        self,
        parquet_paths: Sequence[str],
        *,
        vocab: StampT5Vocab,
        text_column: str,
        max_rows: int,
        seed: int,
        max_source_len: int,
        max_target_len: int,
        noise_density: float,
        mean_span_length: float,
        fallback_training: bool,
        fallback_p_low: float,
        fallback_p_high: float,
    ) -> None:
        super().__init__()
        self.parquet_paths = list(parquet_paths)
        self.vocab = vocab
        self.text_column = str(text_column)
        self.max_rows = int(max_rows)
        self.seed = int(seed)
        self.epoch = 0
        self.max_source_len = int(max_source_len)
        self.max_target_len = int(max_target_len)
        self.noise_density = float(noise_density)
        self.mean_span_length = float(mean_span_length)
        self.fallback_training = bool(fallback_training)
        self.fallback_p_low = float(fallback_p_low)
        self.fallback_p_high = float(fallback_p_high)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)
        rank = _dist_rank()
        world = _dist_world_size()
        shard_id = rank * num_workers + worker_id
        num_shards = world * num_workers

        row_groups: list[tuple[str, int]] = []
        for path in self.parquet_paths:
            pf = pq.ParquetFile(path)
            row_groups.extend((path, rg) for rg in range(pf.num_row_groups))
        rng = random.Random(self.seed + self.epoch * 1009)
        rng.shuffle(row_groups)
        row_groups = row_groups[shard_id::num_shards]
        corrupt_rng = random.Random(self.seed * 131 + self.epoch * 17 + shard_id)
        fallback_rng = random.Random(self.seed * 8191 + self.epoch * 31 + shard_id)

        if self.max_rows > 0:
            base = self.max_rows // max(1, num_shards)
            rem = self.max_rows % max(1, num_shards)
            worker_limit = base + (1 if shard_id < rem else 0)
        else:
            worker_limit = None

        yielded = 0
        for path, rg in row_groups:
            pf = pq.ParquetFile(path)
            names = set(pf.schema_arrow.names)
            if self.text_column not in names:
                raise KeyError(f"{path} missing text column {self.text_column!r}; available={sorted(names)}")
            table = pf.read_row_group(rg, columns=[self.text_column])
            texts = table.column(self.text_column).to_pylist()
            order = list(range(len(texts)))
            rng.shuffle(order)
            for idx in order:
                text = texts[idx]
                if text is None:
                    continue
                tokens = str(text).strip().split()
                if not tokens:
                    continue
                ids = self.vocab.encode_stamp(
                    tokens,
                    training=self.fallback_training,
                    rng=fallback_rng,
                    p_low=self.fallback_p_low,
                    p_high=self.fallback_p_high,
                )
                if len(ids) < 2:
                    continue
                source, target = _span_corrupt(
                    ids,
                    vocab=self.vocab,
                    rng=corrupt_rng,
                    noise_density=self.noise_density,
                    mean_span_length=self.mean_span_length,
                    max_source_len=self.max_source_len,
                    max_target_len=self.max_target_len,
                )
                yield {"input_ids": source, "labels": target}
                yielded += 1
                if worker_limit is not None and yielded >= worker_limit:
                    return


def _collate(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    max_in = max(len(x["input_ids"]) for x in batch)
    max_lab = max(len(x["labels"]) for x in batch)
    input_ids = torch.full((len(batch), max_in), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_in), dtype=torch.long)
    labels = torch.full((len(batch), max_lab), -100, dtype=torch.long)
    for i, row in enumerate(batch):
        src = torch.tensor(row["input_ids"], dtype=torch.long)
        tgt = torch.tensor(row["labels"], dtype=torch.long)
        input_ids[i, : len(src)] = src
        attention_mask[i, : len(src)] = 1
        labels[i, : len(tgt)] = tgt
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _cycle(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        produced = False
        for batch in loader:
            produced = True
            yield batch
        if not produced:
            raise RuntimeError("DataLoader produced no batches.")


def _autocast(device: torch.device, amp: str):
    mode = str(amp).lower()
    if device.type != "cuda" or mode == "none":
        return nullcontext()
    dtype = torch.bfloat16 if mode in {"auto", "bf16"} else torch.float16
    return torch.amp.autocast("cuda", dtype=dtype)


def _make_model(args: argparse.Namespace, vocab: StampT5Vocab):
    from transformers import T5Config, T5ForConditionalGeneration

    cfg = T5Config(
        vocab_size=len(vocab.itos),
        d_model=int(args.d_model),
        d_kv=max(1, int(args.d_model) // int(args.n_heads)),
        d_ff=int(args.d_ff),
        num_layers=int(args.num_layers),
        num_decoder_layers=int(args.num_decoder_layers),
        num_heads=int(args.n_heads),
        dropout_rate=float(args.dropout),
        layer_norm_epsilon=1e-6,
        feed_forward_proj=str(args.feed_forward_proj),
        pad_token_id=vocab.pad_id,
        eos_token_id=vocab.eos_id,
        decoder_start_token_id=vocab.pad_id,
    )
    return T5ForConditionalGeneration(cfg)


def _save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    args: argparse.Namespace,
    vocab: StampT5Vocab,
) -> None:
    raw = model.module if hasattr(model, "module") else model
    payload = {
        "step": int(step),
        "model_state": raw.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "args": vars(args),
        "vocab": vocab.to_dict(),
        "model_config": raw.config.to_dict() if hasattr(raw, "config") else {},
    }
    _atomic_save(payload, path)


@torch.no_grad()
def _evaluate(
    *,
    model: torch.nn.Module,
    dataset: StampT5ParquetDataset,
    vocab: StampT5Vocab,
    args: argparse.Namespace,
    device: torch.device,
    step: int,
) -> dict[str, Any]:
    dataset.set_epoch(10_000 + int(step))
    loader = DataLoader(
        dataset,
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_data_workers),
        collate_fn=lambda b: _collate(b, vocab.pad_id),
        pin_memory=device.type == "cuda",
    )
    model.eval()
    loss_sum = 0.0
    count = 0
    max_batches = int(args.eval_max_batches)
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with _autocast(device, args.amp):
            out = model(**batch)
        loss_sum += float(out.loss.detach().item())
        count += 1
    mean_loss, total_batches = _all_reduce_mean(loss_sum, count, device)
    model.train()
    return {
        "step": int(step),
        "valid_loss": float(mean_loss),
        "valid_ppl": float(math.exp(min(20.0, mean_loss))),
        "valid_batches": int(total_batches),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-parquet", required=True, help="Train parquet glob(s), comma-separated.")
    p.add_argument("--valid-parquet", required=True, help="Validation parquet glob(s), comma-separated.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--text-column", default="stamp")
    p.add_argument("--stamp-vocab", default="/Project_Storage/CFP-03/CFP03-CF-130/data/zinc_pretrain/vocab_merged/motif_vocab.txt")
    p.add_argument("--motif-frequency-file", default="/Project_Storage/CFP-03/CFP03-CF-130/data/zinc_pretrain/vocab_merged/zinc_plus_augment_full_freq.txt")
    p.add_argument("--freq-cutoff", type=int, default=5000)
    p.add_argument("--max-atomic-motifs", type=int, default=12000)
    p.add_argument("--num-extra-ids", type=int, default=100)
    p.add_argument("--fallback-p-low", type=float, default=0.02)
    p.add_argument("--fallback-p-high", type=float, default=0.15)
    p.add_argument("--no-random-motif-fallback", dest="random_motif_fallback", action="store_false", default=True)

    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-decoder-layers", type=int, default=12)
    p.add_argument("--d-ff", type=int, default=3072)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--feed-forward-proj", default="gated-gelu")
    p.add_argument("--max-source-len", type=int, default=256)
    p.add_argument("--max-target-len", type=int, default=128)
    p.add_argument("--noise-density", type=float, default=0.15)
    p.add_argument("--mean-noise-span-length", type=float, default=3.0)

    p.add_argument("--batch-size", type=int, default=64, help="Per-GPU micro batch size.")
    p.add_argument("--global-batch-size", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=200000)
    p.add_argument("--train-max-rows", type=int, default=0, help="0 means all rows each streaming cycle.")
    p.add_argument("--valid-max-rows", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=5000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every-steps", type=int, default=5000)
    p.add_argument("--save-every-steps", type=int, default=10000)
    p.add_argument("--keep-last-checkpoints", type=int, default=10)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--eval-max-batches", type=int, default=0)
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

    train_paths = _resolve_paths(args.train_parquet)
    valid_paths = _resolve_paths(args.valid_parquet)
    if not train_paths:
        raise FileNotFoundError(args.train_parquet)
    if not valid_paths:
        raise FileNotFoundError(args.valid_parquet)

    out_dir = Path(args.output_dir)
    if _is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
    _barrier()

    hybrid = HybridVocab.build(
        args.stamp_vocab,
        freq_cutoff=int(args.freq_cutoff),
        extra_stamp_tokens=_stamp_structural_seed_tokens(),
        frequency_path=args.motif_frequency_file,
        max_atomic_motifs=int(args.max_atomic_motifs),
    )
    vocab = StampT5Vocab(hybrid, num_extra_ids=int(args.num_extra_ids))
    if _is_main():
        (out_dir / "vocab.json").write_text(json.dumps(vocab.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "train_args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[stamp-molt5] train_files={len(train_paths)} valid_files={len(valid_paths)} vocab_size={len(vocab.itos)}", flush=True)

    model = _make_model(args, vocab).to(device)
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
        ckpt = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        global_step = int(ckpt.get("step", 0))

    if _is_dist():
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    train_dataset = StampT5ParquetDataset(
        train_paths,
        vocab=vocab,
        text_column=args.text_column,
        max_rows=int(args.train_max_rows),
        seed=int(args.seed),
        max_source_len=int(args.max_source_len),
        max_target_len=int(args.max_target_len),
        noise_density=float(args.noise_density),
        mean_span_length=float(args.mean_noise_span_length),
        fallback_training=bool(args.random_motif_fallback),
        fallback_p_low=float(args.fallback_p_low),
        fallback_p_high=float(args.fallback_p_high),
    )
    valid_dataset = StampT5ParquetDataset(
        valid_paths,
        vocab=vocab,
        text_column=args.text_column,
        max_rows=int(args.valid_max_rows),
        seed=int(args.seed) + 999,
        max_source_len=int(args.max_source_len),
        max_target_len=int(args.max_target_len),
        noise_density=float(args.noise_density),
        mean_span_length=float(args.mean_noise_span_length),
        fallback_training=False,
        fallback_p_low=float(args.fallback_p_low),
        fallback_p_high=float(args.fallback_p_high),
    )

    per_step_global = int(args.batch_size) * max(1, _dist_world_size())
    accum_steps = max(1, int(math.ceil(int(args.global_batch_size) / max(1, per_step_global))))
    loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_data_workers),
        collate_fn=lambda b: _collate(b, vocab.pad_id),
        pin_memory=device.type == "cuda",
        prefetch_factor=2 if int(args.num_data_workers) > 0 else None,
    )
    batches = _cycle(loader)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp == "fp16"))
    model.train()

    history: list[dict[str, Any]] = []
    t0 = time.time()
    running_loss = 0.0
    running_tokens = 0
    optimizer.zero_grad(set_to_none=True)
    while global_step < int(args.max_steps):
        epoch = global_step // max(1, len(train_paths))
        train_dataset.set_epoch(epoch)
        step_loss = 0.0
        for _ in range(accum_steps):
            batch = next(batches)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = batch["labels"]
            nonpad = int((labels != -100).sum().item())
            with _autocast(device, args.amp):
                out = model(**batch)
                loss = out.loss / accum_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
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
            print(
                f"[stamp-molt5] step={global_step} loss={step_loss:.4f} "
                f"avg_loss={running_loss / max(1, min(global_step, 100)):.4f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} toks/s={running_tokens / elapsed:.0f}",
                flush=True,
            )
            if global_step % 100 == 0:
                running_loss = 0.0
                running_tokens = 0
                t0 = time.time()

        if int(args.eval_every_steps) > 0 and global_step % int(args.eval_every_steps) == 0:
            metrics = _evaluate(model=model, dataset=valid_dataset, vocab=vocab, args=args, device=device, step=global_step)
            if _is_main():
                history.append(metrics)
                (out_dir / "eval_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
                print(
                    f"[stamp-molt5] eval step={global_step} valid_loss={metrics['valid_loss']:.4f} "
                    f"valid_ppl={metrics['valid_ppl']:.3f} batches={metrics['valid_batches']}",
                    flush=True,
                )

        if _is_main() and int(args.save_every_steps) > 0 and global_step % int(args.save_every_steps) == 0:
            ckpt_dir = out_dir / "checkpoints"
            path = ckpt_dir / f"step_{global_step:08d}.pt"
            _save_checkpoint(path=path, model=model, optimizer=optimizer, scheduler=scheduler, step=global_step, args=args, vocab=vocab)
            latest = ckpt_dir / "latest_step.txt"
            latest.write_text(path.name + "\n", encoding="utf-8")
            keep = int(args.keep_last_checkpoints)
            if keep > 0:
                ckpts = sorted(ckpt_dir.glob("step_*.pt"))
                for old in ckpts[:-keep]:
                    old.unlink(missing_ok=True)
            print(f"[stamp-molt5] saved {path}", flush=True)

    if _is_main():
        final_path = out_dir / "model.pt"
        _save_checkpoint(path=final_path, model=model, optimizer=optimizer, scheduler=scheduler, step=global_step, args=args, vocab=vocab)
        print(f"[stamp-molt5] final {final_path}", flush=True)
    _barrier()
    _cleanup_dist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
