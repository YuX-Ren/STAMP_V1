#!/usr/bin/env python3
"""CamS-aligned decoder-only NTP pretraining over sharded STAMP parquet data.

This script is intentionally narrow:
  - input is a set of pre-encoded STAMP parquet shards with a text column
    (default: ``stamp``)
  - train / valid are separated by filename glob, not by an in-file split column
  - the motif vocabulary is seeded from a static ``motif_vocab.txt``
  - training objective is standard left-to-right next-token prediction

It is meant to align STAMP pretraining with the core CamS recipe:
decoder-only backbone + motif-level sequence modeling + fixed tokenizer vocab.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import random
import shutil
import sys
import time
from collections import Counter, deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from rdkit import RDLogger
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.common import iterable_with_progress  # noqa: E402
from stamp.benchmark.lm import (  # noqa: E402
    LMConfig,
    LlamaDecoderLM,
    NextTokenDataset,
    TinyDecoderLM,
    TokenVocab,
    _resolve_amp_mode,
    _resolve_torch_device,
    _set_seed,
)
from stamp.benchmark.molformer_stamp import (  # noqa: E402
    default_stamp_structural_tokens,
    smiles_vocab_tokens_from_stamp_vocabs,
)
from stamp.benchmark.hybrid_vocab import smiles_char_tokens  # noqa: E402
from stamp.benchmark.multiscale_stamp import (  # noqa: E402
    make_scale_tokens,
    parse_csv_arg,
    save_multiscale_manifest,
)
from stamp.vocab.branch_paths import resolve_branch_output_dir  # noqa: E402

from scripts.benchmark.train_ao_gpt_parquet import (  # noqa: E402
    _build_vocab_from_counter,
    _cleanup_distributed,
    _dist_local_rank,
    _dist_rank,
    _dist_world_size,
    _distributed_barrier,
    _distributed_mean,
    _histogram,
    _is_dist,
    _is_main_process,
    _load_vocab_seed_tokens,
    _setup_distributed,
    _summarize_lengths,
    _unwrap_model,
    build_dynamic_lm_collate_fn,
)


@dataclass
class ShardScanSummary:
    train_rows: int
    valid_rows_total: int
    valid_rows_sampled: int
    train_rows_scanned: int
    valid_rows_scanned: int
    train_examples_per_raw_row: float
    train_single_scale_view_counts: dict[str, int]
    valid_single_scale_view_counts: dict[str, int]
    train_lengths: list[int]
    valid_lengths: list[int]
    train_hist: dict[str, int]
    valid_hist: dict[str, int]
    valid_sequences: list[list[int]]
    observed_structural_tokens: list[str]
    oov_token_occurrences: int
    oov_unique_tokens: int
    oov_examples: list[str]


_SCAN_CACHE_NAME = "_scan_cache_cams_decoder_v4.pt"


def _scan_cache_path(out_dir: Path) -> Path:
    return out_dir / _SCAN_CACHE_NAME


def _save_scan_cache(out_dir: Path, summary: ShardScanSummary, vocab: TokenVocab) -> None:
    payload = {
        "summary": {
            "train_rows": summary.train_rows,
            "valid_rows_total": summary.valid_rows_total,
            "valid_rows_sampled": summary.valid_rows_sampled,
            "train_rows_scanned": summary.train_rows_scanned,
            "valid_rows_scanned": summary.valid_rows_scanned,
            "train_examples_per_raw_row": summary.train_examples_per_raw_row,
            "train_single_scale_view_counts": summary.train_single_scale_view_counts,
            "valid_single_scale_view_counts": summary.valid_single_scale_view_counts,
            "train_lengths": summary.train_lengths,
            "valid_lengths": summary.valid_lengths,
            "train_hist": summary.train_hist,
            "valid_hist": summary.valid_hist,
            "valid_sequences": summary.valid_sequences,
            "observed_structural_tokens": summary.observed_structural_tokens,
            "oov_token_occurrences": summary.oov_token_occurrences,
            "oov_unique_tokens": summary.oov_unique_tokens,
            "oov_examples": summary.oov_examples,
        },
        "vocab_itos": vocab.itos,
    }
    final_path = _scan_cache_path(out_dir)
    tmp_path = final_path.with_name(f".{final_path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, final_path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _load_scan_cache(out_dir: Path) -> tuple[ShardScanSummary, TokenVocab]:
    payload = torch.load(_scan_cache_path(out_dir), map_location="cpu", weights_only=False)
    s = payload["summary"]
    vocab_itos = list(payload["vocab_itos"])
    vocab = TokenVocab(stoi={tok: i for i, tok in enumerate(vocab_itos)}, itos=vocab_itos)
    summary = ShardScanSummary(
        train_rows=int(s["train_rows"]),
        valid_rows_total=int(s["valid_rows_total"]),
        valid_rows_sampled=int(s["valid_rows_sampled"]),
        train_rows_scanned=int(s.get("train_rows_scanned", s["train_rows"])),
        valid_rows_scanned=int(s.get("valid_rows_scanned", s["valid_rows_total"])),
        train_examples_per_raw_row=float(s.get("train_examples_per_raw_row", 1.0)),
        train_single_scale_view_counts={
            str(k): int(v) for k, v in dict(s.get("train_single_scale_view_counts", {})).items()
        },
        valid_single_scale_view_counts={
            str(k): int(v) for k, v in dict(s.get("valid_single_scale_view_counts", {})).items()
        },
        train_lengths=[int(x) for x in s["train_lengths"]],
        valid_lengths=[int(x) for x in s["valid_lengths"]],
        train_hist={str(k): int(v) for k, v in dict(s["train_hist"]).items()},
        valid_hist={str(k): int(v) for k, v in dict(s["valid_hist"]).items()},
        valid_sequences=[[int(y) for y in seq] for seq in s["valid_sequences"]],
        observed_structural_tokens=[str(x) for x in s["observed_structural_tokens"]],
        oov_token_occurrences=int(s["oov_token_occurrences"]),
        oov_unique_tokens=int(s["oov_unique_tokens"]),
        oov_examples=[str(x) for x in s["oov_examples"]],
    )
    return summary, vocab


def _load_scan_cache_when_ready(
    out_dir: Path,
    *,
    timeout_s: float = 900.0,
    poll_s: float = 0.5,
) -> tuple[ShardScanSummary, TokenVocab]:
    path = _scan_cache_path(out_dir)
    deadline = time.monotonic() + timeout_s
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            if path.exists() and path.stat().st_size > 0:
                return _load_scan_cache(out_dir)
        except (EOFError, OSError, RuntimeError, pickle.UnpicklingError) as exc:
            last_error = exc
        time.sleep(poll_s)
    raise TimeoutError(f"Timed out waiting for readable scan cache: {path}") from last_error


def _sorted_unique_paths(pattern: str) -> list[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched glob: {pattern}")
    return paths


def _count_parquet_rows(paths: list[str]) -> int:
    total = 0
    for path in paths:
        meta = pq.ParquetFile(path).metadata
        total += int(meta.num_rows) if meta is not None else 0
    return total


def _iter_text_rows(paths: list[str], text_column: str) -> Iterator[str]:
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


def _tokenize_stamp_text(text: str) -> list[str]:
    toks = str(text).split()
    if toks and toks[0] == "[BOS]":
        toks = toks[1:]
    if toks and toks[-1] == "[EOS]":
        toks = toks[:-1]
    return toks


def _append_unique_tokens(tokens: list[str], extra: list[str]) -> list[str]:
    seen = set(tokens)
    for tok in extra:
        if tok not in seen:
            tokens.append(tok)
            seen.add(tok)
    return tokens


def _encode_tokens_with_smiles_fallback(vocab: TokenVocab, toks: list[str]) -> list[int]:
    ids: list[int] = []
    for tok in toks:
        tok_id = vocab.stoi.get(tok)
        if tok_id is not None:
            ids.append(tok_id)
            continue
        if TokenVocab._is_stamp_structural(tok):
            ids.append(vocab.unk_id)
            continue
        try:
            fallback = smiles_char_tokens(tok)
        except ValueError:
            fallback = []
        if fallback and all(piece in vocab.stoi for piece in fallback):
            ids.extend(vocab.stoi[piece] for piece in fallback)
        else:
            ids.append(vocab.unk_id)
    return ids


def _split_concat_views(toks: list[str]) -> list[list[str]]:
    views: list[list[str]] = []
    current: list[str] = []
    for tok in toks:
        if tok == "[CONCAT]":
            if current:
                views.append(current)
                current = []
            continue
        current.append(tok)
    if current:
        views.append(current)
    return views


def _expand_views(toks: list[str], mode: str) -> list[list[str]]:
    if mode == "concat_only":
        return [toks]
    if mode != "cams_multiview":
        raise ValueError(f"Unsupported view mode: {mode}")
    single_scale = _split_concat_views(toks)
    if len(single_scale) <= 1:
        return [toks]
    return single_scale + [toks]


def _validate_single_scale_view_counts(
    summary: ShardScanSummary,
    expected_single_scale_views: int,
) -> None:
    if expected_single_scale_views <= 0:
        return
    expected = str(int(expected_single_scale_views))
    train_bad = {
        count: n
        for count, n in summary.train_single_scale_view_counts.items()
        if count != expected and int(n) > 0
    }
    valid_bad = {
        count: n
        for count, n in summary.valid_single_scale_view_counts.items()
        if count != expected and int(n) > 0
    }
    if train_bad or valid_bad:
        raise ValueError(
            "CaMS multiview mismatch: expected "
            f"{expected_single_scale_views} single-scale views per molecule, "
            f"but observed train={summary.train_single_scale_view_counts} "
            f"valid={summary.valid_single_scale_view_counts}. "
            "Each row must contain the expected number of [CONCAT]-separated "
            "scale sequences; cams_multiview then creates N+1 NTP instances "
            "(N single-scale views + one concat view)."
        )


def _special_loss_ignore_ids(vocab: TokenVocab, scale_names: list[str]) -> set[int]:
    skip_tokens = {"[CONCAT]", "[BOS]", "[EOS]"}
    skip_tokens.update(make_scale_tokens(scale_names))
    ids = {vocab.bos_id, vocab.eos_id}
    ids.update(vocab.stoi[tok] for tok in skip_tokens if tok in vocab.stoi)
    return ids


def _mask_special_targets(targets: torch.Tensor, ignore_ids: set[int]) -> torch.Tensor:
    if not ignore_ids:
        return targets
    out = targets.clone()
    for tok_id in ignore_ids:
        out = out.masked_fill(out == int(tok_id), -100)
    return out


def _truncate_ids_keep_eos(ids: list[int], *, max_seq_len: int, eos_id: int) -> list[int]:
    """Return at most max_seq_len + 1 ids so x/y length stays <= max_seq_len."""
    limit = int(max_seq_len) + 1
    if len(ids) <= limit:
        return ids
    if limit <= 1:
        return ids[:limit]
    if ids[-1] == int(eos_id):
        return ids[: limit - 1] + [ids[-1]]
    return ids[:limit]


def _make_next_token_pair(ids: list[int], *, max_seq_len: int, eos_id: int) -> tuple[list[int], list[int]]:
    ids = _truncate_ids_keep_eos(ids, max_seq_len=max_seq_len, eos_id=eos_id)
    return ids[:-1], ids[1:]


def _scheduled_lr(
    *,
    base_lr: float,
    min_lr: float,
    step: int,
    total_steps: int,
    warmup_steps: int,
    scheduler: str,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if scheduler == "constant":
        return base_lr
    denom = max(1, total_steps - max(0, warmup_steps))
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (base_lr - min_lr)


def _scan_shards_for_vocab_and_valid(
    *,
    train_paths: list[str],
    valid_paths: list[str],
    text_column: str,
    seed_tokens: list[str],
    train_view_mode: str,
    valid_view_mode: str,
    train_scan_max_rows: Optional[int],
    valid_max_rows: int,
    valid_scan_max_rows: Optional[int],
    seed: int,
    show_progress: bool,
) -> tuple[ShardScanSummary, TokenVocab]:
    seed_set = set(seed_tokens)
    counter: Counter[str] = Counter()
    structural_tokens: set[str] = set()
    oov_counter: Counter[str] = Counter()
    train_rows = 0
    train_examples = 0
    valid_rows_total = 0
    train_single_scale_view_counts: Counter[int] = Counter()
    valid_single_scale_view_counts: Counter[int] = Counter()
    train_lengths: list[int] = []
    valid_lengths: list[int] = []
    valid_sample_tokens: list[list[str]] = []
    rng = random.Random(seed + 17)

    train_iter = iterable_with_progress(
        _iter_text_rows(train_paths, text_column),
        show_progress=show_progress,
        desc="Scan train shards",
    )
    for text in train_iter:
        toks = _tokenize_stamp_text(text)
        if not toks:
            continue
        train_rows += 1
        train_single_scale_view_counts[len(_split_concat_views(toks))] += 1
        views = _expand_views(toks, train_view_mode)
        train_examples += len(views)
        for view in views:
            train_lengths.append(len(view))
            for tok in view:
                counter[tok] += 1
                if TokenVocab._is_stamp_structural(tok):
                    structural_tokens.add(tok)
                elif tok not in seed_set:
                    oov_counter[tok] += 1
        if train_scan_max_rows is not None and train_rows >= train_scan_max_rows:
            break

    valid_iter = iterable_with_progress(
        _iter_text_rows(valid_paths, text_column),
        show_progress=show_progress,
        desc="Scan valid shards",
    )
    for text in valid_iter:
        toks = _tokenize_stamp_text(text)
        if not toks:
            continue
        valid_rows_total += 1
        valid_single_scale_view_counts[len(_split_concat_views(toks))] += 1
        valid_lengths.append(len(toks))
        for tok in toks:
            if TokenVocab._is_stamp_structural(tok):
                structural_tokens.add(tok)
            elif tok not in seed_set:
                oov_counter[tok] += 1
        valid_views = _expand_views(toks, valid_view_mode)
        if len(valid_sample_tokens) < valid_max_rows:
            valid_sample_tokens.extend(valid_views[: max(0, valid_max_rows - len(valid_sample_tokens))])
        else:
            j = rng.randint(0, valid_rows_total - 1)
            if j < valid_max_rows and valid_views:
                valid_sample_tokens[j] = valid_views[0]
        if valid_scan_max_rows is not None and valid_rows_total >= valid_scan_max_rows:
            break

    # The encoded msring shards can contain valid fallback fragments that are not
    # in the fixed scale vocab files. Keep observed fallback tokens instead of
    # collapsing them to <unk>, otherwise salt separators and rare multi-bond
    # motifs lose information during CaMS-style NTP.
    dynamic_tokens = Counter({tok: int(freq) for tok, freq in counter.items() if tok in structural_tokens})
    dynamic_tokens.update(oov_counter)
    vocab = _build_vocab_from_counter(dynamic_tokens, seed_tokens=seed_tokens)
    valid_sequences = [
        [vocab.bos_id] + _encode_tokens_with_smiles_fallback(vocab, toks) + [vocab.eos_id]
        for toks in valid_sample_tokens
    ]
    summary = ShardScanSummary(
        train_rows=train_examples,
        valid_rows_total=valid_rows_total,
        valid_rows_sampled=len(valid_sequences),
        train_rows_scanned=train_rows,
        valid_rows_scanned=valid_rows_total,
        train_examples_per_raw_row=(float(train_examples) / float(max(1, train_rows))),
        train_single_scale_view_counts={str(k): int(v) for k, v in sorted(train_single_scale_view_counts.items())},
        valid_single_scale_view_counts={str(k): int(v) for k, v in sorted(valid_single_scale_view_counts.items())},
        train_lengths=train_lengths,
        valid_lengths=valid_lengths[: len(valid_sample_tokens)],
        train_hist=_histogram(train_lengths),
        valid_hist=_histogram(valid_lengths[: len(valid_sample_tokens)]),
        valid_sequences=valid_sequences,
        observed_structural_tokens=sorted(structural_tokens),
        oov_token_occurrences=int(sum(oov_counter.values())),
        oov_unique_tokens=len(oov_counter),
        oov_examples=[tok for tok, _ in oov_counter.most_common(32)],
    )
    return summary, vocab


class _ShardTokenPairDataset(IterableDataset[tuple[list[int], list[int]]]):
    def __init__(
        self,
        paths: list[str],
        *,
        vocab: TokenVocab,
        text_column: str,
        view_mode: str,
        max_seq_len: int,
        max_rows: Optional[int],
        seed: int,
        length_bucket_size: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.paths = list(paths)
        self.vocab = vocab
        self.text_column = text_column
        self.view_mode = view_mode
        self.max_seq_len = int(max_seq_len)
        self.max_rows = max_rows
        self.seed = int(seed)
        self.length_bucket_size = max(0, int(length_bucket_size))
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

        rng = random.Random(self.seed + self.epoch)
        paths = list(self.paths)
        rng.shuffle(paths)
        paths = paths[shard_id::num_shards]

        if self.max_rows is not None:
            base = self.max_rows // num_shards
            rem = self.max_rows % num_shards
            worker_limit = base + (1 if shard_id < rem else 0)
        else:
            worker_limit = None

        yielded = 0
        for path in paths:
            pf = pq.ParquetFile(path)
            rg_indices = list(range(pf.num_row_groups))
            rng.shuffle(rg_indices)
            for rg in rg_indices:
                tab = pf.read_row_group(rg, columns=[self.text_column])
                rows = tab.column(0).to_pylist()
                order = list(range(len(rows)))
                rng.shuffle(order)
                if self.length_bucket_size > 0:
                    buckets: dict[int, list[tuple[list[int], list[int]]]] = {}
                    for idx in order:
                        text = rows[idx]
                        if text is None:
                            continue
                        toks = _tokenize_stamp_text(str(text))
                        if not toks:
                            continue
                        for view in _expand_views(toks, self.view_mode):
                            ids = [self.vocab.bos_id] + _encode_tokens_with_smiles_fallback(self.vocab, view) + [self.vocab.eos_id]
                            if len(ids) < 2:
                                continue
                            x, y = _make_next_token_pair(
                                ids,
                                max_seq_len=self.max_seq_len,
                                eos_id=self.vocab.eos_id,
                            )
                            if not x:
                                continue
                            key = int(math.ceil(len(x) / self.length_bucket_size) * self.length_bucket_size)
                            buckets.setdefault(key, []).append((x, y))
                    keys = list(buckets)
                    rng.shuffle(keys)
                    for key in keys:
                        items = buckets[key]
                        rng.shuffle(items)
                        for x, y in items:
                            yield x, y
                            yielded += 1
                            if worker_limit is not None and yielded >= worker_limit:
                                return
                    continue
                for idx in order:
                    text = rows[idx]
                    if text is None:
                        continue
                    toks = _tokenize_stamp_text(str(text))
                    if not toks:
                        continue
                    for view in _expand_views(toks, self.view_mode):
                        ids = [self.vocab.bos_id] + _encode_tokens_with_smiles_fallback(self.vocab, view) + [self.vocab.eos_id]
                        if len(ids) < 2:
                            continue
                        x, y = _make_next_token_pair(
                            ids,
                            max_seq_len=self.max_seq_len,
                            eos_id=self.vocab.eos_id,
                        )
                        if not x:
                            continue
                        yield x, y
                        yielded += 1
                        if worker_limit is not None and yielded >= worker_limit:
                            return


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CamS-aligned decoder-only NTP pretraining on sharded STAMP parquet.")
    p.add_argument("--train-glob", required=True)
    p.add_argument("--valid-glob", required=True)
    p.add_argument("--stamp-vocab", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--text-column", default="stamp")
    p.add_argument("--multiscale-vocabs", default="", help="Comma-separated STAMP vocab paths for fine-to-coarse scales.")
    p.add_argument("--multiscale-scale-names", default="", help="Comma-separated scale names matching --multiscale-vocabs.")
    p.add_argument("--train-view-mode", default="cams_multiview", choices=["concat_only", "cams_multiview"])
    p.add_argument("--valid-view-mode", default="concat_only", choices=["concat_only", "cams_multiview"])
    p.add_argument(
        "--expected-single-scale-views",
        type=int,
        default=0,
        help="0 disables. Set to the expected number of [CONCAT]-separated single-scale views, e.g. 3 for STAMP 1k/4k/12k or 4 for strict CaMS.",
    )
    p.add_argument("--loss-skip-special-tokens", action="store_true", default=True)
    p.add_argument("--loss-include-special-tokens", dest="loss_skip_special_tokens", action="store_false")
    p.add_argument("--train-max-rows", type=int, default=0, help="0 = all train rows.")
    p.add_argument("--length-bucket-size", type=int, default=0, help="0 disables; otherwise batch nearby sequence lengths to reduce padding.")
    p.add_argument("--valid-max-rows", type=int, default=20000)
    p.add_argument("--scan-train-max-rows", type=int, default=200000, help="Rows to scan for vocab/length stats; 0 = all.")
    p.add_argument("--scan-valid-max-rows", type=int, default=20000, help="Rows to scan for validation sampling; 0 = all.")
    p.add_argument("--structural-max-index", type=int, default=512, help="Pre-seed [J_i]/[B_i] structural tokens up to this index.")
    p.add_argument("--include-smiles-fallback-tokens", action="store_true", default=True)
    p.add_argument("--no-include-smiles-fallback-tokens", dest="include_smiles_fallback_tokens", action="store_false")
    p.add_argument("--scan-only", action="store_true")
    p.add_argument("--no-tqdm", action="store_true")
    p.add_argument("--log-every-micro-steps", type=int, default=20)
    p.add_argument("--loss-log-window", type=int, default=100)

    # CamS-like model size (~100M) and context length by default.
    p.add_argument("--d-model", type=int, default=720)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=2880)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--backbone", choices=["llama", "tiny"], default="llama")
    p.add_argument("--no-tie-embeddings", action="store_true")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--global-batch-size", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lr-scheduler", choices=["cosine", "constant"], default="cosine")
    p.add_argument("--warmup-steps", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-optimizer-steps", type=int, default=0, help="0 = no cap; otherwise stop after this many optimizer steps.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-data-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "none"])
    p.add_argument("--use-adaln", action="store_true", help="Use AO-GPT AdaLN decoder branch; default is plain decoder for CamS.")
    p.add_argument("--torch-compile", action="store_true", default=True)
    p.add_argument("--no-torch-compile", dest="torch_compile", action="store_false")
    p.add_argument("--no-tf32", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-parallel", action="store_true")
    p.add_argument("--save-every-epoch", action="store_true", default=True)
    p.add_argument("--no-save-every-epoch", dest="save_every_epoch", action="store_false")
    p.add_argument("--save-every-steps", type=int, default=0, help="Save checkpoint every N optimizer steps; 0 disables step checkpoints.")
    p.add_argument("--validate-on-save", action="store_true", default=True, help="Run validation whenever a step checkpoint is saved.")
    p.add_argument("--no-validate-on-save", dest="validate_on_save", action="store_false")
    p.add_argument("--reuse-scan-cache", action="store_true")
    p.add_argument("--resume-from", default="", help="Checkpoint path to resume model/optimizer/scaler state from.")
    return p


def _strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {
            (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
            for k, v in state_dict.items()
        }
    return state_dict


def main() -> int:
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")

    train_paths = _sorted_unique_paths(args.train_glob)
    valid_paths = _sorted_unique_paths(args.valid_glob)
    stamp_vocab = Path(args.stamp_vocab)
    if not stamp_vocab.exists():
        raise FileNotFoundError(stamp_vocab)
    resume_path = Path(args.resume_from) if str(args.resume_from).strip() else None
    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(resume_path)

    output_dir = resolve_branch_output_dir(args.output_dir, stamp_vocab=str(stamp_vocab))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    local_rank = _dist_local_rank()
    cfg = LMConfig(
        model_type="decoder",
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
    cfg.use_adaln = bool(args.use_adaln)
    _set_seed(cfg.seed)

    if args.data_parallel and _is_dist():
        raise ValueError("Cannot combine --data-parallel with torchrun/DDP.")
    if _is_dist() and str(args.device).startswith("cuda"):
        dev = _resolve_torch_device(f"cuda:{local_rank}")
    else:
        dev = _resolve_torch_device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index if dev.index is not None else 0)
    _setup_distributed(dev)

    vocab_header, seed_tokens = _load_vocab_seed_tokens(stamp_vocab)
    seed_tokens = _append_unique_tokens(
        seed_tokens,
        default_stamp_structural_tokens(max_index=max(1, int(args.structural_max_index))),
    )
    multiscale_vocab_paths = parse_csv_arg(args.multiscale_vocabs)
    multiscale_scale_names = parse_csv_arg(args.multiscale_scale_names)
    if multiscale_vocab_paths:
        if multiscale_scale_names and len(multiscale_scale_names) != len(multiscale_vocab_paths):
            raise ValueError("--multiscale-scale-names must match --multiscale-vocabs length.")
        if not multiscale_scale_names:
            multiscale_scale_names = [f"s{i}" for i in range(len(multiscale_vocab_paths))]
        for vp in multiscale_vocab_paths:
            if not Path(vp).exists():
                raise FileNotFoundError(vp)
            _scale_header, scale_tokens = _load_vocab_seed_tokens(Path(vp))
            seed_tokens = _append_unique_tokens(seed_tokens, scale_tokens)
        seed_tokens = _append_unique_tokens(seed_tokens, make_scale_tokens(multiscale_scale_names))
    if bool(args.include_smiles_fallback_tokens):
        seed_tokens = _append_unique_tokens(
            seed_tokens,
            smiles_vocab_tokens_from_stamp_vocabs([stamp_vocab] + [Path(p) for p in multiscale_vocab_paths]),
        )
    train_max_rows = None if int(args.train_max_rows) <= 0 else int(args.train_max_rows)
    train_rows_total = _count_parquet_rows(train_paths)
    valid_rows_total = _count_parquet_rows(valid_paths)
    train_scan_max_rows = None if int(args.scan_train_max_rows) <= 0 else int(args.scan_train_max_rows)
    valid_scan_max_rows = None if int(args.scan_valid_max_rows) <= 0 else int(args.scan_valid_max_rows)
    cache_path = _scan_cache_path(out_dir)

    def _set_total_counts(summary: ShardScanSummary) -> None:
        examples_per_raw = max(1.0, float(summary.train_examples_per_raw_row))
        if train_max_rows is not None:
            summary.train_rows = int(train_max_rows)
        else:
            summary.train_rows = int(math.ceil(float(train_rows_total) * examples_per_raw))
        summary.valid_rows_total = int(valid_rows_total)

    reused_scan_cache = False
    if _is_main_process():
        if bool(args.reuse_scan_cache) and cache_path.exists():
            try:
                scan_summary, vocab = _load_scan_cache(out_dir)
                reused_scan_cache = True
            except (EOFError, OSError, RuntimeError, pickle.UnpicklingError) as exc:
                print(
                    f"[WARN] ignoring unreadable scan cache {cache_path}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                scan_summary, vocab = _scan_shards_for_vocab_and_valid(
                    train_paths=train_paths,
                    valid_paths=valid_paths,
                    text_column=args.text_column,
                    seed_tokens=seed_tokens,
                    train_view_mode=args.train_view_mode,
                    valid_view_mode=args.valid_view_mode,
                    train_scan_max_rows=train_scan_max_rows,
                    valid_max_rows=int(args.valid_max_rows),
                    valid_scan_max_rows=valid_scan_max_rows,
                    seed=cfg.seed,
                    show_progress=not args.no_tqdm,
                )
                _set_total_counts(scan_summary)
                _save_scan_cache(out_dir, scan_summary, vocab)
        else:
            scan_summary, vocab = _scan_shards_for_vocab_and_valid(
                train_paths=train_paths,
                valid_paths=valid_paths,
                text_column=args.text_column,
                seed_tokens=seed_tokens,
                train_view_mode=args.train_view_mode,
                valid_view_mode=args.valid_view_mode,
                train_scan_max_rows=train_scan_max_rows,
                valid_max_rows=int(args.valid_max_rows),
                valid_scan_max_rows=valid_scan_max_rows,
                seed=cfg.seed,
                show_progress=not args.no_tqdm,
            )
            _set_total_counts(scan_summary)
            _save_scan_cache(out_dir, scan_summary, vocab)
        _set_total_counts(scan_summary)
        print(
            json.dumps(
                {
                    "train_files": len(train_paths),
                    "valid_files": len(valid_paths),
                    "train_rows": scan_summary.train_rows,
                    "valid_rows_total": scan_summary.valid_rows_total,
                    "valid_rows_sampled": scan_summary.valid_rows_sampled,
                    "train_rows_scanned": scan_summary.train_rows_scanned,
                    "valid_rows_scanned": scan_summary.valid_rows_scanned,
                    "train_examples_per_raw_row": scan_summary.train_examples_per_raw_row,
                    "train_single_scale_view_counts": scan_summary.train_single_scale_view_counts,
                    "valid_single_scale_view_counts": scan_summary.valid_single_scale_view_counts,
                    "expected_single_scale_views": int(args.expected_single_scale_views),
                    "vocab_size": len(vocab.itos),
                    "seed_vocab_size": len(seed_tokens),
                    "observed_structural_tokens": len(scan_summary.observed_structural_tokens),
                    "oov_token_occurrences": scan_summary.oov_token_occurrences,
                    "oov_unique_tokens": scan_summary.oov_unique_tokens,
                    "train_view_mode": args.train_view_mode,
                    "valid_view_mode": args.valid_view_mode,
                    "loss_skip_special_tokens": bool(args.loss_skip_special_tokens),
                    "reused_scan_cache": reused_scan_cache,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    _distributed_barrier(dev)
    if not _is_main_process():
        scan_summary, vocab = _load_scan_cache_when_ready(out_dir)
        _set_total_counts(scan_summary)
    _validate_single_scale_view_counts(scan_summary, int(args.expected_single_scale_views))

    if args.scan_only:
        _cleanup_distributed()
        return 0
    if scan_summary.train_rows <= 0:
        _cleanup_distributed()
        raise ValueError("No train rows found after shard scan.")
    if not scan_summary.valid_sequences:
        _cleanup_distributed()
        raise ValueError("No validation rows sampled after shard scan.")

    train_ds = _ShardTokenPairDataset(
        train_paths,
        vocab=vocab,
        text_column=args.text_column,
        view_mode=args.train_view_mode,
        max_seq_len=cfg.max_seq_len,
        max_rows=train_max_rows,
        seed=cfg.seed,
        length_bucket_size=int(args.length_bucket_size),
        rank=_dist_rank(),
        world_size=_dist_world_size(),
    )
    valid_sequences_rank = [
        _truncate_ids_keep_eos(seq, max_seq_len=cfg.max_seq_len, eos_id=vocab.eos_id)
        for seq in scan_summary.valid_sequences[_dist_rank():: max(1, _dist_world_size())]
    ]
    valid_ds = NextTokenDataset(valid_sequences_rank, cfg.max_seq_len, vocab.pad_id)

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
    special_loss_ignore_ids = (
        _special_loss_ignore_ids(vocab, multiscale_scale_names)
        if bool(args.loss_skip_special_tokens)
        else set()
    )

    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    use_amp, amp_dtype, use_scaler, _amp_label = _resolve_amp_mode(cfg.amp, dev)
    if args.backbone == "llama":
        model = LlamaDecoderLM(vocab_size=len(vocab.itos), cfg=cfg, tie_embeddings=not bool(args.no_tie_embeddings))
    else:
        model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg)
    resume_payload: dict[str, Any] | None = None
    resume_optimizer_step = 0
    resume_epoch = 0
    if resume_path is not None:
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(_strip_compile_prefix(resume_payload["model_state"]), strict=True)
        resume_optimizer_step = int(resume_payload.get("optimizer_step", 0) or 0)
        resume_epoch = int(resume_payload.get("epoch", 0) or 0)
        if _is_main_process():
            print(
                f"[resume] path={resume_path} epoch={resume_epoch} optimizer_step={resume_optimizer_step}",
                flush=True,
            )
    if cfg.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[misc]
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
    if resume_payload is not None and resume_payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        for state in optimizer.state.values():
            for k, v in list(state.items()):
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(dev)

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)  # pragma: no cover
    if resume_payload is not None and resume_payload.get("scaler_state") is not None:
        try:
            scaler.load_state_dict(resume_payload["scaler_state"])
        except Exception as exc:
            if _is_main_process():
                print(f"[WARN] could not restore scaler state: {exc}", file=sys.stderr, flush=True)

    history: list[dict[str, float]] = list(resume_payload.get("history", [])) if resume_payload is not None else []
    validation_history: list[dict[str, float]] = (
        list(resume_payload.get("validation_history", [])) if resume_payload is not None else []
    )
    non_blocking = dev.type == "cuda"
    dp_world_size = torch.cuda.device_count() if bool(args.data_parallel) and dev.type == "cuda" else 1
    effective_global_batch = int(args.global_batch_size) if int(args.global_batch_size) > 0 else int(cfg.batch_size)
    min_global_batch = int(cfg.batch_size) * max(1, _dist_world_size())
    if effective_global_batch < min_global_batch:
        raise ValueError("global_batch_size must be >= batch_size when provided.")
    grad_accum_steps = max(
        1,
        math.ceil(effective_global_batch / max(1, int(cfg.batch_size) * max(1, _dist_world_size()))),
    )
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
    first_epoch = max(1, resume_epoch + 1) if resume_payload is not None else 1
    last_epoch = max(first_epoch, resume_epoch + int(cfg.epochs)) if resume_payload is not None else int(cfg.epochs)
    total_opt_steps = max(1, train_opt_steps * int(last_epoch))
    if int(args.max_optimizer_steps) > 0:
        total_opt_steps = min(total_opt_steps, int(args.max_optimizer_steps))
    elif resume_payload is not None:
        total_opt_steps = max(1, int(resume_optimizer_step) + train_opt_steps * int(cfg.epochs))
    if resume_payload is not None and int(resume_optimizer_step) >= int(total_opt_steps):
        raise ValueError(
            f"resume optimizer_step={resume_optimizer_step} is already >= total_optimizer_steps={total_opt_steps}; "
            "increase --max-optimizer-steps."
        )
    ckpt_dir = out_dir / "checkpoints"
    if (args.save_every_epoch or int(args.save_every_steps) > 0) and _is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    saved_multiscale_vocab_files: list[str] = []
    if _is_main_process():
        shutil.copy2(stamp_vocab, out_dir / "stamp_vocab.txt")
        if multiscale_vocab_paths:
            for i, src in enumerate(multiscale_vocab_paths):
                ext = Path(src).suffix or ".txt"
                dest_name = f"stamp_vocab_scale_{i:02d}{ext}"
                shutil.copy2(src, out_dir / dest_name)
                saved_multiscale_vocab_files.append(dest_name)
            save_multiscale_manifest(
                out_dir / "multiscale_manifest.json",
                scale_names=multiscale_scale_names,
                vocab_files=saved_multiscale_vocab_files,
                text_column=args.text_column,
            )
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
                    "total_optimizer_steps": int(total_opt_steps),
                    "resume_optimizer_step": int(resume_optimizer_step),
                    "first_epoch": int(first_epoch),
                    "last_epoch": int(last_epoch),
                    "length_bucket_size": int(args.length_bucket_size),
                    "lr_scheduler": args.lr_scheduler,
                    "warmup_steps": int(args.warmup_steps),
                    "min_lr": float(args.min_lr),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    _distributed_barrier(dev)

    def _save_checkpoint(epoch: int, optimizer_step: int, name: str) -> None:
        model_state = _unwrap_model(model).state_dict()
        checkpoint_payload = {
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": (scaler.state_dict() if scaler is not None else None),
            "cfg": vars(cfg),
            "representation": "stamp",
            "model_type": "decoder",
            "backbone": args.backbone,
            "tie_embeddings": (not bool(args.no_tie_embeddings)),
            "stamp_vocab": "stamp_vocab.txt",
            "multiscale_vocab_files": saved_multiscale_vocab_files if _is_main_process() else [],
            "multiscale_scale_names": multiscale_scale_names,
            "multiscale_text_column": args.text_column,
            "train_view_mode": args.train_view_mode,
            "valid_view_mode": args.valid_view_mode,
            "loss_skip_special_tokens": bool(args.loss_skip_special_tokens),
            "lr_scheduler": args.lr_scheduler,
            "warmup_steps": int(args.warmup_steps),
            "min_lr": float(args.min_lr),
            "epoch": int(epoch),
            "optimizer_step": int(optimizer_step),
            "history": history,
            "validation_history": validation_history,
            "world_size": int(_dist_world_size()),
        }
        torch.save(checkpoint_payload, out_dir / "model.pt")
        torch.save(checkpoint_payload, ckpt_dir / name)

    def _run_validation(epoch: int, optimizer_step: int, tag: str) -> float:
        was_training = model.training
        model.eval()
        valid_losses: list[float] = []
        with torch.no_grad():
            for x, y in loader_valid:
                x = x.to(dev, non_blocking=non_blocking)
                y = y.to(dev, non_blocking=non_blocking)
                y_loss = _mask_special_targets(y, special_loss_ignore_ids)
                amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True) if use_amp else nullcontext()
                with amp_ctx:
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_loss.reshape(-1), ignore_index=-100)
                valid_losses.append(float(loss.item()))
        va = float(np.mean(valid_losses)) if valid_losses else float("nan")
        va = _distributed_mean(va, device=dev)
        if was_training:
            model.train()
        if _is_main_process():
            rec = {
                "tag": str(tag),
                "epoch": int(epoch),
                "optimizer_step": int(optimizer_step),
                "valid_loss": float(va),
                "valid_ppl": float(math.exp(va)) if not math.isnan(va) else float("nan"),
            }
            validation_history.append(rec)
            (out_dir / "validation_history.json").write_text(
                json.dumps(validation_history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"[valid-cams-decoder] tag={tag} epoch={epoch} step={optimizer_step} "
                f"valid_loss={rec['valid_loss']:.4f} valid_ppl={rec['valid_ppl']:.2f}",
                flush=True,
            )
        return float(va)

    optimizer_step = int(resume_optimizer_step)
    for epoch in range(int(first_epoch), int(last_epoch) + 1):
        if optimizer_step >= int(total_opt_steps):
            break
        epoch_t0 = time.time()
        train_ds.set_epoch(epoch)
        model.train()
        train_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        progress_total = train_steps
        if int(args.max_optimizer_steps) > 0:
            remaining_opt_steps = max(0, int(total_opt_steps) - int(optimizer_step))
            progress_total = min(train_steps, max(1, remaining_opt_steps * grad_accum_steps))
        iterator = iterable_with_progress(
            loader_train,
            show_progress=(not args.no_tqdm) and _is_main_process(),
            total=progress_total,
            desc=f"Train cams-decoder[stamp] ep{epoch}",
        )
        micro_step = 0
        progress_t0 = time.time()
        progress_examples = 0
        progress_padded_tokens = 0
        recent_losses: deque[float] = deque(maxlen=max(1, int(args.loss_log_window)))
        stop_after_epoch = False
        for x, y in iterator:
            micro_step += 1
            progress_examples += int(x.shape[0])
            progress_padded_tokens += int(x.shape[0] * x.shape[1])
            x = x.to(dev, non_blocking=non_blocking)
            y = y.to(dev, non_blocking=non_blocking)
            y_loss = _mask_special_targets(y, special_loss_ignore_ids)
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True) if use_amp else nullcontext()
            with amp_ctx:
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_loss.reshape(-1), ignore_index=-100)
                loss_for_backward = loss / float(grad_accum_steps)
            if use_scaler:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()
            loss_value = float(loss.item())
            recent_losses.append(loss_value)
            should_step = (micro_step % grad_accum_steps == 0) or (micro_step == train_steps)
            if should_step:
                lr_now = _scheduled_lr(
                    base_lr=float(cfg.lr),
                    min_lr=float(args.min_lr),
                    step=int(optimizer_step),
                    total_steps=int(total_opt_steps),
                    warmup_steps=max(0, int(args.warmup_steps)),
                    scheduler=str(args.lr_scheduler),
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr_now
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if int(args.save_every_steps) > 0 and optimizer_step % int(args.save_every_steps) == 0:
                    if bool(args.validate_on_save):
                        _run_validation(epoch, optimizer_step, f"step_{optimizer_step:08d}")
                    if _is_main_process():
                        _save_checkpoint(epoch, optimizer_step, f"step_{optimizer_step:08d}.pt")
                if optimizer_step >= int(total_opt_steps):
                    stop_after_epoch = True
                    break
            log_every_micro_steps = max(1, int(args.log_every_micro_steps))
            if _is_main_process() and hasattr(iterator, "set_postfix") and micro_step % log_every_micro_steps == 0:
                dt = max(1e-9, time.time() - progress_t0)
                iterator.set_postfix(
                    batch=int(x.shape[0]),
                    seqlen=int(x.shape[1]),
                    loss=f"{loss_value:.4f}",
                    loss_avg=f"{float(np.mean(recent_losses)):.4f}",
                    ex_s=f"{progress_examples / dt:.0f}",
                    tok_s=f"{progress_padded_tokens / dt:.0f}",
                    opt=int(optimizer_step),
                    lr=f"{float(optimizer.param_groups[0]['lr']):.2e}",
                )
            train_losses.append(loss_value)

        train_loss_mean = float(np.mean(train_losses)) if train_losses else float("nan")
        train_loss_mean = _distributed_mean(train_loss_mean, device=dev)

        tr = train_loss_mean
        va = _run_validation(epoch, optimizer_step, f"epoch_{epoch:04d}")
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
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"[train-cams-decoder] epoch={epoch} train_loss={tr:.4f} valid_loss={va:.4f} "
            f"train_ppl={history[-1]['train_ppl']:.2f} valid_ppl={history[-1]['valid_ppl']:.2f} "
            f"epoch_wall_s={epoch_wall_seconds:.1f} train_rows_per_s={train_rows_per_sec:.1f} "
            f"opt_steps_per_s={optimizer_steps_per_sec:.3f}",
            flush=True,
        )
        (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.save_every_epoch:
            _save_checkpoint(epoch, optimizer_step, f"epoch_{epoch:04d}.pt")
        else:
            _save_checkpoint(epoch, optimizer_step, f"step_{optimizer_step:08d}.pt")
        if stop_after_epoch:
            break

    if _is_main_process():
        (out_dir / "vocab.json").write_text(json.dumps(vocab.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        param_iter = _unwrap_model(model).parameters()
        meta = {
            "representation": "stamp",
            "train_size": scan_summary.train_rows,
            "valid_size": scan_summary.valid_rows_total,
            "n_train_encoded": scan_summary.train_rows,
            "n_valid_encoded": scan_summary.valid_rows_sampled,
            "n_train_scanned": scan_summary.train_rows_scanned,
            "n_valid_scanned": scan_summary.valid_rows_scanned,
            "vocab_size": len(vocab.itos),
            "model_type": "decoder",
            "backbone": args.backbone,
            "tie_embeddings": (not bool(args.no_tie_embeddings)),
            "cfg": vars(cfg),
            "device": str(dev),
            "world_size": int(_dist_world_size()),
            "data_parallel_world_size": int(dp_world_size),
            "n_model_params": int(sum(p.numel() for p in param_iter)),
            "micro_batch_size_per_rank": int(micro_batch_report),
            "global_batch_size": int(effective_global_batch),
            "grad_accum_steps": int(grad_accum_steps),
            "train_optimizer_steps_per_epoch_per_rank": int(train_opt_steps),
            "total_optimizer_steps": int(total_opt_steps),
            "resume_from": str(resume_path) if resume_path is not None else "",
            "resume_optimizer_step": int(resume_optimizer_step),
            "first_epoch": int(first_epoch),
            "last_epoch": int(last_epoch),
            "lr_scheduler": args.lr_scheduler,
            "warmup_steps": int(args.warmup_steps),
            "min_lr": float(args.min_lr),
            "padding_mode": "dynamic_batch_max",
            "length_bucket_size": int(args.length_bucket_size),
            "train_effective_max_seq_len": max(scan_summary.train_lengths) if scan_summary.train_lengths else 0,
            "valid_effective_max_seq_len": max(scan_summary.valid_lengths) if scan_summary.valid_lengths else 0,
            "train_token_length_stats": _summarize_lengths(scan_summary.train_lengths),
            "valid_token_length_stats": _summarize_lengths(scan_summary.valid_lengths),
            "train_token_length_histogram": scan_summary.train_hist,
            "valid_token_length_histogram": scan_summary.valid_hist,
            "stamp_vocab": "stamp_vocab.txt",
            "seed_vocab_path": str(stamp_vocab),
            "seed_vocab_size": len(seed_tokens),
            "seed_vocab_header": vocab_header,
            "train_glob": args.train_glob,
            "valid_glob": args.valid_glob,
            "text_column": args.text_column,
            "save_every_epoch": bool(args.save_every_epoch),
            "multiscale_vocab_files": saved_multiscale_vocab_files,
            "multiscale_scale_names": multiscale_scale_names,
            "train_view_mode": args.train_view_mode,
            "valid_view_mode": args.valid_view_mode,
            "loss_skip_special_tokens": bool(args.loss_skip_special_tokens),
            "special_loss_ignore_tokens": [vocab.itos[i] for i in sorted(special_loss_ignore_ids) if 0 <= i < len(vocab.itos)],
            "observed_structural_tokens": scan_summary.observed_structural_tokens,
            "oov_token_occurrences": scan_summary.oov_token_occurrences,
            "oov_unique_tokens": scan_summary.oov_unique_tokens,
            "oov_examples": scan_summary.oov_examples,
        }
        (out_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output_dir": str(out_dir), "history": history, "device": str(dev)}, ensure_ascii=False, indent=2))

    _cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
