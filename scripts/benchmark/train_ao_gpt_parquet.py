#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import shutil
import time
import sys
import tempfile
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from rdkit import Chem, RDLogger
from rdkit.Chem import QED
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset, IterableDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.common import iterable_with_progress  # noqa: E402
from stamp.benchmark.hybrid_vocab import FCSTAMP_CONTROL_TOKENS, HybridVocab  # noqa: E402
from stamp.benchmark.lm import (  # noqa: E402
    LMConfig,
    NextTokenDataset,
    TinyDecoderLM,
    TokenVocab,
    _compute_ao_gpt_loss,
    _compute_ar_loss,
    LlamaDecoderLM,
    _resolve_amp_mode,
    _resolve_torch_device,
    _set_seed,
    build_dynamic_lm_collate_fn,
    sample_model_sequences,
)
from stamp.benchmark.metrics import _load_sa_scorer  # noqa: E402
from stamp.core.stamp_codec import STAMPEncoder  # noqa: E402
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


def _distributed_sum_and_count(
    value_sum: float,
    count: int,
    *,
    device: torch.device,
) -> tuple[float, int]:
    if not dist.is_initialized():
        return float(value_sum), int(count)
    tensor = torch.tensor([float(value_sum), float(count)], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor[0].item()), int(round(float(tensor[1].item())))


def _distributed_barrier(device: torch.device) -> None:
    if not dist.is_initialized():
        return
    if device.type == "cuda" and device.index is not None:
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


def _cycle_forever(loader):
    """Infinitely yield batches, re-iterating the loader when exhausted.

    The train loop runs an EXACT, rank-identical step budget (`train_steps`,
    derived from the global row count / world_size — same on every rank).
    The IterableDataset, after per-shard floor capping, produces slightly
    fewer rows than `train_steps * batch_size`, so a plain `for b in loader`
    ends the epoch via StopIteration *before* the step budget is reached —
    and the exact stopping micro_step can differ per rank (None-skips from
    failed decode vary per shard). That makes one rank reach the epoch-end
    scalar all-reduce while another is still doing gradient-bucket
    all-reduces → permanent NCCL collective desync (watchdog timeout).

    Cycling the loader removes any dependence on the dataset's own length:
    every rank performs precisely `train_steps` forward/backward passes, so
    the per-rank collective count is identical by construction.
    """
    while True:
        produced = False
        for batch in loader:
            produced = True
            yield batch
        if not produced:
            return  # empty loader — never spin forever


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


def _stamp_structural_seed_tokens(max_index: int = 255) -> list[str]:
    """STAMP control tokens that should be representable even if rare.

    The motif vocab file contains motif SMILES, not topology/control tokens.
    Most control tokens are discovered from the scan parquet, but fallback
    decompositions can introduce rare tokens such as ``[Ring]`` that are absent
    from a small scan sample. Seeding them prevents accidental ``<unk>`` labels.
    """
    toks = ["[END]", "[BACK]", "[Ring]", "[S_R]", "[S_S]", "[S_E]", "[S_Z]"]
    toks.extend(f"[J_{i}]" for i in range(max_index + 1))
    toks.extend(f"[B_{i}]" for i in range(max_index + 1))
    toks.extend(FCSTAMP_CONTROL_TOKENS)
    return toks


QED_CONDITION_TOKENS = ["[C_QED_LOW]", "[C_QED_MID]", "[C_QED_HIGH]", "[C_QED_VHIGH]"]
SA_CONDITION_TOKENS = ["[C_SA_LOW]", "[C_SA_MID]", "[C_SA_HIGH]", "[C_SA_UNK]"]


def _is_condition_token(tok: str) -> bool:
    return tok.startswith("[C_") and tok.endswith("]")


def _condition_seed_tokens(condition_mode: str) -> list[str]:
    mode = str(condition_mode or "none").strip().lower()
    if mode == "none":
        return []
    if mode == "qed":
        return list(QED_CONDITION_TOKENS)
    if mode == "qed_sa":
        return list(QED_CONDITION_TOKENS) + list(SA_CONDITION_TOKENS)
    raise ValueError(f"Unsupported condition_mode: {condition_mode!r}")


def _qed_condition_token(qed: float) -> str:
    if qed < 0.4:
        return "[C_QED_LOW]"
    if qed < 0.6:
        return "[C_QED_MID]"
    if qed < 0.8:
        return "[C_QED_HIGH]"
    return "[C_QED_VHIGH]"


def _sa_condition_token(sa_value: Optional[float]) -> str:
    if sa_value is None or not math.isfinite(float(sa_value)):
        return "[C_SA_UNK]"
    if float(sa_value) <= 3.0:
        return "[C_SA_LOW]"
    if float(sa_value) <= 4.0:
        return "[C_SA_MID]"
    return "[C_SA_HIGH]"


def _condition_tokens_for_smiles(
    smiles: Any,
    *,
    condition_mode: str,
    sa_scorer: Any = None,
) -> list[str]:
    mode = str(condition_mode or "none").strip().lower()
    if mode == "none":
        return []
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return []
    try:
        qed_val = float(QED.qed(mol))
    except Exception:
        return []
    out = [_qed_condition_token(qed_val)]
    if mode == "qed_sa":
        sa_val: Optional[float] = None
        if sa_scorer is not None:
            try:
                sa_val = float(sa_scorer.calculateScore(mol))
            except Exception:
                sa_val = None
        out.append(_sa_condition_token(sa_val))
    elif mode != "qed":
        raise ValueError(f"Unsupported condition_mode: {condition_mode!r}")
    return out


def _condition_tokens_from_value(value: Any, *, condition_mode: str) -> list[str]:
    mode = str(condition_mode or "none").strip().lower()
    if mode == "none":
        return []
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw_tokens = [str(tok).strip() for tok in value if str(tok).strip()]
    else:
        raw_tokens = [tok for tok in str(value).replace(",", " ").split() if tok]
    qed = [tok for tok in raw_tokens if tok in QED_CONDITION_TOKENS]
    sa = [tok for tok in raw_tokens if tok in SA_CONDITION_TOKENS]
    if mode == "qed":
        return [qed[0]] if qed else []
    if mode == "qed_sa":
        return ([qed[0], sa[0]] if qed and sa else [])
    raise ValueError(f"Unsupported condition_mode: {condition_mode!r}")


def _condition_tokens_from_row(
    cols: dict[str, list[Any]],
    idx: int,
    *,
    condition_column: str,
    has_condition: bool,
    smiles_column: str,
    has_smiles: bool,
    condition_mode: str,
    sa_scorer: Any = None,
) -> list[str]:
    if str(condition_mode or "none").strip().lower() == "none":
        return []
    if has_condition:
        return _condition_tokens_from_value(cols[condition_column][idx], condition_mode=condition_mode)
    return _condition_tokens_for_smiles(
        cols[smiles_column][idx] if has_smiles else "",
        condition_mode=condition_mode,
        sa_scorer=sa_scorer,
    )


def _default_quality_condition_tokens(condition_mode: str) -> list[str]:
    mode = str(condition_mode or "none").strip().lower()
    if mode == "qed":
        return ["[C_QED_HIGH]"]
    if mode == "qed_sa":
        return ["[C_QED_HIGH]", "[C_SA_LOW]"]
    return []


def _parse_condition_tokens(text: str, *, condition_mode: str = "none") -> list[str]:
    raw = str(text or "").strip()
    if raw:
        return [tok for tok in raw.replace(",", " ").split() if tok]
    return _default_quality_condition_tokens(condition_mode)


def _strip_condition_tokens(tokens: Iterable[str]) -> list[str]:
    return [tok for tok in tokens if not _is_condition_token(tok)]


def _parse_label_start(value: Any, n_tokens: int) -> int:
    try:
        out = int(value)
    except Exception:
        return 0
    return max(0, min(int(n_tokens), out))


def _build_vocab_from_counter(
    counter: Counter[str],
    *,
    seed_tokens: list[str],
) -> TokenVocab:
    vocab = [TokenVocab.PAD, TokenVocab.BOS, TokenVocab.EOS, TokenVocab.UNK]
    added = set(vocab)
    for tok in list(seed_tokens) + _stamp_structural_seed_tokens():
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


ParquetRowKey = tuple[str, int, int]


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
    valid_label_starts: list[int]
    valid_keys: list[ParquetRowKey]


def _resolve_path_specs(sources: Iterable[str]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for spec in sources:
        expanded = sorted(glob.glob(str(spec)))
        if not expanded and os.path.exists(str(spec)):
            expanded = [str(spec)]
        for path in expanded:
            resolved = str(Path(path).resolve())
            if resolved not in seen:
                paths.append(resolved)
                seen.add(resolved)
    return paths


def _infer_split_from_path(path: str, train_split: str, valid_split: str) -> str:
    name = Path(path).name.lower()
    if name.startswith(f"{str(valid_split).lower()}_"):
        return valid_split
    if name.startswith(f"{str(train_split).lower()}_"):
        return train_split
    if name.startswith("val_") and valid_split == "val":
        return valid_split
    if name.startswith("valid_") and valid_split in {"val", "valid"}:
        return valid_split
    if name.startswith("train_"):
        return train_split
    return train_split


def _parquet_read_columns(
    pf: pq.ParquetFile,
    *,
    text_column: str,
    split_column: str,
    require_all_pass: bool,
    smiles_column: str = "",
    require_smiles: bool = False,
    condition_column: str = "",
    require_condition: bool = False,
    label_start_column: str = "",
) -> tuple[list[str], bool, bool, bool, bool]:
    names = set(pf.schema.names)
    if text_column not in names:
        raise KeyError(f"Parquet file is missing text column {text_column!r}; available={sorted(names)}")
    has_split = bool(split_column) and split_column in names
    has_all_pass = bool(require_all_pass) and "all_pass" in names
    has_smiles = bool(smiles_column) and smiles_column in names
    has_condition = bool(condition_column) and condition_column in names
    has_label_start = bool(label_start_column) and label_start_column in names
    if require_smiles and not has_smiles:
        raise KeyError(f"Parquet file is missing smiles column {smiles_column!r}; available={sorted(names)}")
    if require_condition and not has_condition:
        raise KeyError(f"Parquet file is missing condition column {condition_column!r}; available={sorted(names)}")
    columns = [text_column]
    if has_split:
        columns.append(split_column)
    if has_all_pass:
        columns.append("all_pass")
    if has_smiles and smiles_column not in columns:
        columns.append(smiles_column)
    if has_condition and condition_column not in columns:
        columns.append(condition_column)
    if has_label_start and label_start_column not in columns:
        columns.append(label_start_column)
    return columns, has_split, has_all_pass, has_smiles, has_condition


def _scan_cache_path(out_dir: Path) -> Path:
    return out_dir / "_scan_cache.pt"


def _build_scan_cache_config(
    *,
    parquet_paths: Iterable[str],
    text_column: str,
    smiles_column: str,
    condition_column: str,
    label_start_column: str,
    split_column: str,
    train_split: str,
    valid_split: str,
    require_all_pass: bool,
    train_max_rows: Optional[int],
    valid_max_rows: int,
    stamp_vocab: str | Path,
    motif_frequency_file: str | Path | None,
    motif_fallback: bool,
    freq_cutoff: int,
    max_atomic_motifs: int,
    condition_mode: str,
) -> dict[str, Any]:
    return {
        "parquet_paths": [str(Path(p).resolve()) for p in parquet_paths],
        "text_column": str(text_column),
        "smiles_column": str(smiles_column),
        "condition_column": str(condition_column),
        "label_start_column": str(label_start_column),
        "split_column": str(split_column),
        "train_split": str(train_split),
        "valid_split": str(valid_split),
        "require_all_pass": bool(require_all_pass),
        "train_max_rows": None if train_max_rows is None else int(train_max_rows),
        "valid_max_rows": int(valid_max_rows),
        "stamp_vocab": str(Path(stamp_vocab).resolve()),
        "motif_frequency_file": (
            str(Path(motif_frequency_file).resolve()) if motif_frequency_file else ""
        ),
        "motif_fallback": bool(motif_fallback),
        "freq_cutoff": int(freq_cutoff),
        "max_atomic_motifs": int(max_atomic_motifs),
        "condition_mode": str(condition_mode),
    }


def _read_scan_cache_payload(out_dir: Path) -> dict[str, Any]:
    return torch.load(_scan_cache_path(out_dir), map_location="cpu")


def _atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _canonical_smiles(smi: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _hybrid_tokens_to_stamp_tokens(tokens: Iterable[str]) -> list[str]:
    from stamp.benchmark.hybrid_vocab import BOS, EOS, MASK, PAD, UNK, is_fcstamp_control_token, is_stamp_structural

    special = {PAD, BOS, EOS, MASK, UNK}
    out: list[str] = []
    buf: list[str] = []
    for tok in tokens:
        if tok in special or _is_condition_token(tok) or is_fcstamp_control_token(tok):
            continue
        if is_stamp_structural(tok):
            if buf:
                out.append("".join(buf))
                buf = []
            out.append(tok)
        else:
            # Hybrid vocab has no motif-boundary marker; STAMP structural
            # tokens delimit adjacent motif/SMILES-char spans.
            buf.append(tok)
    if buf:
        out.append("".join(buf))
    return out


def _count_high_quality(unique_valid_smiles: Iterable[str], sa: Any) -> int:
    if sa is None:
        return 0
    n_pass = 0
    for smi in unique_valid_smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        try:
            qed_val = float(QED.qed(mol))
            sa_val = float(sa.calculateScore(mol))
        except Exception:
            continue
        if qed_val >= 0.6 and sa_val <= 4.0:
            n_pass += 1
    return n_pass


def _quality_score(metrics: dict[str, Any]) -> float:
    val = metrics.get("quality")
    try:
        out = float(val)
    except Exception:
        return float("-inf")
    return out if math.isfinite(out) else float("-inf")


def _load_quality_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    records = payload.get("checkpoints") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return []
    return [r for r in records if isinstance(r, dict)]


def _write_quality_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"checkpoints": records}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_quality_history(path: Path, metrics: dict[str, Any]) -> None:
    if path.exists():
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                rows = []
        except Exception:
            rows = []
    else:
        rows = []
    rows.append(metrics)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_best_quality_link(out_dir: Path, best_path: Path) -> None:
    link_path = out_dir / "model.pt"
    try:
        if link_path.exists() or link_path.is_symlink():
            if not link_path.is_symlink():
                print(
                    f"[train-ao-gpt-parquet] best_quality_link skipped; regular file exists: {link_path}",
                    flush=True,
                )
                return
            link_path.unlink()
        rel_target = os.path.relpath(best_path, start=out_dir)
        link_path.symlink_to(rel_target)
    except Exception as exc:
        print(f"[train-ao-gpt-parquet] best_quality_link failed: {exc}", flush=True)


def _init_model_weights_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    cfg: LMConfig,
) -> dict[str, Any]:
    """Warm-start model weights only, allowing vocab-row expansion.

    This is for finetuning when extra control tokens are appended to the vocab.
    Shape-compatible parameters load exactly; embedding/head matrices with a
    larger first dimension copy the checkpoint rows and keep new rows randomly
    initialized.
    """
    path = str(checkpoint_path)
    ckpt = torch.load(path, map_location="cpu")
    ckpt_model_type = str(ckpt.get("model_type", cfg.model_type))
    if ckpt_model_type != str(cfg.model_type):
        raise ValueError(f"Init checkpoint model_type={ckpt_model_type!r} != requested {cfg.model_type!r}")
    ckpt_cfg = ckpt.get("cfg") or {}
    if int(ckpt_cfg.get("d_model", cfg.d_model)) != int(cfg.d_model):
        raise ValueError("Init checkpoint d_model does not match requested config.")
    if int(ckpt_cfg.get("n_layers", cfg.n_layers)) != int(cfg.n_layers):
        raise ValueError("Init checkpoint n_layers does not match requested config.")
    if int(ckpt_cfg.get("n_heads", cfg.n_heads)) != int(cfg.n_heads):
        raise ValueError("Init checkpoint n_heads does not match requested config.")

    src = ckpt["model_state"]
    dst = _unwrap_model(model).state_dict()
    loadable: dict[str, torch.Tensor] = {}
    exact = 0
    expanded = 0
    skipped: list[str] = []
    for key, dst_tensor in dst.items():
        src_tensor = src.get(key)
        if not torch.is_tensor(src_tensor):
            skipped.append(key)
            continue
        if tuple(src_tensor.shape) == tuple(dst_tensor.shape):
            loadable[key] = src_tensor
            exact += 1
            continue
        if (
            src_tensor.ndim >= 1
            and dst_tensor.ndim == src_tensor.ndim
            and tuple(src_tensor.shape[1:]) == tuple(dst_tensor.shape[1:])
            and int(src_tensor.shape[0]) <= int(dst_tensor.shape[0])
        ):
            merged = dst_tensor.detach().clone()
            merged[: int(src_tensor.shape[0])] = src_tensor.to(dtype=merged.dtype)
            loadable[key] = merged
            expanded += 1
            continue
        skipped.append(key)
    missing, unexpected = _unwrap_model(model).load_state_dict(loadable, strict=False)
    return {
        "path": path,
        "exact_tensors": int(exact),
        "expanded_tensors": int(expanded),
        "skipped_tensors": skipped[:32],
        "n_skipped_tensors": int(len(skipped)),
        "missing_keys": list(missing)[:32],
        "unexpected_keys": list(unexpected)[:32],
    }


def _scan_cache_matches(out_dir: Path, expected_config: dict[str, Any]) -> bool:
    try:
        payload = _read_scan_cache_payload(out_dir)
    except Exception:
        return False
    cached_config = payload.get("scan_config")
    return isinstance(cached_config, dict) and cached_config == expected_config


def _save_scan_cache(
    out_dir: Path,
    summary: ScanSummary,
    vocab: Any,
    *,
    scan_config: dict[str, Any],
) -> None:
    vocab_type = "hybrid" if isinstance(vocab, HybridVocab) else "token"
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
            "valid_label_starts": summary.valid_label_starts,
            "valid_keys": summary.valid_keys,
        },
        "vocab_type": vocab_type,
        "vocab": vocab.to_dict(),
        "vocab_itos": vocab.itos,
        "scan_config": scan_config,
    }
    cache_path = _scan_cache_path(out_dir)
    with tempfile.NamedTemporaryFile(dir=out_dir, prefix=cache_path.name, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _load_scan_cache(out_dir: Path) -> tuple[ScanSummary, Any]:
    payload = _read_scan_cache_payload(out_dir)
    summary_payload = payload["summary"]
    if payload.get("vocab_type") == "hybrid":
        vocab = HybridVocab.from_dict(payload["vocab"])
    else:
        vocab_itos = list(payload.get("vocab_itos", payload.get("vocab", {}).get("itos", [])))
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
        valid_label_starts=(
            [int(x) for x in summary_payload.get("valid_label_starts", [])]
            if "valid_label_starts" in summary_payload
            else [0 for _ in summary_payload["valid_sequences"]]
        ),
        valid_keys=[
            (str(key[0]), int(key[1]), int(key[2]))
            for key in summary_payload.get("valid_keys", [])
        ],
    )
    return summary, vocab


def _scan_parquet_for_vocab_and_valid(
    parquet_paths: Sequence[str],
    *,
    text_column: str,
    smiles_column: str,
    condition_column: str,
    label_start_column: str,
    split_column: str,
    train_split: str,
    valid_split: str,
    seed_tokens: list[str],
    require_all_pass: bool,
    train_max_rows: Optional[int],
    valid_max_rows: int,
    seed: int,
    show_progress: bool,
    condition_mode: str,
) -> tuple[ScanSummary, TokenVocab]:
    counter: Counter[str] = Counter()
    train_rows = 0
    valid_rows_total = 0
    train_lengths: list[int] = []
    valid_lengths: list[int] = []
    valid_sample_tokens: list[list[str]] = []
    valid_sample_label_starts: list[int] = []
    valid_sample_lengths: list[int] = []
    rng = random.Random(seed + 17)
    mode = str(condition_mode or "none").strip().lower()
    use_condition_column = bool(condition_column) and mode != "none"
    sa_scorer = _load_sa_scorer() if mode == "qed_sa" and not use_condition_column else None
    require_smiles = mode != "none" and not use_condition_column
    require_condition = use_condition_column

    row_groups: list[tuple[str, int]] = []
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        row_groups.extend((path, rg) for rg in range(pf.num_row_groups))
    row_group_iter = iterable_with_progress(
        row_groups,
        show_progress=show_progress,
        total=len(row_groups),
        desc="Scan parquet",
    )
    for path, rg in row_group_iter:
        pf = pq.ParquetFile(path)
        columns, has_split, has_all_pass, has_smiles, has_condition = _parquet_read_columns(
            pf,
            text_column=text_column,
            split_column=split_column,
            require_all_pass=require_all_pass,
            smiles_column=smiles_column,
            require_smiles=require_smiles,
            condition_column=condition_column,
            require_condition=require_condition,
            label_start_column=label_start_column,
        )
        tab = pf.read_row_group(rg, columns=columns)
        cols = {name: tab.column(i).to_pylist() for i, name in enumerate(columns)}
        has_label_start = bool(label_start_column) and label_start_column in cols
        n = len(cols[text_column])
        for i in range(n):
            if has_all_pass and not bool(cols["all_pass"][i]):
                continue
            text = cols[text_column][i]
            if text is None:
                continue
            text = str(text).strip()
            if not text:
                continue
            toks = text.split()
            if not toks:
                continue
            label_start = _parse_label_start(cols[label_start_column][i], len(toks)) if has_label_start else 0
            if mode != "none":
                cond = _condition_tokens_from_row(
                    cols,
                    i,
                    condition_column=condition_column,
                    has_condition=has_condition,
                    smiles_column=smiles_column,
                    has_smiles=has_smiles,
                    condition_mode=condition_mode,
                    sa_scorer=sa_scorer,
                )
                if not cond:
                    continue
                toks = cond + toks
                if has_label_start:
                    label_start += len(cond)
            split = cols[split_column][i] if has_split else _infer_split_from_path(path, train_split, valid_split)
            if split == train_split:
                counter.update(toks)
                train_rows += 1
                train_lengths.append(len(toks))
                if train_max_rows is not None and train_rows >= train_max_rows:
                    break
            elif split == valid_split:
                valid_rows_total += 1
                tok_len = len(toks)
                valid_lengths.append(tok_len)
                if len(valid_sample_tokens) < valid_max_rows:
                    valid_sample_tokens.append(toks)
                    valid_sample_label_starts.append(label_start)
                    valid_sample_lengths.append(tok_len)
                else:
                    j = rng.randint(0, valid_rows_total - 1)
                    if j < valid_max_rows:
                        valid_sample_tokens[j] = toks
                        valid_sample_label_starts[j] = label_start
                        valid_sample_lengths[j] = tok_len
        if train_max_rows is not None and train_rows >= train_max_rows:
            break

    vocab = _build_vocab_from_counter(counter, seed_tokens=seed_tokens)
    valid_sequences = [
        [vocab.bos_id] + vocab.encode_tokens(toks) + [vocab.eos_id]
        for toks in valid_sample_tokens
    ]
    valid_label_starts = (
        [int(x) for x in valid_sample_label_starts]
        if valid_sample_label_starts
        else [0 for _ in valid_sequences]
    )
    summary = ScanSummary(
        train_rows=train_rows,
        valid_rows_total=valid_rows_total,
        valid_rows_sampled=len(valid_sequences),
        train_lengths=train_lengths,
        valid_lengths=valid_sample_lengths,
        train_hist=_histogram(train_lengths),
        valid_hist=_histogram(valid_sample_lengths),
        token_counter=counter,
        valid_sequences=valid_sequences,
        valid_label_starts=valid_label_starts,
        valid_keys=[],
    )
    return summary, vocab


def _scan_parquet_for_hybrid_valid(
    parquet_paths: Sequence[str],
    *,
    text_column: str,
    smiles_column: str,
    condition_column: str,
    label_start_column: str,
    split_column: str,
    train_split: str,
    valid_split: str,
    vocab: HybridVocab,
    require_all_pass: bool,
    train_max_rows: Optional[int],
    valid_max_rows: int,
    seed: int,
    show_progress: bool,
    condition_mode: str,
) -> tuple[ScanSummary, HybridVocab]:
    train_rows = 0
    valid_rows_total = 0
    train_lengths: list[int] = []
    valid_lengths_all: list[int] = []
    valid_sample_ids: list[list[int]] = []
    valid_sample_label_starts: list[int] = []
    valid_sample_lengths: list[int] = []
    rng = random.Random(seed + 17)
    mode = str(condition_mode or "none").strip().lower()
    use_condition_column = bool(condition_column) and mode != "none"
    sa_scorer = _load_sa_scorer() if mode == "qed_sa" and not use_condition_column else None
    require_smiles = mode != "none" and not use_condition_column
    require_condition = use_condition_column

    row_groups: list[tuple[str, int]] = []
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        row_groups.extend((path, rg) for rg in range(pf.num_row_groups))
    row_group_iter = iterable_with_progress(
        row_groups,
        show_progress=show_progress,
        total=len(row_groups),
        desc="Scan parquet (hybrid)",
    )
    for path, rg in row_group_iter:
        pf = pq.ParquetFile(path)
        columns, has_split, has_all_pass, has_smiles, has_condition = _parquet_read_columns(
            pf,
            text_column=text_column,
            split_column=split_column,
            require_all_pass=require_all_pass,
            smiles_column=smiles_column,
            require_smiles=require_smiles,
            condition_column=condition_column,
            require_condition=require_condition,
            label_start_column=label_start_column,
        )
        tab = pf.read_row_group(rg, columns=columns)
        cols = {name: tab.column(i).to_pylist() for i, name in enumerate(columns)}
        has_label_start = bool(label_start_column) and label_start_column in cols
        n = len(cols[text_column])
        for i in range(n):
            if has_all_pass and not bool(cols["all_pass"][i]):
                continue
            text = cols[text_column][i]
            if text is None:
                continue
            toks = str(text).strip().split()
            if not toks:
                continue
            label_start = _parse_label_start(cols[label_start_column][i], len(toks)) if has_label_start else 0
            if mode != "none":
                cond = _condition_tokens_from_row(
                    cols,
                    i,
                    condition_column=condition_column,
                    has_condition=has_condition,
                    smiles_column=smiles_column,
                    has_smiles=has_smiles,
                    condition_mode=condition_mode,
                    sa_scorer=sa_scorer,
                )
                if not cond:
                    continue
                toks = cond + toks
                if has_label_start:
                    label_start += len(cond)
            ids = [vocab.bos_id] + vocab.encode_stream(toks, training=False) + [vocab.eos_id]
            encoded_label_start = (
                len(vocab.encode_stream(toks[:label_start], training=False))
                if has_label_start
                else 0
            )
            seq_len = len(ids)
            split = cols[split_column][i] if has_split else _infer_split_from_path(path, train_split, valid_split)
            if split == train_split:
                train_rows += 1
                train_lengths.append(seq_len)
                if train_max_rows is not None and train_rows >= train_max_rows:
                    break
            elif split == valid_split:
                valid_rows_total += 1
                valid_lengths_all.append(seq_len)
                if len(valid_sample_ids) < valid_max_rows:
                    valid_sample_ids.append(ids)
                    valid_sample_label_starts.append(encoded_label_start)
                    valid_sample_lengths.append(seq_len)
                else:
                    j = rng.randint(0, valid_rows_total - 1)
                    if j < valid_max_rows:
                        valid_sample_ids[j] = ids
                        valid_sample_label_starts[j] = encoded_label_start
                        valid_sample_lengths[j] = seq_len
        if train_max_rows is not None and train_rows >= train_max_rows:
            break

    summary = ScanSummary(
        train_rows=train_rows,
        valid_rows_total=valid_rows_total,
        valid_rows_sampled=len(valid_sample_ids),
        train_lengths=train_lengths,
        valid_lengths=valid_sample_lengths,
        train_hist=_histogram(train_lengths),
        valid_hist=_histogram(valid_sample_lengths),
        token_counter=Counter(),
        valid_sequences=valid_sample_ids,
        valid_label_starts=(
            [int(x) for x in valid_sample_label_starts]
            if valid_sample_label_starts
            else [0 for _ in valid_sample_ids]
        ),
        valid_keys=[],
    )
    return summary, vocab



class _NextTokenDatasetWithLabelStart(Dataset[tuple[list[int], list[int]]]):
    def __init__(
        self,
        sequences: Sequence[list[int]],
        label_starts: Sequence[int],
        max_seq_len: int,
    ) -> None:
        self.samples: list[tuple[list[int], list[int]]] = []
        for seq, label_start in zip(sequences, label_starts):
            if len(seq) < 2:
                continue
            ids = list(seq[: max_seq_len + 1])
            x = ids[:-1]
            y = ids[1:]
            if not x:
                continue
            ignore = max(0, min(len(y), int(label_start)))
            if ignore > 0:
                y = [-100] * ignore + y[ignore:]
            if not any(v != -100 for v in y):
                continue
            self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.samples[idx]


class _ParquetTokenPairDataset(IterableDataset[tuple[list[int], list[int]]]):
    def __init__(
        self,
        parquet_paths: Sequence[str],
        *,
        vocab: Any,
        text_column: str,
        smiles_column: str,
        condition_column: str,
        label_start_column: str,
        split_column: str,
        split_value: str,
        require_all_pass: bool,
        max_seq_len: int,
        max_rows: Optional[int],
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        training_fallback: bool = False,
        fallback_p_low: float = 0.02,
        fallback_p_high: float = 0.15,
        condition_mode: str = "none",
        exclude_keys: Sequence[ParquetRowKey] | None = None,
    ) -> None:
        super().__init__()
        self.parquet_paths = [str(Path(p).resolve()) for p in parquet_paths]
        self.vocab = vocab
        self.text_column = text_column
        self.smiles_column = smiles_column
        self.condition_column = condition_column
        self.label_start_column = label_start_column
        self.split_column = split_column
        self.split_value = split_value
        self.require_all_pass = bool(require_all_pass)
        self.max_seq_len = int(max_seq_len)
        self.max_rows = max_rows
        self.seed = int(seed)
        self.epoch = 0
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.training_fallback = bool(training_fallback)
        self.fallback_p_low = float(fallback_p_low)
        self.fallback_p_high = float(fallback_p_high)
        self.condition_mode = str(condition_mode or "none").strip().lower()
        self.exclude_keys = {
            (str(Path(k[0]).resolve()), int(k[1]), int(k[2]))
            for k in (exclude_keys or [])
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[tuple[list[int], list[int]]]:
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)
        shard_id = self.rank * num_workers + worker_id
        num_shards = self.world_size * num_workers

        row_groups: list[tuple[str, int]] = []
        for path in self.parquet_paths:
            pf = pq.ParquetFile(path)
            row_groups.extend((path, rg) for rg in range(pf.num_row_groups))
        rng = random.Random(self.seed + self.epoch)
        fallback_rng = random.Random(self.seed * 131 + self.epoch * 17 + shard_id)
        use_condition_column = bool(self.condition_column) and self.condition_mode != "none"
        sa_scorer = _load_sa_scorer() if self.condition_mode == "qed_sa" and not use_condition_column else None
        require_smiles = self.condition_mode != "none" and not use_condition_column
        require_condition = use_condition_column
        rng.shuffle(row_groups)
        row_groups = row_groups[shard_id::num_shards]

        if self.max_rows is not None:
            base = self.max_rows // num_shards
            rem = self.max_rows % num_shards
            worker_limit = base + (1 if shard_id < rem else 0)
        else:
            worker_limit = None

        yielded = 0
        for path, rg in row_groups:
            pf = pq.ParquetFile(path)
            columns, has_split, has_all_pass, has_smiles, has_condition = _parquet_read_columns(
                pf,
                text_column=self.text_column,
                split_column=self.split_column,
                require_all_pass=self.require_all_pass,
                smiles_column=self.smiles_column,
                require_smiles=require_smiles,
                condition_column=self.condition_column,
                require_condition=require_condition,
                label_start_column=self.label_start_column,
            )
            tab = pf.read_row_group(rg, columns=columns)
            cols = {name: tab.column(i).to_pylist() for i, name in enumerate(columns)}
            has_label_start = bool(self.label_start_column) and self.label_start_column in cols
            order = list(range(len(cols[self.text_column])))
            rng.shuffle(order)
            for idx in order:
                if (str(Path(path).resolve()), int(rg), int(idx)) in self.exclude_keys:
                    continue
                split = (
                    cols[self.split_column][idx]
                    if has_split
                    else _infer_split_from_path(path, self.split_value, "val")
                )
                if split != self.split_value:
                    continue
                if has_all_pass and not bool(cols["all_pass"][idx]):
                    continue
                text = cols[self.text_column][idx]
                if text is None:
                    continue
                toks = str(text).split()
                if not toks:
                    continue
                label_start = (
                    _parse_label_start(cols[self.label_start_column][idx], len(toks))
                    if has_label_start
                    else 0
                )
                if self.condition_mode != "none":
                    cond = _condition_tokens_from_row(
                        cols,
                        idx,
                        condition_column=self.condition_column,
                        has_condition=has_condition,
                        smiles_column=self.smiles_column,
                        has_smiles=has_smiles,
                        condition_mode=self.condition_mode,
                        sa_scorer=sa_scorer,
                    )
                    if not cond:
                        continue
                    toks = cond + toks
                    if has_label_start:
                        label_start += len(cond)
                if hasattr(self.vocab, "encode_stream"):
                    if has_label_start:
                        prompt_body = self.vocab.encode_stream(
                            toks[:label_start],
                            training=self.training_fallback,
                            rng=fallback_rng,
                            p_low=self.fallback_p_low,
                            p_high=self.fallback_p_high,
                        )
                        target_body = self.vocab.encode_stream(
                            toks[label_start:],
                            training=self.training_fallback,
                            rng=fallback_rng,
                            p_low=self.fallback_p_low,
                            p_high=self.fallback_p_high,
                        )
                        body = prompt_body + target_body
                        encoded_label_start = len(prompt_body)
                    else:
                        body = self.vocab.encode_stream(
                            toks,
                            training=self.training_fallback,
                            rng=fallback_rng,
                            p_low=self.fallback_p_low,
                            p_high=self.fallback_p_high,
                        )
                        encoded_label_start = 0
                else:
                    if has_label_start:
                        prompt_body = self.vocab.encode_tokens(toks[:label_start])
                        target_body = self.vocab.encode_tokens(toks[label_start:])
                        body = prompt_body + target_body
                        encoded_label_start = len(prompt_body)
                    else:
                        body = self.vocab.encode_tokens(toks)
                        encoded_label_start = 0
                ids = [self.vocab.bos_id] + body + [self.vocab.eos_id]
                if len(ids) < 2:
                    continue
                x = ids[:-1][: self.max_seq_len]
                y = ids[1:][: self.max_seq_len]
                ignore = max(0, min(len(y), int(encoded_label_start)))
                if ignore > 0:
                    y = [-100] * ignore + y[ignore:]
                if not any(v != -100 for v in y):
                    continue
                if not x:
                    continue
                yield x, y
                yielded += 1
                if worker_limit is not None and yielded >= worker_limit:
                    return


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Large-scale AO-GPT pretraining directly from parquet stamp sequences.")
    p.add_argument("--parquet", required=True, help="Parquet path or glob. Multi-shard STAMP parquet is supported.")
    p.add_argument("--stamp-vocab", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--text-column", default="stamp_smiles")
    p.add_argument("--smiles-column", default="smiles")
    p.add_argument(
        "--label-start-column",
        default="",
        help="Optional prompt token-count column. If set, AR/finetune loss ignores tokens before this target boundary.",
    )
    p.add_argument(
        "--condition-column",
        default="",
        help="Optional precomputed condition-token column. If set, condition training reads this column instead of computing QED/SA from SMILES online.",
    )
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
    p.add_argument("--save-latest-model", action="store_true", default=True)
    p.add_argument("--no-save-latest-model", dest="save_latest_model", action="store_false")
    p.add_argument(
        "--save-every-steps",
        type=int,
        default=0,
        help="Save training checkpoints every N optimizer steps. 0 disables step checkpoints.",
    )
    p.add_argument(
        "--keep-last-step-checkpoints",
        type=int,
        default=0,
        help="If >0, keep only the most recent N step_*.pt checkpoints.",
    )
    p.add_argument(
        "--checkpoint-top-k-by-quality",
        type=int,
        default=0,
        help="If >0, evaluate sampling quality at each step checkpoint and keep only the top K by quality.",
    )
    p.add_argument("--quality-eval-samples", type=int, default=0, help="Samples per checkpoint quality eval. 0 uses 1024 when top-K quality saving is enabled.")
    p.add_argument("--quality-eval-batch-size", type=int, default=256)
    p.add_argument("--quality-eval-temperature", type=float, default=0.7)
    p.add_argument("--quality-eval-top-k", type=int, default=0)
    p.add_argument("--quality-eval-top-p", type=float, default=0.95)
    p.add_argument("--quality-eval-max-new", type=int, default=0, help="0 = max_seq_len - 1.")
    p.add_argument(
        "--condition-mode",
        default="none",
        choices=["none", "qed", "qed_sa"],
        help="Prepend discrete property condition tokens computed from --smiles-column.",
    )
    p.add_argument(
        "--quality-condition-tokens",
        default="",
        help="Prompt tokens for checkpoint quality eval. Empty uses a high-quality default for --condition-mode.",
    )
    p.add_argument(
        "--resume-from",
        default="",
        help="Resume model/optimizer/scaler/global_step from a training checkpoint.",
    )
    p.add_argument(
        "--init-from",
        default="",
        help="Warm-start model weights only from a checkpoint; optimizer/step are reset and vocab expansion is allowed.",
    )
    p.add_argument(
        "--reuse-scan-cache",
        action="store_true",
        help="Reuse existing output_dir/_scan_cache.pt instead of rescanning parquet on rank 0.",
    )
    p.add_argument("--debug-phase-prints", action="store_true", help="Print rank/phase markers for debugging distributed hangs.")
    p.add_argument(
        "--model-type",
        default="ao_gpt",
        choices=["ao_gpt", "llama"],
        help="ao_gpt = TinyDecoderLM with AdaLN/target_pos_emb (random AO orders + AR). "
             "llama = LlamaDecoderLM clean AR (RoPE + RMSNorm + SwiGLU + tied emb, no biases).",
    )
    p.add_argument(
        "--ao-random-ratio",
        type=float,
        default=0.9,
        help="For ao_gpt only: fraction of rows using random any-order permutation; 0 = pure AR, 1 = fully random AO.",
    )
    p.add_argument(
        "--motif-fallback",
        action="store_true",
        help="Use HybridVocab motif->SMILES fallback tokenization: high-frequency "
             "motifs can stay atomic, rare motifs expand to SMILES regex tokens, "
             "and training randomly expands atomic motifs.",
    )
    p.add_argument("--freq-cutoff", type=int, default=5000)
    p.add_argument(
        "--max-atomic-motifs",
        type=int,
        default=0,
        help="If >0, keep only the top-K atomic motifs by frequency after applying --freq-cutoff.",
    )
    p.add_argument(
        "--motif-frequency-file",
        default="",
        help="Optional motif frequency table used for cutoff decisions. "
             "Supports motif-vocab format (smi, heavy_atoms, freq) and "
             "full-frequency format (smi, freq, in_source_vocab).",
    )
    p.add_argument("--fallback-p-low", type=float, default=0.02)
    p.add_argument("--fallback-p-high", type=float, default=0.15)
    p.add_argument("--no-random-motif-fallback", dest="training_fallback", action="store_false", default=True)
    return p


def main() -> int:
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")
    debug = bool(args.debug_phase_prints)

    parquet_paths = _resolve_path_specs([args.parquet])
    if not parquet_paths:
        raise FileNotFoundError(args.parquet)
    stamp_vocab = Path(args.stamp_vocab)
    if not stamp_vocab.exists():
        raise FileNotFoundError(stamp_vocab)
    motif_frequency_file = Path(args.motif_frequency_file) if str(args.motif_frequency_file).strip() else None
    if motif_frequency_file is not None and not motif_frequency_file.exists():
        raise FileNotFoundError(motif_frequency_file)

    output_dir = resolve_branch_output_dir(args.output_dir, stamp_vocab=str(stamp_vocab))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    local_rank = _dist_local_rank()
    cfg = LMConfig(
        model_type=str(args.model_type),
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
        ao_random_ratio=float(args.ao_random_ratio),
    )
    _set_seed(cfg.seed)
    if args.data_parallel and _is_dist():
        raise ValueError("Cannot combine --data-parallel with torchrun/DDP.")
    if _is_dist() and str(args.device).startswith("cuda"):
        dev = _resolve_torch_device(f"cuda:{local_rank}")
    else:
        dev = _resolve_torch_device(args.device)
    if dev.type == "cuda" and dev.index is None:
        dev = _resolve_torch_device(f"cuda:{local_rank}")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    _debug_print(debug, f"[debug rank={_dist_rank()}] device={dev} world_size={_dist_world_size()} local_rank={local_rank}")
    _setup_distributed(dev)
    _debug_print(debug, f"[debug rank={_dist_rank()}] distributed_initialized={dist.is_initialized()}")

    vocab_header, seed_tokens = _load_vocab_seed_tokens(stamp_vocab)
    condition_mode = str(args.condition_mode or "none").strip().lower()
    condition_tokens = _condition_seed_tokens(condition_mode)
    seed_tokens = list(seed_tokens) + condition_tokens
    extra_tokens = _stamp_structural_seed_tokens() + condition_tokens
    hybrid_seed_vocab = (
        HybridVocab.build(
            str(stamp_vocab),
            freq_cutoff=int(args.freq_cutoff),
            extra_stamp_tokens=extra_tokens,
            frequency_path=motif_frequency_file,
            max_atomic_motifs=int(args.max_atomic_motifs),
        )
        if bool(args.motif_fallback)
        else None
    )
    train_max_rows = (None if int(args.train_max_rows) <= 0 else int(args.train_max_rows))
    scan_cache_config = _build_scan_cache_config(
        parquet_paths=parquet_paths,
        text_column=args.text_column,
        smiles_column=args.smiles_column,
        condition_column=args.condition_column,
        label_start_column=args.label_start_column,
        split_column=args.split_column,
        train_split=args.train_split,
        valid_split=args.valid_split,
        require_all_pass=bool(args.require_all_pass),
        train_max_rows=train_max_rows,
        valid_max_rows=int(args.valid_max_rows),
        stamp_vocab=stamp_vocab,
        motif_frequency_file=motif_frequency_file,
        motif_fallback=bool(args.motif_fallback),
        freq_cutoff=int(args.freq_cutoff),
        max_atomic_motifs=int(args.max_atomic_motifs),
        condition_mode=condition_mode,
    )
    if _is_main_process():
        cache_path = _scan_cache_path(out_dir)
        reused_scan_cache = False
        if (
            bool(args.reuse_scan_cache)
            and cache_path.exists()
            and _scan_cache_matches(out_dir, scan_cache_config)
        ):
            _debug_print(debug, f"[debug rank={_dist_rank()}] reuse_scan_cache path={cache_path}")
            scan_summary, vocab = _load_scan_cache(out_dir)
            reused_scan_cache = True
        else:
            _debug_print(debug, f"[debug rank={_dist_rank()}] begin_scan")
            if hybrid_seed_vocab is not None:
                scan_summary, vocab = _scan_parquet_for_hybrid_valid(
                    parquet_paths=parquet_paths,
                    text_column=args.text_column,
                    smiles_column=args.smiles_column,
                    condition_column=args.condition_column,
                    label_start_column=args.label_start_column,
                    split_column=args.split_column,
                    train_split=args.train_split,
                    valid_split=args.valid_split,
                    vocab=hybrid_seed_vocab,
                    require_all_pass=bool(args.require_all_pass),
                    train_max_rows=train_max_rows,
                    valid_max_rows=int(args.valid_max_rows),
                    seed=cfg.seed,
                    show_progress=not args.no_tqdm,
                    condition_mode=condition_mode,
                )
            else:
                scan_summary, vocab = _scan_parquet_for_vocab_and_valid(
                    parquet_paths=parquet_paths,
                    text_column=args.text_column,
                    smiles_column=args.smiles_column,
                    condition_column=args.condition_column,
                    label_start_column=args.label_start_column,
                    split_column=args.split_column,
                    train_split=args.train_split,
                    valid_split=args.valid_split,
                    seed_tokens=seed_tokens,
                    require_all_pass=bool(args.require_all_pass),
                    train_max_rows=train_max_rows,
                    valid_max_rows=int(args.valid_max_rows),
                    seed=cfg.seed,
                    show_progress=not args.no_tqdm,
                    condition_mode=condition_mode,
                )
            _save_scan_cache(out_dir, scan_summary, vocab, scan_config=scan_cache_config)
            _debug_print(debug, f"[debug rank={_dist_rank()}] end_scan saved_cache")
        print(
            json.dumps(
                {
                    "train_rows": scan_summary.train_rows,
                    "valid_rows_total": scan_summary.valid_rows_total,
                    "valid_rows_sampled": scan_summary.valid_rows_sampled,
                    "vocab_size": len(vocab.itos),
                    "seed_vocab_size": len(seed_tokens),
                    "motif_fallback": bool(args.motif_fallback),
                    "motif_frequency_file": str(motif_frequency_file) if motif_frequency_file else "",
                    "max_atomic_motifs": int(args.max_atomic_motifs),
                    "condition_mode": condition_mode,
                    "condition_tokens": condition_tokens,
                    "atomic_motifs": int(len(vocab.atomic_motifs)) if isinstance(vocab, HybridVocab) else None,
                    "world_size": _dist_world_size(),
                    "reused_scan_cache": reused_scan_cache,
                    "valid_source": "parquet",
                    "valid_excluded_rows": len(scan_summary.valid_keys),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    if not _is_main_process():
        cache_path = _scan_cache_path(out_dir)
        _debug_print(debug, f"[debug rank={_dist_rank()}] wait_cache path={cache_path}")
        while not cache_path.exists() or not _scan_cache_matches(out_dir, scan_cache_config):
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
        parquet_paths=parquet_paths,
        vocab=vocab,
        text_column=args.text_column,
        smiles_column=args.smiles_column,
        condition_column=args.condition_column,
        label_start_column=args.label_start_column,
        split_column=args.split_column,
        split_value=args.train_split,
        require_all_pass=bool(args.require_all_pass),
        max_seq_len=cfg.max_seq_len,
        max_rows=train_max_rows,
        seed=cfg.seed,
        rank=_dist_rank(),
        world_size=_dist_world_size(),
        training_fallback=bool(args.motif_fallback and args.training_fallback),
        fallback_p_low=float(args.fallback_p_low),
        fallback_p_high=float(args.fallback_p_high),
        condition_mode=condition_mode,
        exclude_keys=scan_summary.valid_keys,
    )
    valid_sequences_rank = scan_summary.valid_sequences[_dist_rank():: max(1, _dist_world_size())]
    valid_label_starts_rank = scan_summary.valid_label_starts[_dist_rank():: max(1, _dist_world_size())]
    if args.label_start_column:
        valid_ds = _NextTokenDatasetWithLabelStart(valid_sequences_rank, valid_label_starts_rank, cfg.max_seq_len)
    else:
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
    is_llama = str(cfg.model_type).strip().lower() == "llama"
    if is_llama:
        model = LlamaDecoderLM(vocab_size=len(vocab.itos), cfg=cfg)
        loss_fn = lambda m, x, y: _compute_ar_loss(m, x, y)
        _debug_print(debug, "[debug] using LlamaDecoderLM (pure AR, RoPE+RMSNorm+SwiGLU, tied emb)")
    else:
        model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg)
        loss_fn = lambda m, x, y: _compute_ao_gpt_loss(m, x, y, cfg, random_ratio=cfg.ao_random_ratio)
    _debug_print(debug, f"[debug rank={_dist_rank()}] model_created")
    init_path = str(args.init_from or "").strip()
    if init_path and str(args.resume_from or "").strip():
        raise ValueError("Use either --init-from or --resume-from, not both.")
    init_summary: Optional[dict[str, Any]] = None
    if init_path:
        init_summary = _init_model_weights_from_checkpoint(model, init_path, cfg)
        if _is_main_process():
            print(f"[train-ao-gpt-parquet] initialized_from={json.dumps(init_summary, ensure_ascii=False)}", flush=True)
    if cfg.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[misc]
        except Exception:
            pass
    model.to(dev)
    _debug_print(debug, f"[debug rank={_dist_rank()}] model_to_device_done")
    if _is_dist():
        _debug_print(debug, f"[debug rank={_dist_rank()}] before_ddp_wrap")
        # broadcast_buffers=False: our RoPE cos/sin buffers (LlamaDecoderLM) are
        # deterministic-identical across ranks, so per-forward broadcast is a
        # waste of NCCL bandwidth AND a desync risk (different ranks reaching
        # the broadcast at different timings causes NCCL watchdog timeouts).
        # static_graph=True: required for tied weights (LlamaDecoderLM ties
        # head.weight = tok_emb.weight). Without it, DDP's reducer double-counts
        # the shared parameter's gradient bucket, producing a different number
        # of all-reduce ops per rank → NCCL desync mid-epoch (worse with bigger
        # models where timing drift exposes it). static_graph also lets DDP
        # skip the unused-param trace each step.
        model = DDP(
            model,
            device_ids=[dev.index] if dev.type == "cuda" and dev.index is not None else None,
            output_device=(dev.index if dev.type == "cuda" else None),
            broadcast_buffers=False,
            static_graph=True,
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
    resume_epoch = 1
    resume_step_in_epoch = 0
    resume_global_step = 0
    resume_path = str(args.resume_from or "").strip()
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        ckpt_model_type = str(ckpt.get("model_type", cfg.model_type))
        if ckpt_model_type != str(cfg.model_type):
            raise ValueError(f"Resume checkpoint model_type={ckpt_model_type!r} != requested {cfg.model_type!r}")
        ckpt_cfg = ckpt.get("cfg") or {}
        if int(ckpt_cfg.get("d_model", cfg.d_model)) != int(cfg.d_model):
            raise ValueError("Resume checkpoint d_model does not match requested config.")
        if int(ckpt_cfg.get("n_layers", cfg.n_layers)) != int(cfg.n_layers):
            raise ValueError("Resume checkpoint n_layers does not match requested config.")
        if int(ckpt_cfg.get("n_heads", cfg.n_heads)) != int(cfg.n_heads):
            raise ValueError("Resume checkpoint n_heads does not match requested config.")
        _unwrap_model(model).load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(dev)
        if scaler is not None and ckpt.get("scaler_state") is not None:
            try:
                scaler.load_state_dict(ckpt["scaler_state"])
            except Exception:
                pass
        history = list(ckpt.get("history") or [])
        resume_epoch = max(1, int(ckpt.get("epoch", 1)))
        resume_step_in_epoch = max(0, int(ckpt.get("step_in_epoch", 0)))
        resume_global_step = max(0, int(ckpt.get("global_step", 0)))
        if _is_main_process():
            print(
                f"[train-ao-gpt-parquet] resumed_from={resume_path} "
                f"epoch={resume_epoch} step_in_epoch={resume_step_in_epoch} "
                f"global_step={resume_global_step}",
                flush=True,
            )
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
    epoch_rows = scan_summary.train_rows
    train_rows_rank = math.ceil(epoch_rows / max(1, _dist_world_size()))
    train_steps = max(1, math.ceil(train_rows_rank / max(1, cfg.batch_size)))
    train_opt_steps = max(1, math.ceil(train_rows_rank / max(1, int(cfg.batch_size) * grad_accum_steps)))
    ckpt_dir = out_dir / "checkpoints"
    save_every_steps = max(0, int(args.save_every_steps))
    keep_last_step_checkpoints = max(0, int(args.keep_last_step_checkpoints))
    checkpoint_top_k_by_quality = max(0, int(args.checkpoint_top_k_by_quality))
    quality_eval_samples = int(args.quality_eval_samples)
    if checkpoint_top_k_by_quality > 0 and quality_eval_samples <= 0:
        quality_eval_samples = 1024
    quality_eval_batch_size = max(1, int(args.quality_eval_batch_size))
    quality_eval_max_new = (
        max(1, int(args.quality_eval_max_new))
        if int(args.quality_eval_max_new) > 0
        else max(1, int(cfg.max_seq_len) - 1)
    )
    quality_condition_tokens = _parse_condition_tokens(
        str(args.quality_condition_tokens),
        condition_mode=condition_mode,
    )
    quality_records_path = ckpt_dir / "quality_checkpoints.json"
    quality_history_path = ckpt_dir / "quality_eval_history.json"
    if (args.save_every_epoch or save_every_steps > 0) and _is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_payload(
        *,
        epoch: int,
        global_step: int,
        step_in_epoch: int,
        quality_eval: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return {
            "model_state": _unwrap_model(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": (scaler.state_dict() if scaler is not None else None),
            "cfg": vars(cfg),
            "representation": "stamp",
            "model_type": str(cfg.model_type),
            "stamp_vocab": "stamp_vocab.txt",
            "vocab_type": "hybrid" if isinstance(vocab, HybridVocab) else "token",
            "motif_fallback": bool(args.motif_fallback),
            "condition_mode": condition_mode,
            "condition_tokens": condition_tokens,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "step_in_epoch": int(step_in_epoch),
            "history": history,
            "quality_eval": quality_eval,
            "world_size": int(_dist_world_size()),
        }

    def _evaluate_sampling_quality(*, epoch: int, global_step: int, step_in_epoch: int) -> dict[str, Any]:
        n_samples = max(0, int(quality_eval_samples))
        metrics: dict[str, Any] = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "step_in_epoch": int(step_in_epoch),
            "num_samples": int(n_samples),
            "temperature": float(args.quality_eval_temperature),
            "top_k": int(args.quality_eval_top_k),
            "top_p": float(args.quality_eval_top_p),
            "max_new": int(quality_eval_max_new),
            "condition_tokens": list(quality_condition_tokens),
        }
        if n_samples <= 0:
            metrics.update(
                {
                    "n_valid": 0,
                    "n_unique": 0,
                    "n_high_quality_unique": 0,
                    "validity": float("nan"),
                    "uniqueness": float("nan"),
                    "quality": float("nan"),
                }
            )
            return metrics

        unwrapped = _unwrap_model(model)
        was_training = bool(model.training)
        model.eval()
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if dev.type == "cuda" else None
        decoder = STAMPEncoder(motif_tokenizer=None)
        sample_vocab = (
            vocab
            if hasattr(vocab, "decode_ids")
            else TokenVocab(stoi=dict(vocab.stoi), itos=list(vocab.itos))
        )
        sa = _load_sa_scorer()
        valid_smiles: list[str] = []
        unique_seen: dict[str, None] = {}
        attempts = 0
        try:
            torch.manual_seed(int(cfg.seed) + int(global_step))
            if dev.type == "cuda":
                torch.cuda.manual_seed_all(int(cfg.seed) + int(global_step))
            with torch.no_grad():
                while attempts < n_samples:
                    cur_n = min(quality_eval_batch_size, n_samples - attempts)
                    seqs = sample_model_sequences(
                        unwrapped,
                        sample_vocab,
                        n_samples=cur_n,
                        max_new_tokens=quality_eval_max_new,
                        temperature=float(args.quality_eval_temperature),
                        top_k=int(args.quality_eval_top_k),
                        top_p=float(args.quality_eval_top_p),
                        prompt_tokens=quality_condition_tokens,
                        device=dev,
                        ao_decode_mode="ar",
                    )
                    for toks in seqs:
                        try:
                            stamp_toks = (
                                _hybrid_tokens_to_stamp_tokens(toks)
                                if isinstance(vocab, HybridVocab)
                                else _strip_condition_tokens(toks)
                            )
                            mol = decoder.decode_stamp_to_mol(" ".join(stamp_toks))
                            if len(Chem.GetMolFrags(mol)) != 1:
                                continue
                            can = _canonical_smiles(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
                        except Exception:
                            can = None
                        if can:
                            valid_smiles.append(can)
                            unique_seen.setdefault(can, None)
                    attempts += cur_n
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if dev.type == "cuda" and cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            if was_training:
                model.train()

        unique_valid = list(unique_seen.keys())
        n_quality = _count_high_quality(unique_valid, sa)
        metrics.update(
            {
                "num_attempts": int(attempts),
                "n_valid": int(len(valid_smiles)),
                "n_unique": int(len(unique_valid)),
                "n_high_quality_unique": int(n_quality),
                "validity": 100.0 * float(len(valid_smiles)) / max(1, attempts),
                "uniqueness": 100.0 * float(len(unique_valid)) / max(1, attempts),
                "quality": 100.0 * float(n_quality) / max(1, attempts),
                "sa_available": bool(sa is not None),
            }
        )
        return metrics

    def _save_step_checkpoint(*, epoch: int, global_step: int, step_in_epoch: int) -> None:
        if not _is_main_process():
            return
        quality_eval = (
            _evaluate_sampling_quality(epoch=epoch, global_step=global_step, step_in_epoch=step_in_epoch)
            if checkpoint_top_k_by_quality > 0
            else None
        )
        if quality_eval is not None:
            _append_quality_history(quality_history_path, quality_eval)
            print(
                f"[train-ao-gpt-parquet] quality_eval global_step={global_step} "
                f"quality={float(quality_eval['quality']):.2f} "
                f"validity={float(quality_eval['validity']):.2f} "
                f"uniqueness={float(quality_eval['uniqueness']):.2f} "
                f"n_unique={int(quality_eval['n_unique'])}/{int(quality_eval['num_attempts'])}",
                flush=True,
            )

        records = _load_quality_records(quality_records_path) if checkpoint_top_k_by_quality > 0 else []
        if checkpoint_top_k_by_quality > 0:
            score = _quality_score(quality_eval or {})
            if len(records) >= checkpoint_top_k_by_quality:
                worst = min(records, key=lambda r: (_quality_score(r.get("quality_eval", {})), int(r.get("global_step", 0))))
                worst_score = _quality_score(worst.get("quality_eval", {}))
                if score <= worst_score:
                    print(
                        f"[train-ao-gpt-parquet] step_checkpoint skipped_not_topk "
                        f"global_step={global_step} quality={score:.2f} topk_min={worst_score:.2f}",
                        flush=True,
                    )
                    return

        step_path = ckpt_dir / f"step_{global_step:08d}.pt"
        if checkpoint_top_k_by_quality <= 0 and keep_last_step_checkpoints > 0:
            existing = sorted(ckpt_dir.glob("step_*.pt"))
            for old_path in existing[: max(0, len(existing) - keep_last_step_checkpoints + 1)]:
                try:
                    old_path.unlink()
                except FileNotFoundError:
                    pass
        _atomic_torch_save(
            _checkpoint_payload(
                epoch=epoch,
                global_step=global_step,
                step_in_epoch=step_in_epoch,
                quality_eval=quality_eval,
            ),
            step_path,
        )
        (ckpt_dir / "latest_step.txt").write_text(step_path.name + "\n", encoding="utf-8")
        if checkpoint_top_k_by_quality > 0:
            records.append(
                {
                    "path": str(step_path),
                    "name": step_path.name,
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "step_in_epoch": int(step_in_epoch),
                    "quality_eval": quality_eval,
                }
            )
            records = sorted(
                records,
                key=lambda r: (_quality_score(r.get("quality_eval", {})), int(r.get("global_step", 0))),
                reverse=True,
            )
            for old_record in records[checkpoint_top_k_by_quality:]:
                old_path = Path(str(old_record.get("path") or ""))
                try:
                    old_path.unlink()
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    print(f"[train-ao-gpt-parquet] failed_to_remove_old_quality_ckpt path={old_path} err={exc}", flush=True)
            records = records[:checkpoint_top_k_by_quality]
            _write_quality_records(quality_records_path, records)
            best_path = Path(str(records[0].get("path"))) if records else step_path
            (ckpt_dir / "best_quality_checkpoint.txt").write_text(best_path.name + "\n", encoding="utf-8")
            _update_best_quality_link(out_dir, best_path)
        elif keep_last_step_checkpoints > 0:
            step_paths = sorted(ckpt_dir.glob("step_*.pt"))
            for old_path in step_paths[:-keep_last_step_checkpoints]:
                try:
                    old_path.unlink()
                except FileNotFoundError:
                    pass
        print(
            f"[train-ao-gpt-parquet] step_checkpoint global_step={global_step} "
            f"epoch={epoch} step_in_epoch={step_in_epoch} path={step_path}",
            flush=True,
        )

    if _is_main_process():
        shutil.copy2(stamp_vocab, out_dir / "stamp_vocab.txt")
        (out_dir / "vocab.json").write_text(json.dumps(vocab.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        if isinstance(vocab, HybridVocab):
            (out_dir / "hybrid_vocab.json").write_text(json.dumps(vocab.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
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
                    "save_every_steps": int(save_every_steps),
                    "keep_last_step_checkpoints": int(keep_last_step_checkpoints),
                    "checkpoint_top_k_by_quality": int(checkpoint_top_k_by_quality),
                    "quality_eval_samples": int(quality_eval_samples),
                    "quality_eval_batch_size": int(quality_eval_batch_size),
                    "quality_eval_temperature": float(args.quality_eval_temperature),
                    "quality_eval_top_k": int(args.quality_eval_top_k),
                    "quality_eval_top_p": float(args.quality_eval_top_p),
                    "quality_eval_max_new": int(quality_eval_max_new),
                    "condition_mode": condition_mode,
                    "quality_condition_tokens": list(quality_condition_tokens),
                    "resume_from": resume_path or None,
                    "init_from": init_path or None,
                    "init_summary": init_summary,
                    "resume_epoch": int(resume_epoch),
                    "resume_step_in_epoch": int(resume_step_in_epoch),
                    "resume_global_step": int(resume_global_step),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    global_opt_step = int(resume_global_step)
    start_epoch = 1
    resume_skip_opt_steps = 0
    if resume_path:
        if resume_step_in_epoch >= train_opt_steps:
            start_epoch = int(resume_epoch) + 1
        else:
            start_epoch = int(resume_epoch)
            resume_skip_opt_steps = int(resume_step_in_epoch)
    if _is_main_process() and resume_path:
        print(
            f"[train-ao-gpt-parquet] resume_plan start_epoch={start_epoch} "
            f"resume_skip_opt_steps={resume_skip_opt_steps} "
            f"train_opt_steps_per_epoch={train_opt_steps}",
            flush=True,
        )

    for epoch in range(start_epoch, cfg.epochs + 1):
        _debug_print(debug, f"[debug rank={_dist_rank()}] epoch={epoch} train_start")
        epoch_t0 = time.time()
        train_ds.set_epoch(epoch)
        model.train()
        train_loss_sum = 0.0
        train_token_count = 0
        optimizer.zero_grad(set_to_none=True)
        # Exactly `train_steps` iterations on EVERY rank, fed by an infinite
        # cycling iterator. `train_steps` is identical across ranks (global
        # rows / world_size), so the per-rank collective count is identical
        # by construction — the only NCCL-desync-proof contract. The dataset
        # producing slightly fewer rows than the budget just means the
        # cycling iterator reshuffles and tops up the tail (<1 batch here).
        batch_iter = _cycle_forever(loader_train)
        skip_micro_steps = 0
        if epoch == start_epoch and resume_skip_opt_steps > 0:
            skip_micro_steps = min(train_steps, int(resume_skip_opt_steps) * int(grad_accum_steps))
            if _is_main_process():
                print(
                    f"[train-ao-gpt-parquet] resume_skip epoch={epoch} "
                    f"micro_steps={skip_micro_steps}",
                    flush=True,
                )
            for _skip in range(skip_micro_steps):
                try:
                    next(batch_iter)
                except StopIteration:
                    break
        remaining_train_steps = max(0, train_steps - skip_micro_steps)
        bounded_iter = (next(batch_iter) for _ in range(remaining_train_steps))
        iterator = iterable_with_progress(
            bounded_iter,
            show_progress=(not args.no_tqdm) and _is_main_process(),
            total=remaining_train_steps,
            desc=f"Train ao-pretrain[stamp] ep{epoch}",
        )
        micro_step = int(skip_micro_steps)
        for x, y in iterator:
            micro_step += 1
            x = x.to(dev, non_blocking=non_blocking)
            y = y.to(dev, non_blocking=non_blocking)
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True) if use_amp else nullcontext()
            with amp_ctx:
                loss = loss_fn(model, x, y)
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
                global_opt_step += 1
                if save_every_steps > 0 and global_opt_step % save_every_steps == 0:
                    _save_step_checkpoint(
                        epoch=epoch,
                        global_step=global_opt_step,
                        step_in_epoch=math.ceil(micro_step / grad_accum_steps),
                    )
                    _distributed_barrier(dev)
            batch_token_count = int((y != -100).sum().item())
            train_loss_sum += float(loss.item()) * batch_token_count
            train_token_count += batch_token_count
        _debug_print(debug, f"[debug rank={_dist_rank()}] epoch={epoch} train_done tokens={train_token_count}")

        train_loss_sum, train_token_count = _distributed_sum_and_count(train_loss_sum, train_token_count, device=dev)
        train_loss_mean = (
            float(train_loss_sum / max(1, train_token_count))
            if train_token_count > 0
            else float("nan")
        )

        model.eval()
        valid_loss_sum = 0.0
        valid_token_count = 0
        _debug_print(debug, f"[debug rank={_dist_rank()}] epoch={epoch} valid_start")
        with torch.no_grad():
            for x, y in loader_valid:
                x = x.to(dev, non_blocking=non_blocking)
                y = y.to(dev, non_blocking=non_blocking)
                amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True) if use_amp else nullcontext()
                with amp_ctx:
                    loss = loss_fn(model, x, y)
                batch_token_count = int((y != -100).sum().item())
                valid_loss_sum += float(loss.item()) * batch_token_count
                valid_token_count += batch_token_count
        _debug_print(debug, f"[debug rank={_dist_rank()}] epoch={epoch} valid_done tokens={valid_token_count}")

        tr = train_loss_mean
        valid_loss_sum, valid_token_count = _distributed_sum_and_count(valid_loss_sum, valid_token_count, device=dev)
        va = (
            float(valid_loss_sum / max(1, valid_token_count))
            if valid_token_count > 0
            else float("nan")
        )
        if _is_main_process():
            epoch_wall_seconds = float(time.time() - epoch_t0)
            train_rows_per_sec = float(epoch_rows) / max(1e-9, epoch_wall_seconds)
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
            checkpoint_payload = _checkpoint_payload(
                epoch=epoch,
                global_step=global_opt_step,
                step_in_epoch=train_opt_steps,
            )
            if bool(args.save_latest_model):
                _atomic_torch_save(checkpoint_payload, out_dir / "model.pt")
            if args.save_every_epoch:
                _atomic_torch_save(checkpoint_payload, ckpt_dir / f"epoch_{epoch:04d}.pt")
            _debug_print(debug, f"[debug rank={_dist_rank()}] epoch={epoch} checkpoint_written")
        _distributed_barrier(dev)
    if _is_main_process():
        (out_dir / "vocab.json").write_text(json.dumps(vocab.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        if isinstance(vocab, HybridVocab):
            (out_dir / "hybrid_vocab.json").write_text(json.dumps(vocab.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        param_iter = _unwrap_model(model).parameters()
        meta = {
            "representation": "stamp",
            "train_size": scan_summary.train_rows,
            "valid_size": scan_summary.valid_rows_total,
            "n_train_encoded": scan_summary.train_rows,
            "n_valid_encoded": scan_summary.valid_rows_sampled,
            "valid_source": "parquet",
            "valid_excluded_rows": len(scan_summary.valid_keys),
            "vocab_size": len(vocab.itos),
            "vocab_type": "hybrid" if isinstance(vocab, HybridVocab) else "token",
            "motif_fallback": bool(args.motif_fallback),
            "motif_frequency_file": str(motif_frequency_file) if motif_frequency_file else "",
            "freq_cutoff": int(args.freq_cutoff),
            "max_atomic_motifs": int(args.max_atomic_motifs),
            "fallback_p_low": float(args.fallback_p_low),
            "fallback_p_high": float(args.fallback_p_high),
            "training_fallback": bool(args.training_fallback),
            "atomic_motifs": int(len(vocab.atomic_motifs)) if isinstance(vocab, HybridVocab) else None,
            "model_type": str(cfg.model_type),
            "cfg": vars(cfg),
            "device": str(dev),
            "world_size": int(_dist_world_size()),
            "data_parallel_world_size": int(dp_world_size),
            "n_model_params": int(sum(p.numel() for p in param_iter)),
            "micro_batch_size_per_rank": int(micro_batch_report),
            "global_batch_size": int(effective_global_batch),
            "grad_accum_steps": int(grad_accum_steps),
            "train_optimizer_steps_per_epoch_per_rank": int(train_opt_steps),
            "save_every_steps": int(save_every_steps),
            "keep_last_step_checkpoints": int(keep_last_step_checkpoints),
            "checkpoint_top_k_by_quality": int(checkpoint_top_k_by_quality),
            "quality_eval_samples": int(quality_eval_samples),
            "quality_eval_batch_size": int(quality_eval_batch_size),
            "quality_eval_temperature": float(args.quality_eval_temperature),
            "quality_eval_top_k": int(args.quality_eval_top_k),
            "quality_eval_top_p": float(args.quality_eval_top_p),
            "quality_eval_max_new": int(quality_eval_max_new),
            "condition_mode": condition_mode,
            "condition_tokens": condition_tokens,
            "quality_condition_tokens": list(quality_condition_tokens),
            "save_latest_model": bool(args.save_latest_model),
            "resume_from": resume_path or None,
            "init_from": init_path or None,
            "init_summary": init_summary,
            "resume_epoch": int(resume_epoch),
            "resume_step_in_epoch": int(resume_step_in_epoch),
            "resume_global_step": int(resume_global_step),
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
            "parquet_path": args.parquet,
            "parquet_paths": parquet_paths,
            "text_column": args.text_column,
            "smiles_column": args.smiles_column,
            "condition_column": args.condition_column,
            "split_column": args.split_column,
            "train_split": args.train_split,
            "valid_split": args.valid_split,
            "require_all_pass": bool(args.require_all_pass),
            "save_every_epoch": bool(args.save_every_epoch),
            "scan_cache_config": scan_cache_config,
        }
        (out_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output_dir": str(out_dir), "history": history, "device": str(dev)}, ensure_ascii=False, indent=2))
        _debug_print(debug, f"[debug rank={_dist_rank()}] main_finalize_done")
    _distributed_barrier(dev)
    _cleanup_distributed()
    _debug_print(debug, f"[debug rank={_dist_rank()}] cleanup_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
