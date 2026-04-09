from __future__ import annotations

import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .common import canonicalize_smiles, chemically_equivalent_smiles, iterable_with_progress
from .representations import BaseRepresentation, UnsupportedRepresentation, build_representation


def _compute_representation_stats_worker(payload: tuple[str, Sequence[str], Optional[int]]) -> dict[str, Any]:
    name, smiles_list, max_items = payload
    try:
        rep = build_representation(name)
        return compute_representation_statistics(
            smiles_list,
            rep,
            max_items=max_items,
            show_progress=False,
        )
    except Exception as exc:
        return {"repr": name, "error": "exception", "message": str(exc)}


def _safe_float(v: Sequence[float], fn, default: float = 0.0) -> float:
    if not v:
        return default
    return float(fn(v))


def _percentile(v: Sequence[float], q: float) -> float:
    if not v:
        return 0.0
    return float(np.percentile(np.asarray(v, dtype=float), q))


def _stamp_branch_depth_and_attachment(tokens: Sequence[str]) -> Tuple[int, int, int]:
    frames: list[int] = []
    max_depth = 0
    attachment_count = 0
    branch_count = 0

    for tok in tokens:
        if tok.startswith("[J_"):
            attachment_count += 1
            continue
        if tok.startswith("[B_") and tok.endswith("]"):
            try:
                n = int(tok[3:-1])
            except Exception:
                n = 0
            branch_count += 1
            frames.append(max(0, n))
            max_depth = max(max_depth, len(frames))
            continue
        if tok in {"[END]", "[BACK]"} or tok.startswith("[S_"):
            continue

        if frames:
            for i in range(len(frames)):
                frames[i] -= 1
            while frames and frames[-1] <= 0:
                frames.pop()

    return max_depth, attachment_count, branch_count


def compute_representation_statistics(
    smiles_list: Sequence[str],
    rep: BaseRepresentation,
    max_items: Optional[int] = None,
    show_progress: bool = False,
) -> Dict[str, Any]:
    if isinstance(rep, UnsupportedRepresentation):
        return {
            "repr": rep.name,
            "error": "unsupported_placeholder",
            "message": "Representation placeholder not implemented.",
        }

    rows = smiles_list[: max_items if max_items is not None else len(smiles_list)]

    char_lens: list[int] = []
    token_lens: list[int] = []
    smile_char_lens: list[int] = []
    smile_token_lens: list[int] = []

    vocab = Counter()
    total_tokens = 0
    special_tokens = 0

    enc_ok = 0
    dec_ok = 0
    exact_ok = 0

    branch_depths: list[int] = []
    attachment_counts: list[int] = []

    iterator = iterable_with_progress(rows, show_progress=show_progress, total=len(rows), desc=f"Stats[{rep.name}]")
    for smiles in iterator:
        canon = canonicalize_smiles(smiles)
        if canon is None:
            continue

        smile_char_lens.append(len(canon))
        smile_token_lens.append(len(list(canon)))

        try:
            encoded = rep.encode(canon)
        except Exception:
            continue
        enc_ok += 1

        toks = rep.tokenize(encoded)
        char_lens.append(len(encoded))
        token_lens.append(len(toks))
        total_tokens += len(toks)
        for t in toks:
            vocab[t] += 1
            if rep.is_special_token(t):
                special_tokens += 1

        if rep.name == "stamp":
            depth, attach_cnt, _ = _stamp_branch_depth_and_attachment(toks)
            branch_depths.append(depth)
            attachment_counts.append(attach_cnt)

        if rep.supports_decode:
            try:
                decoded = rep.decode(encoded)
                dec_ok += 1
                if chemically_equivalent_smiles(decoded, canon):
                    exact_ok += 1
            except Exception:
                pass

    stats: Dict[str, Any] = {
        "repr": rep.name,
        "n_input": len(rows),
        "n_encode_ok": enc_ok,
        "encode_valid_rate": enc_ok / max(1, len(rows)),
        "decode_valid_rate": dec_ok / max(1, enc_ok),
        "decode_exact_rate": exact_ok / max(1, enc_ok),
        "avg_char_length": _safe_float(char_lens, np.mean),
        "median_char_length": _safe_float(char_lens, np.median),
        "avg_token_length": _safe_float(token_lens, np.mean),
        "median_token_length": _safe_float(token_lens, np.median),
        "p95_token_length": _percentile(token_lens, 95),
        "max_token_length": max(token_lens) if token_lens else 0,
        "vocab_size": len(vocab),
        "special_token_ratio": (special_tokens / total_tokens) if total_tokens else 0.0,
        "char_len_ratio_vs_smiles": (
            _safe_float(char_lens, np.mean) / max(1e-8, _safe_float(smile_char_lens, np.mean))
        )
        if smile_char_lens
        else 0.0,
        "token_len_ratio_vs_smiles": (
            _safe_float(token_lens, np.mean) / max(1e-8, _safe_float(smile_token_lens, np.mean))
        )
        if smile_token_lens
        else 0.0,
    }

    if branch_depths:
        stats["avg_branch_depth"] = _safe_float(branch_depths, np.mean)
        stats["max_branch_depth"] = max(branch_depths)
    if attachment_counts:
        stats["avg_attachment_token_count"] = _safe_float(attachment_counts, np.mean)

    histogram: Dict[str, int] = {}
    for x in token_lens:
        k = str(int(x))
        histogram[k] = histogram.get(k, 0) + 1
    stats["token_length_histogram"] = histogram

    return stats


def run_representation_benchmark(
    smiles_list: Sequence[str],
    representations: Sequence[str],
    output_dir: str | Path,
    max_items: Optional[int] = None,
    show_progress: bool = False,
    num_workers: int = 1,
) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    if num_workers <= 0:
        num_workers = 1

    if num_workers == 1:
        for name in representations:
            try:
                rep = build_representation(name)
                stats = compute_representation_statistics(
                    smiles_list,
                    rep,
                    max_items=max_items,
                    show_progress=show_progress,
                )
            except Exception as exc:
                stats = {"repr": name, "error": "exception", "message": str(exc)}
            results.append(stats)
    else:
        payloads = [(name, smiles_list, max_items) for name in representations]
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            mapped = pool.map(_compute_representation_stats_worker, payloads, chunksize=1)
            mapped = iterable_with_progress(
                mapped,
                show_progress=show_progress,
                total=len(payloads),
                desc="repr-stats workers",
            )
            results = list(mapped)

    summary = {
        "n_smiles": min(len(smiles_list), max_items) if max_items is not None else len(smiles_list),
        "representations": list(representations),
        "results": results,
    }
    (out / "representation_stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    field_order = [
        "repr",
        "n_input",
        "n_encode_ok",
        "encode_valid_rate",
        "decode_valid_rate",
        "decode_exact_rate",
        "avg_char_length",
        "median_char_length",
        "avg_token_length",
        "median_token_length",
        "p95_token_length",
        "max_token_length",
        "vocab_size",
        "special_token_ratio",
        "char_len_ratio_vs_smiles",
        "token_len_ratio_vs_smiles",
        "avg_branch_depth",
        "max_branch_depth",
        "avg_attachment_token_count",
        "error",
        "message",
    ]
    with (out / "representation_stats_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in field_order})

    return summary
