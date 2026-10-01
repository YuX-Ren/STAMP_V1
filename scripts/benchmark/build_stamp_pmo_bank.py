#!/usr/bin/env python3
"""Build an offline STAMP motif/subtree bank for PMO tasks.

This is the STAMP-native analogue of GenMol's PMO ``get_vocab.py``:

    ZINC250k molecules
      -> oracle labels for one PMO task
      -> STAMP encode
      -> extract motif groups and short contiguous subtrees
      -> score each unit by mean oracle score of molecules containing it

The generated bank is intentionally oracle-pretested and should be treated
like GenMol's fragment vocabulary: it is built before the 10k PMO loop from
ZINC250k-only information.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.mcts_guacamol import parse_next_group  # noqa: E402
from stamp.benchmark.mcts_unified.pmo_oracle import PMO_TASKS  # noqa: E402
from stamp.core.stamp_codec import STAMPEncoder  # noqa: E402


_WORKER_ENCODER: Optional[STAMPEncoder] = None


def _init_worker() -> None:
    global _WORKER_ENCODER
    _WORKER_ENCODER = STAMPEncoder()


@dataclass
class UnitAgg:
    unit_type: str
    tokens: Tuple[str, ...]
    support: int = 0
    total_score: float = 0.0
    best_score: float = 0.0
    first_smiles: str = ""
    examples: List[Tuple[float, str]] = field(default_factory=list)

    def add(self, score: float, smiles: str) -> None:
        self.support += 1
        self.total_score += float(score)
        self.best_score = max(self.best_score, float(score))
        if not self.first_smiles:
            self.first_smiles = smiles
        self.examples.append((float(score), smiles))
        if len(self.examples) > 8:
            self.examples.sort(key=lambda x: -x[0])
            del self.examples[8:]

    @property
    def mean_score(self) -> float:
        return self.total_score / max(1, self.support)


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def clean_tokens(tokens: Sequence[str]) -> List[str]:
    out: List[str] = []
    for tok in tokens:
        if not tok or tok.startswith("<"):
            continue
        out.append(tok)
        if tok == "[END]":
            break
    return out


def split_motif_groups(tokens: Sequence[str]) -> List[Tuple[int, int, Tuple[str, ...]]]:
    cleaned = clean_tokens(tokens)
    groups: List[Tuple[int, int, Tuple[str, ...]]] = []
    i = 0
    while i < len(cleaned):
        if cleaned[i] == "[END]":
            break
        end = parse_next_group(cleaned, i)
        if end is None or end <= i:
            i += 1
            continue
        groups.append((i, end, tuple(cleaned[i:end])))
        i = end
    return groups


def unit_signature(tokens: Sequence[str]) -> str:
    joints = sum(1 for t in tokens if t.startswith("[J_"))
    branches = sum(1 for t in tokens if t.startswith("[B_"))
    rings = sum(1 for t in tokens if t == "[Ring]")
    motifs = sum(1 for t in tokens if t and not t.startswith("["))
    return f"motifs={motifs};joints={joints};branches={branches};rings={rings}"


def motif_head(tokens: Sequence[str]) -> str:
    return next((t for t in tokens if t and not t.startswith("[")), "")


def iter_input_rows(path: Path, smiles_column: str, max_mols: Optional[int]) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        first = f.readline()
        if not first:
            return
        f.seek(0)
        header = [x.strip() for x in first.rstrip("\n").split(",")]
        has_header = smiles_column in header
        if has_header:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_mols is not None and i >= max_mols:
                    break
                yield row
        else:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if max_mols is not None and i >= max_mols:
                    break
                if not row:
                    continue
                yield {smiles_column: row[0]}


def add_unit(
    units: Dict[Tuple[str, Tuple[str, ...]], UnitAgg],
    unit_type: str,
    tokens: Sequence[str],
    score: float,
    smiles: str,
) -> None:
    key = (unit_type, tuple(tokens))
    agg = units.get(key)
    if agg is None:
        agg = UnitAgg(unit_type=unit_type, tokens=tuple(tokens))
        units[key] = agg
    agg.add(score, smiles)


def extract_units(
    *,
    units: Dict[Tuple[str, Tuple[str, ...]], UnitAgg],
    tokens: Sequence[str],
    score: float,
    smiles: str,
    max_subtree_groups: int,
    unit_types: Set[str],
) -> int:
    groups = split_motif_groups(tokens)
    if not groups:
        return 0

    seen: set[Tuple[str, Tuple[str, ...]]] = set()
    if "motif" in unit_types:
        for _start, _end, group in groups:
            key = ("motif", group)
            if key not in seen:
                add_unit(units, "motif", group, score, smiles)
                seen.add(key)

    if "subtree" in unit_types:
        cleaned = clean_tokens(tokens)
        n = len(groups)
        for i in range(n):
            for width in range(2, max_subtree_groups + 1):
                j = i + width
                if j > n:
                    break
                start = groups[i][0]
                end = groups[j - 1][1]
                subtree = tuple(cleaned[start:end])
                key = ("subtree", subtree)
                if key not in seen:
                    add_unit(units, "subtree", subtree, score, smiles)
                    seen.add(key)

    # Lightweight context unit: previous/current/next motif heads.  This is a
    # reason feature for later planning, not directly executable.
    if "context" in unit_types:
        heads = [motif_head(g) for _s, _e, g in groups]
        for i, head in enumerate(heads):
            prev_h = heads[i - 1] if i > 0 else "<root>"
            next_h = heads[i + 1] if i + 1 < len(heads) else "<leaf>"
            ctx = (prev_h, "=>", head, "=>", next_h)
            key = ("context", ctx)
            if key not in seen:
                add_unit(units, "context", ctx, score, smiles)
                seen.add(key)
    return len(seen)


def merge_units(dst: Dict[Tuple[str, Tuple[str, ...]], UnitAgg], src: Dict[Tuple[str, Tuple[str, ...]], UnitAgg]) -> None:
    for key, agg in src.items():
        cur = dst.get(key)
        if cur is None:
            dst[key] = agg
            continue
        cur.support += agg.support
        cur.total_score += agg.total_score
        cur.best_score = max(cur.best_score, agg.best_score)
        if not cur.first_smiles:
            cur.first_smiles = agg.first_smiles
        cur.examples.extend(agg.examples)
        cur.examples.sort(key=lambda x: -x[0])
        del cur.examples[8:]


def process_chunk(
    rows: Sequence[Tuple[str, float]],
    max_subtree_groups: int,
    unit_types_tuple: Tuple[str, ...],
) -> Tuple[int, int, int, int, Dict[Tuple[str, Tuple[str, ...]], UnitAgg]]:
    global _WORKER_ENCODER
    if _WORKER_ENCODER is None:
        _WORKER_ENCODER = STAMPEncoder()
    local: Dict[Tuple[str, Tuple[str, ...]], UnitAgg] = {}
    n_seen = n_valid = n_encoded = n_units = 0
    unit_types = set(unit_types_tuple)
    for raw, score in rows:
        n_seen += 1
        smiles = canonicalize_smiles(raw)
        if smiles is None:
            continue
        n_valid += 1
        try:
            tokens = _WORKER_ENCODER.encode_smiles(smiles).split()
        except Exception:
            continue
        n_encoded += 1
        n_units += extract_units(
            units=local,
            tokens=tokens,
            score=score,
            smiles=smiles,
            max_subtree_groups=max_subtree_groups,
            unit_types=unit_types,
        )
    return n_seen, n_valid, n_encoded, n_units, local


def write_bank(
    *,
    units: Dict[Tuple[str, Tuple[str, ...]], UnitAgg],
    output_dir: Path,
    task: str,
    min_support: int,
    top_motifs: int,
    top_subtrees: int,
    top_contexts: int,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    limits = {"motif": top_motifs, "subtree": top_subtrees, "context": top_contexts}
    summary: Dict[str, object] = {"task": task, "files": {}, "counts": {}}
    for unit_type, limit in limits.items():
        rows = [
            agg for agg in units.values()
            if agg.unit_type == unit_type and agg.support >= min_support
        ]
        rows.sort(key=lambda a: (a.mean_score, a.best_score, a.support), reverse=True)
        rows = rows[: max(0, limit)]
        out = output_dir / f"{unit_type}s.csv"
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "unit_id",
                    "unit_type",
                    "tokens",
                    "head",
                    "support",
                    "mean_score",
                    "best_score",
                    "signature",
                    "first_smiles",
                    "top_examples",
                ],
            )
            writer.writeheader()
            for idx, agg in enumerate(rows):
                examples = sorted(agg.examples, key=lambda x: -x[0])[:5]
                writer.writerow(
                    {
                        "unit_id": f"{unit_type}_{idx:06d}",
                        "unit_type": unit_type,
                        "tokens": " ".join(agg.tokens),
                        "head": motif_head(agg.tokens),
                        "support": agg.support,
                        "mean_score": f"{agg.mean_score:.10f}",
                        "best_score": f"{agg.best_score:.10f}",
                        "signature": unit_signature(agg.tokens),
                        "first_smiles": agg.first_smiles,
                        "top_examples": json.dumps(examples, ensure_ascii=False),
                    }
                )
        summary["files"][unit_type] = str(out)
        summary["counts"][unit_type] = len(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build ZINC250k-only STAMP PMO motif/subtree bank.")
    p.add_argument("--input", required=True, help="ZINC250k CSV/SMI file.")
    p.add_argument("--smiles-column", default="smiles")
    p.add_argument("--task", required=True, choices=PMO_TASKS)
    p.add_argument("--score-column", default=None,
                   help="Use an existing oracle score column instead of calling TDC.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-mols", type=int, default=None)
    p.add_argument("--max-subtree-groups", type=int, default=3)
    p.add_argument("--unit-types", default="motif,subtree,context",
                   help="Comma-separated unit types: motif,subtree,context.")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=2000)
    p.add_argument("--min-support", type=int, default=2)
    p.add_argument("--top-motifs", type=int, default=1000)
    p.add_argument("--top-subtrees", type=int, default=5000)
    p.add_argument("--top-contexts", type=int, default=1000)
    p.add_argument("--progress-every", type=int, default=5000)
    return p


def main() -> int:
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    oracle = None
    if args.score_column is None:
        from tdc import Oracle as TDCOracle
        oracle = TDCOracle(name=args.task)

    encoder = STAMPEncoder()
    units: Dict[Tuple[str, Tuple[str, ...]], UnitAgg] = {}
    n_seen = n_valid = n_encoded = n_units = 0
    t0 = time.time()
    unit_types = {
        x.strip() for x in args.unit_types.split(",") if x.strip()
    }
    bad_unit_types = unit_types - {"motif", "subtree", "context"}
    if bad_unit_types:
        raise ValueError(f"Unknown --unit-types: {sorted(bad_unit_types)}")

    def row_iter() -> Iterable[Tuple[str, float]]:
        for row in iter_input_rows(input_path, args.smiles_column, args.max_mols):
            raw = (row.get(args.smiles_column) or "").strip()
            if not raw:
                continue
            if args.score_column is not None:
                val = row.get(args.score_column)
                if val is None or val == "":
                    continue
                try:
                    score = float(val)
                except ValueError:
                    continue
            else:
                smiles = canonicalize_smiles(raw)
                if smiles is None:
                    continue
                try:
                    score = float(oracle(smiles))  # type: ignore[misc]
                except Exception:
                    continue
            if not np.isfinite(score):
                continue
            yield raw, max(0.0, min(1.0, score))

    if args.num_workers > 1:
        pending = []
        chunk: List[Tuple[str, float]] = []
        next_report = args.progress_every
        with ProcessPoolExecutor(max_workers=args.num_workers, initializer=_init_worker) as ex:
            for item in row_iter():
                chunk.append(item)
                if len(chunk) >= args.chunk_size:
                    pending.append(
                        ex.submit(
                            process_chunk,
                            tuple(chunk),
                            args.max_subtree_groups,
                            tuple(sorted(unit_types)),
                        )
                    )
                    chunk = []
            if chunk:
                pending.append(
                    ex.submit(
                        process_chunk,
                        tuple(chunk),
                        args.max_subtree_groups,
                        tuple(sorted(unit_types)),
                    )
                )
            for fut in as_completed(pending):
                c_seen, c_valid, c_encoded, c_units, c_aggs = fut.result()
                n_seen += c_seen
                n_valid += c_valid
                n_encoded += c_encoded
                n_units += c_units
                merge_units(units, c_aggs)
                if args.progress_every > 0 and n_seen >= next_report:
                    dt = time.time() - t0
                    print(
                        f"[bank] seen={n_seen} valid={n_valid} encoded={n_encoded} "
                        f"unique_units={len(units)} elapsed={dt:.1f}s",
                        flush=True,
                    )
                    while n_seen >= next_report:
                        next_report += args.progress_every
    else:
        for raw, score in row_iter():
            n_seen += 1
            smiles = canonicalize_smiles(raw)
            if smiles is None:
                continue
            n_valid += 1
            try:
                tokens = encoder.encode_smiles(smiles).split()
            except Exception:
                continue
            n_encoded += 1
            n_units += extract_units(
                units=units,
                tokens=tokens,
                score=score,
                smiles=smiles,
                max_subtree_groups=args.max_subtree_groups,
                unit_types=unit_types,
            )
            if args.progress_every > 0 and n_seen % args.progress_every == 0:
                dt = time.time() - t0
                print(
                    f"[bank] seen={n_seen} valid={n_valid} encoded={n_encoded} "
                    f"unique_units={len(units)} elapsed={dt:.1f}s",
                    flush=True,
                )

    summary = write_bank(
        units=units,
        output_dir=output_dir,
        task=args.task,
        min_support=args.min_support,
        top_motifs=args.top_motifs,
        top_subtrees=args.top_subtrees,
        top_contexts=args.top_contexts,
    )
    summary.update(
        {
            "input": str(input_path),
            "n_seen": n_seen,
            "n_valid": n_valid,
            "n_encoded": n_encoded,
            "n_extracted_units": n_units,
            "n_unique_units_raw": len(units),
            "elapsed_seconds": time.time() - t0,
            "score_source": args.score_column or f"tdc:{args.task}",
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
