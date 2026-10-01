#!/usr/bin/env python
"""Build MoleculeACE activity-cliff pair CSVs.

The output is directly compatible with
``scripts/benchmark/analyze_edit_mechanism.py`` because it includes:

    pair_id, edit_class, smiles_a, smiles_b

Additional columns preserve the MoleculeACE task, split, activity values,
Tanimoto similarity, and cliff_mol flags.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
RDLogger.DisableLog("rdApp.*")

from stamp.benchmark.common import canonicalize_smiles  # noqa: E402


def _morgan_fp(mol: Chem.Mol, radius: int, n_bits: int):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def _parse_splits(spec: str) -> Optional[set[str]]:
    clean = str(spec or "all").strip()
    if not clean or clean.lower() == "all":
        return None
    return {part.strip() for part in clean.split(",") if part.strip()}


def _split_value(values: Sequence[Any]) -> str:
    unique = sorted({str(v) for v in values if str(v)})
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    return "mixed"


def _clean_moleculeace_frame(
    csv_path: Path,
    *,
    smiles_col: str,
    target_col: str,
    cliff_col: str,
    split_col: str,
    splits: Optional[set[str]],
    keep_duplicates: bool,
    radius: int,
    n_bits: int,
) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    missing = [col for col in (smiles_col, target_col) if col not in df.columns]
    if missing:
        raise KeyError(f"{csv_path}: missing required columns {missing}")
    if splits is not None and split_col in df.columns:
        df = df[df[split_col].astype(str).isin(splits)].copy()

    records: List[Dict[str, Any]] = []
    for row_idx, row in df.iterrows():
        smiles = canonicalize_smiles(str(row[smiles_col]), strip_salts=True)
        if smiles is None:
            continue
        try:
            y = float(row[target_col])
        except Exception:
            continue
        if not math.isfinite(y):
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0 or len(Chem.GetMolFrags(mol)) != 1:
            continue
        cliff = int(row[cliff_col]) if cliff_col in df.columns and not pd.isna(row[cliff_col]) else 0
        split = str(row[split_col]) if split_col in df.columns and not pd.isna(row[split_col]) else ""
        records.append(
            {
                "source_row": int(row_idx),
                "smiles": smiles,
                "y": y,
                "cliff_mol": cliff,
                "split": split,
                "mol": mol,
                "fp": _morgan_fp(mol, radius, n_bits),
            }
        )

    if keep_duplicates:
        return records

    by_smiles: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        by_smiles.setdefault(str(rec["smiles"]), []).append(rec)

    deduped: List[Dict[str, Any]] = []
    for smiles, items in by_smiles.items():
        ys = [float(item["y"]) for item in items]
        mol = items[0]["mol"]
        deduped.append(
            {
                "source_row": ",".join(str(item["source_row"]) for item in items),
                "smiles": smiles,
                "y": float(sum(ys) / len(ys)),
                "cliff_mol": int(max(int(item["cliff_mol"]) for item in items)),
                "split": _split_value([item["split"] for item in items]),
                "mol": mol,
                "fp": items[0]["fp"],
            }
        )
    return deduped


def _passes_cliff_filter(a: Dict[str, Any], b: Dict[str, Any], mode: str) -> bool:
    if mode == "none":
        return True
    a_cliff = int(a.get("cliff_mol", 0)) == 1
    b_cliff = int(b.get("cliff_mol", 0)) == 1
    if mode == "either":
        return a_cliff or b_cliff
    if mode == "both":
        return a_cliff and b_cliff
    raise ValueError(f"Unknown cliff filter: {mode}")


def _pair_rows_for_task(
    csv_path: Path,
    records: Sequence[Dict[str, Any]],
    *,
    min_tanimoto: float,
    min_activity_diff: float,
    cliff_filter: str,
    edit_class: str,
    max_pairs_per_target: int,
    max_pairs_per_molecule: int,
) -> List[Dict[str, Any]]:
    task = csv_path.stem
    pair_rows: List[Dict[str, Any]] = []
    per_mol_counts: Dict[int, int] = {}

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a = records[i]
            b = records[j]
            if not _passes_cliff_filter(a, b, cliff_filter):
                continue
            activity_diff = abs(float(a["y"]) - float(b["y"]))
            if activity_diff < float(min_activity_diff):
                continue
            sim = float(DataStructs.TanimotoSimilarity(a["fp"], b["fp"]))
            if sim < float(min_tanimoto):
                continue

            low, high = (a, b) if float(a["y"]) <= float(b["y"]) else (b, a)
            pair_rows.append(
                {
                    "pair_id": f"{task}:{low['source_row']}:{high['source_row']}",
                    "edit_class": edit_class,
                    "target": task,
                    "source_file": csv_path.name,
                    "smiles_a": low["smiles"],
                    "smiles_b": high["smiles"],
                    "y_a": float(low["y"]),
                    "y_b": float(high["y"]),
                    "activity_diff": float(activity_diff),
                    "tanimoto_morgan_r2": float(sim),
                    "split_a": low["split"],
                    "split_b": high["split"],
                    "cliff_mol_a": int(low["cliff_mol"]),
                    "cliff_mol_b": int(high["cliff_mol"]),
                    "source_row_a": low["source_row"],
                    "source_row_b": high["source_row"],
                }
            )

    pair_rows.sort(
        key=lambda row: (
            -float(row["activity_diff"]),
            -float(row["tanimoto_morgan_r2"]),
            str(row["target"]),
            str(row["pair_id"]),
        )
    )

    if max_pairs_per_molecule > 0:
        filtered: List[Dict[str, Any]] = []
        for row in pair_rows:
            a_key = hash((row["target"], row["source_row_a"]))
            b_key = hash((row["target"], row["source_row_b"]))
            if per_mol_counts.get(a_key, 0) >= max_pairs_per_molecule:
                continue
            if per_mol_counts.get(b_key, 0) >= max_pairs_per_molecule:
                continue
            filtered.append(row)
            per_mol_counts[a_key] = per_mol_counts.get(a_key, 0) + 1
            per_mol_counts[b_key] = per_mol_counts.get(b_key, 0) + 1
        pair_rows = filtered

    if max_pairs_per_target > 0:
        pair_rows = pair_rows[:max_pairs_per_target]
    return pair_rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id",
        "edit_class",
        "target",
        "source_file",
        "smiles_a",
        "smiles_b",
        "y_a",
        "y_b",
        "activity_diff",
        "tanimoto_morgan_r2",
        "split_a",
        "split_b",
        "cliff_mol_a",
        "cliff_mol_b",
        "source_row_a",
        "source_row_b",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _iter_csvs(path: Path, pattern: str) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    yield from sorted(path.glob(pattern))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MoleculeACE activity-cliff molecule-pair CSV for STAMP edit analysis."
    )
    parser.add_argument("--moleculeace-dir", required=True, help="Directory containing MoleculeACE benchmark CSVs, or one CSV.")
    parser.add_argument("--output", required=True, help="Output pair CSV.")
    parser.add_argument("--pattern", default="CHEMBL*_*.csv")
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--target-col", default="y [pEC50/pKi]")
    parser.add_argument("--cliff-col", default="cliff_mol")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--splits", default="all", help="Comma-separated split values, or 'all'.")
    parser.add_argument("--min-tanimoto", type=float, default=0.9)
    parser.add_argument("--min-activity-diff", type=float, default=1.0)
    parser.add_argument("--cliff-filter", choices=("both", "either", "none"), default="both")
    parser.add_argument("--edit-class", default="moleculeace_activity_cliff")
    parser.add_argument("--max-pairs-per-target", type=int, default=0, help="0 = no cap.")
    parser.add_argument("--max-pairs-per-molecule", type=int, default=0, help="0 = no cap.")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--morgan-bits", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ace_path = Path(args.moleculeace_dir)
    splits = _parse_splits(args.splits)

    all_pairs: List[Dict[str, Any]] = []
    summaries: List[str] = []
    for csv_path in _iter_csvs(ace_path, args.pattern):
        records = _clean_moleculeace_frame(
            csv_path,
            smiles_col=args.smiles_col,
            target_col=args.target_col,
            cliff_col=args.cliff_col,
            split_col=args.split_col,
            splits=splits,
            keep_duplicates=bool(args.keep_duplicates),
            radius=int(args.morgan_radius),
            n_bits=int(args.morgan_bits),
        )
        pairs = _pair_rows_for_task(
            csv_path,
            records,
            min_tanimoto=float(args.min_tanimoto),
            min_activity_diff=float(args.min_activity_diff),
            cliff_filter=str(args.cliff_filter),
            edit_class=str(args.edit_class),
            max_pairs_per_target=int(args.max_pairs_per_target),
            max_pairs_per_molecule=int(args.max_pairs_per_molecule),
        )
        all_pairs.extend(pairs)
        summaries.append(f"{csv_path.name}: molecules={len(records)} pairs={len(pairs)}")

    all_pairs.sort(key=lambda row: (str(row["target"]), str(row["pair_id"])))
    _write_csv(Path(args.output), all_pairs)

    print(f"[moleculeace-cliffs] wrote {len(all_pairs)} pairs to {args.output}")
    for line in summaries:
        print(f"[moleculeace-cliffs] {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
