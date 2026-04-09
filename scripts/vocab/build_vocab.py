#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import RDLogger

from stamp.vocab.stamp_vocab_types import BuildConfig
from stamp.vocab.vocab_builder import StampVocabBuilder, iter_smiles_from_file


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build STAMP motif vocabulary (core + schema) from a SMILES dataset."
    )
    p.add_argument("input", help="Input dataset path (.csv with smiles column, or .smiles/.smi text).")
    p.add_argument("--smiles-column", default="smiles", help="SMILES column for CSV input.")
    p.add_argument("--output-dir", default="results/vocab/stamp_vocab", help="Output directory.")
    p.add_argument("--max-mols", type=int, default=None, help="Optional max number of molecules.")
    p.add_argument("--min-core-freq", type=int, default=20)
    p.add_argument("--min-schema-freq", type=int, default=5)
    p.add_argument("--min-core-score", type=float, default=0.0)
    p.add_argument("--max-vocab-size", type=int, default=5000)
    p.add_argument("--num-workers", type=int, default=1, help="Number of worker processes.")
    p.add_argument("--chunksize", type=int, default=64, help="Parallel map chunksize.")
    p.add_argument(
        "--no-decomposition-records",
        action="store_true",
        help="Skip decomposition_records.jsonl export.",
    )
    p.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return p


def estimate_total_records(
    input_path: str,
    smiles_column: str = "smiles",
    max_mols: Optional[int] = None,
) -> Optional[int]:
    path = Path(input_path)
    total: Optional[int] = None

    try:
        if path.suffix.lower() in {".smi", ".smiles", ".txt"}:
            with path.open("r", encoding="utf-8") as f:
                total = sum(1 for line in f if line.strip())
        else:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                total = 0
                for row in reader:
                    smi = (row.get(smiles_column) or "").strip()
                    if smi:
                        total += 1
    except Exception:
        total = None

    if total is not None and max_mols is not None:
        total = min(total, max_mols)
    return total


def main() -> int:
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")
    total = estimate_total_records(
        args.input,
        smiles_column=args.smiles_column,
        max_mols=args.max_mols,
    )

    cfg = BuildConfig(
        min_core_freq=args.min_core_freq,
        min_schema_freq=args.min_schema_freq,
        min_core_score=args.min_core_score,
        max_vocab_size=args.max_vocab_size,
    )
    builder = StampVocabBuilder(cfg)

    smiles_for_fit = iter_smiles_from_file(
        args.input,
        smiles_column=args.smiles_column,
        max_mols=args.max_mols,
    )
    builder.fit_smiles(
        smiles_for_fit,
        show_progress=not args.no_tqdm,
        total=total,
        progress_desc="Build vocab",
        num_workers=max(1, args.num_workers),
        chunksize=max(1, args.chunksize),
    )

    out_dir = Path(args.output_dir)
    builder.save(out_dir)

    if not args.no_decomposition_records:
        smiles_for_records = iter_smiles_from_file(
            args.input,
            smiles_column=args.smiles_column,
            max_mols=args.max_mols,
        )
        builder.export_decomposition_records_smiles(
            smiles_for_records,
            out_dir / "decomposition_records.jsonl",
            show_progress=not args.no_tqdm,
            total=total,
            progress_desc="Export records",
            num_workers=max(1, args.num_workers),
            chunksize=max(1, args.chunksize),
        )

    print(f"Saved core vocab: {out_dir / 'core_vocab.json'}")
    print(f"Saved schema vocab: {out_dir / 'schema_vocab.json'}")
    print(f"Saved core-schema map: {out_dir / 'core_schema_map.json'}")
    if not args.no_decomposition_records:
        print(f"Saved decomposition records: {out_dir / 'decomposition_records.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
