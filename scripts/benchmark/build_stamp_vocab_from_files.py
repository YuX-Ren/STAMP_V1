#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from itertools import chain, islice
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import RDLogger

from stamp.vocab.stamp_vocab_types import BuildConfig
from stamp.vocab.vocab_builder import StampVocabBuilder, iter_smiles_from_file


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build STAMP motif vocab from multiple inputs.")
    p.add_argument("--input", action="append", required=True)
    p.add_argument("--smiles-column", default="smiles")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-mols", type=int, default=None)
    p.add_argument("--min-core-freq", type=int, default=1)
    p.add_argument("--min-schema-freq", type=int, default=1)
    p.add_argument("--min-core-score", type=float, default=0.0)
    p.add_argument("--max-vocab-size", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--chunksize", type=int, default=64)
    p.add_argument("--no-decomposition-records", action="store_true")
    p.add_argument("--no-tqdm", action="store_true")
    return p


def _estimate_total_records(
    input_paths: Iterable[str],
    smiles_column: str,
    max_mols: Optional[int],
) -> Optional[int]:
    total = 0
    try:
        for input_path in input_paths:
            path = Path(input_path)
            if path.suffix.lower() in {".smi", ".smiles", ".txt"}:
                with path.open("r", encoding="utf-8") as f:
                    total += sum(1 for line in f if line.strip())
            else:
                with path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        smi = (row.get(smiles_column) or "").strip()
                        if smi:
                            total += 1
        if max_mols is not None:
            total = min(total, max_mols)
        return total
    except Exception:
        return None


def _iter_multi_inputs(input_paths: list[str], smiles_column: str, max_mols: Optional[int]):
    stream = chain.from_iterable(
        iter_smiles_from_file(path, smiles_column=smiles_column, max_mols=None)
        for path in input_paths
    )
    if max_mols is not None:
        stream = islice(stream, int(max_mols))
    return stream


def main() -> int:
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")

    cfg = BuildConfig(
        min_core_freq=args.min_core_freq,
        min_schema_freq=args.min_schema_freq,
        min_core_score=args.min_core_score,
        max_vocab_size=(None if int(args.max_vocab_size) <= 0 else int(args.max_vocab_size)),
    )
    builder = StampVocabBuilder(cfg)
    total = _estimate_total_records(args.input, args.smiles_column, args.max_mols)

    builder.fit_smiles(
        _iter_multi_inputs(args.input, args.smiles_column, args.max_mols),
        show_progress=not args.no_tqdm,
        total=total,
        progress_desc="Build vocab",
        num_workers=max(1, args.num_workers),
        chunksize=max(1, args.chunksize),
    )
    out_dir = Path(args.output_dir)
    builder.save(out_dir)

    if not args.no_decomposition_records:
        builder.export_decomposition_records_smiles(
            _iter_multi_inputs(args.input, args.smiles_column, args.max_mols),
            out_dir / "decomposition_records.jsonl",
            show_progress=not args.no_tqdm,
            total=total,
            progress_desc="Export records",
            num_workers=max(1, args.num_workers),
            chunksize=max(1, args.chunksize),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
