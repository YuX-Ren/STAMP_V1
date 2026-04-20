#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.data import prepare_dataset_splits  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare train/valid/test SMILES splits without optional benchmark deps.")
    p.add_argument("--input", required=True)
    p.add_argument("--smiles-column", default="smiles")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-size", type=int, required=True)
    p.add_argument("--valid-size", type=int, required=True)
    p.add_argument("--test-size", type=int, required=True)
    p.add_argument("--low-resource-sizes", default="10000,50000,100000")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-mols", type=int, default=None)
    p.add_argument("--no-tqdm", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")
    low_resource_sizes = tuple(
        int(x.strip()) for x in args.low_resource_sizes.split(",") if x.strip()
    )
    summary = prepare_dataset_splits(
        input_path=args.input,
        output_dir=args.output_dir,
        smiles_column=args.smiles_column,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        low_resource_sizes=low_resource_sizes,
        seed=args.seed,
        max_mols=args.max_mols,
        show_progress=not args.no_tqdm,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
