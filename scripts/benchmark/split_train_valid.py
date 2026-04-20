"""Split a SMILES corpus into train/valid (+ optional test).

Useful for turning a large mining corpus (e.g. ChEMBL + MoleculeNet train) into
the train/valid pair expected by ``experiments_cli.py train-lm``.

Examples:
    python scripts/benchmark/split_train_valid.py \
        --input data/combined_chembl_mnet/smiles.csv \
        --smiles-column smiles \
        --output-dir data/combined_chembl_mnet/split \
        --valid-fraction 0.02 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import List


def load_smiles(path: Path, smiles_column: str) -> List[str]:
    out: List[str] = []
    suffix = path.suffix.lower()
    if suffix in {".smi", ".smiles", ".txt"}:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().split()[0] if line.strip() else ""
                if s:
                    out.append(s)
    else:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = (row.get(smiles_column) or "").strip().strip('"').strip()
                if s:
                    out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--smiles-column", default="smiles")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--valid-fraction", type=float, default=0.02)
    ap.add_argument("--test-fraction", type=float, default=0.0, help="0 = no test split")
    ap.add_argument("--max-valid", type=int, default=50000, help="Cap valid at this many mols")
    ap.add_argument("--max-test", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--format",
        default="smiles",
        choices=["smiles", "csv"],
        help="Output file format. 'smiles' = one SMILES per line (expected by train-lm).",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[split] loading {args.input}", flush=True)
    smis = load_smiles(args.input, args.smiles_column)
    print(f"[split] {len(smis)} SMILES loaded", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(smis)

    n = len(smis)
    n_valid = min(int(n * args.valid_fraction), args.max_valid)
    n_test = min(int(n * args.test_fraction), args.max_test) if args.test_fraction > 0 else 0

    valid = smis[:n_valid]
    test = smis[n_valid : n_valid + n_test]
    train = smis[n_valid + n_test :]

    def _write(path: Path, items: List[str]) -> None:
        with path.open("w", encoding="utf-8") as f:
            if args.format == "csv":
                f.write("smiles\n")
            for s in items:
                f.write(s + "\n")

    ext = "smiles" if args.format == "smiles" else "csv"
    _write(args.output_dir / f"train.{ext}", train)
    _write(args.output_dir / f"valid.{ext}", valid)
    if test:
        _write(args.output_dir / f"test.{ext}", test)

    print(f"[split] train={len(train)} valid={len(valid)} test={len(test)}", flush=True)
    print(f"[split] wrote to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
