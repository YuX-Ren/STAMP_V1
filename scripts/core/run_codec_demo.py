#!/usr/bin/env python3
"""Run canonical SMILES roundtrips on a small file without model artifacts."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.core.stamp_codec import STAMPEncoder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "examples/codec/molecules.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "results/demo/roundtrips.tsv")
    args = parser.parse_args()
    molecules = [s.strip() for s in args.input.read_text().splitlines() if s.strip()]
    if not molecules:
        parser.error("input must contain at least one SMILES")
    if args.input.resolve() == args.output.resolve():
        parser.error("output must differ from input")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    codec = STAMPEncoder()
    passed = 0
    # Exclusive creation protects existing user results from accidental replacement.
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["smiles", "stamp", "recovered_smiles", "status"])
        for smiles in molecules:
            try:
                tokens, recovered, ok = codec.roundtrip_smiles(smiles)
                passed += int(ok)
                writer.writerow([smiles, tokens, recovered, "OK" if ok else "FAIL"])
            except Exception as exc:
                writer.writerow([smiles, "", "", f"ERROR: {exc}"])
    print(f"SUMMARY\tpassed={passed}\ttotal={len(molecules)}\tfailed={len(molecules) - passed}")
    return 0 if passed == len(molecules) else 2


if __name__ == "__main__":
    raise SystemExit(main())
