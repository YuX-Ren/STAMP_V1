#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.core.stamp_codec import STAMPEncoder


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert a SMILES string into a STAMP sequence."
    )
    p.add_argument(
        "smiles",
        nargs="?",
        help="Input SMILES. If omitted, read one SMILES per line from stdin.",
    )
    p.add_argument(
        "--max-branch-n",
        type=int,
        default=None,
        help="Optional upper bound for emitted [B_n] token values.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    encoder = STAMPEncoder(max_branch_n=args.max_branch_n)

    if args.smiles:
        print(encoder.encode_smiles(args.smiles))
        return 0

    has_any = False
    for line in sys.stdin:
        s = line.strip()
        if not s:
            continue
        has_any = True
        try:
            print(encoder.encode_smiles(s))
        except Exception as exc:
            print(f"ERROR\t{s}\t{exc}", file=sys.stderr)
    if not has_any:
        print("No input SMILES provided.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
