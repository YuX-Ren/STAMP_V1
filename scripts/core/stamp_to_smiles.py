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
        description="Convert a STAMP sequence into a SMILES string."
    )
    p.add_argument(
        "stamp",
        nargs="?",
        help="Input STAMP sequence. If omitted, read one STAMP sequence per line from stdin.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    codec = STAMPEncoder()

    if args.stamp:
        print(codec.decode_stamp(args.stamp, canonical=True))
        return 0

    has_any = False
    for line in sys.stdin:
        s = line.strip()
        if not s:
            continue
        has_any = True
        try:
            print(codec.decode_stamp(s, canonical=True))
        except Exception as exc:
            print(f"ERROR\t{s}\t{exc}", file=sys.stderr)
    if not has_any:
        print("No input STAMP provided.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
