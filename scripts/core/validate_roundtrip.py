#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.core.stamp_codec import STAMPEncoder


def iter_smiles(args_smiles: List[str]) -> Iterable[str]:
    if args_smiles:
        for s in args_smiles:
            s = s.strip()
            if s:
                yield s
        return
    for line in sys.stdin:
        s = line.strip()
        if s:
            yield s


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate SMILES -> STAMP -> SMILES roundtrip consistency."
    )
    p.add_argument(
        "smiles",
        nargs="*",
        help="Input SMILES list. If omitted, read one SMILES per line from stdin.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Only print failures.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    codec = STAMPEncoder()
    total = 0
    passed = 0

    for s in iter_smiles(args.smiles):
        total += 1
        try:
            stamp, recovered, ok = codec.roundtrip_smiles(s)
            if ok:
                passed += 1
            if (not args.quiet) or (not ok):
                status = "OK" if ok else "FAIL"
                print(f"{status}\t{s}\t{stamp}\t{recovered}")
        except Exception as exc:
            print(f"ERROR\t{s}\t{exc}")

    if total == 0:
        print("No input SMILES provided.", file=sys.stderr)
        return 1

    print(f"SUMMARY\tpassed={passed}\ttotal={total}\tfailed={total - passed}")
    return 0 if passed == total else 2


if __name__ == "__main__":
    raise SystemExit(main())
