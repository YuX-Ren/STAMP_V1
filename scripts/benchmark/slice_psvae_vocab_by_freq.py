#!/usr/bin/env python3
"""Slice a PS-VAE vocab.txt by USAGE frequency from a token_counts.tsv.

Output is a vocab.txt with the same header but only top-K motifs (by
empirical usage on the corpus). STAMP grammar tokens ([J_*]/[B_*]/[S_*]/
[Ring]/[END]/[BACK]) are not motifs and are filtered out before ranking.
Motifs that appear in the base vocab but never in the corpus are dropped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple


_STAMP_SPECIAL_PREFIX = ("[J_", "[B_", "[S_")
_STAMP_SPECIAL_EXACT = {"[Ring]", "[END]", "[BACK]"}


def is_stamp_special(token: str) -> bool:
    if token in _STAMP_SPECIAL_EXACT:
        return True
    return token.startswith(_STAMP_SPECIAL_PREFIX) and token.endswith("]")


def load_base_vocab(path: Path) -> Tuple[Dict[str, Tuple[int, int]], dict]:
    entries: Dict[str, Tuple[int, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        header = json.loads(f.readline())
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            smi, a, fr = parts
            entries[smi] = (int(a), int(fr))
    return entries, header


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base PS-VAE vocab.txt path.")
    ap.add_argument("--token-counts", required=True, help="token_counts.tsv from profile_stamp_vocab_freq.py.")
    ap.add_argument("--top-k", type=int, required=True, help="Number of motifs to keep (by usage freq).")
    ap.add_argument("--output", required=True, help="Output vocab.txt path.")
    args = ap.parse_args()

    base_path = Path(args.base)
    counts_path = Path(args.token_counts)
    out_path = Path(args.output)

    entries, header = load_base_vocab(base_path)

    motif_counts = []
    with counts_path.open("r", encoding="utf-8") as f:
        first = f.readline()
        if not first.startswith("token\t"):
            raise SystemExit(f"Unexpected header in {counts_path}: {first!r}")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            tok, c = parts[0], int(parts[1])
            if is_stamp_special(tok):
                continue
            if tok not in entries:
                # Encoder fallback to atom-level may emit single atoms not in
                # base vocab; ignore them — they're degenerate seeds.
                continue
            motif_counts.append((tok, c))

    motif_counts.sort(key=lambda x: -x[1])
    kept = motif_counts[: args.top_k]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        # Write entries sorted by base-vocab convention: -freq, -atoms, smi
        # but here use the EMPIRICAL freq from the corpus profile, not the
        # mining freq, to keep the file freq column meaningful.
        kept_sorted = sorted(
            kept,
            key=lambda x: (-x[1], -entries[x[0]][0], x[0]),
        )
        for smi, c in kept_sorted:
            atom_num, _orig_freq = entries[smi]
            f.write(f"{smi}\t{atom_num}\t{c}\n")

    n_total_motifs = len(motif_counts)
    cum_total = sum(c for _, c in motif_counts)
    cum_kept = sum(c for _, c in kept)
    coverage = cum_kept / cum_total if cum_total > 0 else 0.0
    print(f"Base vocab motifs: {len(entries)}")
    print(f"Motifs seen in corpus: {n_total_motifs}")
    print(f"Kept top-{args.top_k}: {len(kept)}")
    print(f"Motif-only coverage retained: {coverage:.4%} of {cum_total} occurrences")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
