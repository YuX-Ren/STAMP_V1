"""Prefix-slice a PS-VAE vocab into smaller nested vocabs.

Relies on the `*.history.json` sibling emitted by `mine_principal_subgraphs`,
which records the exact insertion order (seeds first, then BPE merges in
greedy merge order). The vocab at size K is the first K entries of that
ordered list, filtered to those present in the base vocab with their
original (atom_num, freq) tuples.

Usage:
    python scripts/benchmark/slice_psvae_vocab.py \
        --base results/vocab/stamp_psvae_chembl34_8192/vocab.txt \
        --sizes 2000 4096 8192 \
        --output-root results/vocab
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_base_vocab(base_path: Path) -> Tuple[Dict[str, Tuple[int, int]], dict]:
    entries: Dict[str, Tuple[int, int]] = {}
    header = {}
    with base_path.open("r", encoding="utf-8") as f:
        header = json.loads(f.readline())
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            smi, a, fr = parts
            entries[smi] = (int(a), int(fr))
    return entries, header


def load_history(history_path: Path) -> List[str]:
    data = json.loads(history_path.read_text(encoding="utf-8"))
    return data["merge_order"]


def write_sliced_vocab(
    entries: Dict[str, Tuple[int, int]],
    ordered_smis: List[str],
    header: dict,
    target_size: int,
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sliced = ordered_smis[:target_size]
    # Keep same sort order as mine writes out (frequency descending).
    sliced_sorted = sorted(
        sliced,
        key=lambda x: (-entries[x][1], -entries[x][0], x),
    )
    with output_path.open("w", encoding="utf-8") as fout:
        fout.write(json.dumps(header) + "\n")
        for smi in sliced_sorted:
            a, fr = entries[smi]
            fout.write(f"{smi}\t{a}\t{fr}\n")
    return len(sliced_sorted)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="psvae")
    args = parser.parse_args()

    history_path = args.base.with_suffix(args.base.suffix + ".history.json")
    if not history_path.exists():
        print(f"ERROR: missing {history_path}", file=sys.stderr)
        return 1

    entries, header = load_base_vocab(args.base)
    ordered = load_history(history_path)
    print(
        f"[slice] base vocab={len(entries)}, history length={len(ordered)}",
        flush=True,
    )

    for sz in args.sizes:
        out_dir = args.output_root / f"stamp_{args.tag}_{sz}"
        out_path = out_dir / "vocab.txt"
        written = write_sliced_vocab(entries, ordered, header, sz, out_path)
        print(f"[slice] -> {out_path}: {written} motifs (requested {sz})", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
