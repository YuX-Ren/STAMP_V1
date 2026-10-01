#!/usr/bin/env python3
"""Clean a raw PubChem .smi dump and produce pretraining splits + vocab-mining subset.

Cleaning steps (parallel):
    1. RDKit-parse each SMILES.
    2. If multi-component (salt/hydrate/etc.), keep the largest fragment.
    3. Re-canonicalize (isomeric).
    4. Drop None / empty / atomcount==0.
    5. Deduplicate the resulting canonical strings.

Outputs (under <output-dir>/):
    train.smi              full train set (one canonical SMILES per line)
    valid.smi              held-out validation set for MLM PPL
    vocab_mining_5M.smi    random 5M subset of train for STAMP vocab mining
    split_summary.json     counts, args, seed
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path
from typing import List, Optional


_W_CANON = None


def _init_worker() -> None:
    global _W_CANON
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from stamp.benchmark.common import canonicalize_smiles
    _W_CANON = canonicalize_smiles


def _clean_one(smi: str) -> Optional[str]:
    return _W_CANON(smi, strip_salts=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to raw .smi file.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--valid-size", type=int, default=100_000)
    ap.add_argument("--vocab-subset-size", type=int, default=5_000_000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunksize", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-mols", type=int, default=0, help="0 = all.")
    ap.add_argument("--progress-every", type=int, default=500_000)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw SMILES from {in_path} ...", flush=True)

    def _iter_lines():
        n = 0
        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                yield s
                n += 1
                if args.max_mols > 0 and n >= args.max_mols:
                    return

    seen = set()
    cleaned: List[str] = []
    n_in = 0
    n_drop = 0
    n_dup = 0
    t0 = time.time()

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, initializer=_init_worker) as pool:
        for canon in pool.imap_unordered(_clean_one, _iter_lines(), chunksize=args.chunksize):
            n_in += 1
            if canon is None:
                n_drop += 1
            elif canon in seen:
                n_dup += 1
            else:
                seen.add(canon)
                cleaned.append(canon)
            if n_in % args.progress_every == 0:
                print(
                    f"[{n_in:>10}] kept={len(cleaned)} drop={n_drop} dup={n_dup} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )

    print(
        f"Cleaning done: in={n_in} kept={len(cleaned)} drop={n_drop} dup={n_dup} "
        f"elapsed={time.time() - t0:.1f}s",
        flush=True,
    )

    rng = random.Random(args.seed)
    rng.shuffle(cleaned)

    if len(cleaned) <= args.valid_size:
        print(f"ERROR: cleaned={len(cleaned)} <= valid_size={args.valid_size}", file=sys.stderr)
        return 1

    valid = cleaned[: args.valid_size]
    train = cleaned[args.valid_size :]

    train_path = out_dir / "train.smi"
    valid_path = out_dir / "valid.smi"
    with train_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(train) + "\n")
    with valid_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(valid) + "\n")

    vocab_subset_path: Optional[Path] = None
    if args.vocab_subset_size > 0:
        k = min(args.vocab_subset_size, len(train))
        vocab_subset = rng.sample(train, k)
        vocab_subset_path = out_dir / f"vocab_mining_{k // 1_000_000}M.smi"
        with vocab_subset_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(vocab_subset) + "\n")

    summary = {
        "input": str(in_path),
        "output_dir": str(out_dir),
        "seed": args.seed,
        "n_input_lines": n_in,
        "n_cleaned_unique": len(cleaned),
        "n_invalid_or_empty": n_drop,
        "n_duplicates": n_dup,
        "n_train": len(train),
        "n_valid": len(valid),
        "vocab_subset_size": (len(vocab_subset) if vocab_subset_path else 0),
        "vocab_subset_path": (str(vocab_subset_path) if vocab_subset_path else None),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "elapsed_seconds": time.time() - t0,
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
