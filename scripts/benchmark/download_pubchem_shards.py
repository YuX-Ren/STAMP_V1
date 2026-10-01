#!/usr/bin/env python3
"""Download N parquet shards from hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC,
extract canonical SMILES, and write to a single .smi file.

Each shard is ~1M rows, ~168 MB. The dataset is globally shuffled, so taking
the first N shards yields an i.i.d. sample of size ~N million.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq


REPO_ID = "hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shards", type=int, default=10, help="Number of shards (~1M each).")
    ap.add_argument("--start", type=int, default=1, help="Starting shard index (1..124).")
    ap.add_argument("--cache-dir", required=True, help="Where to download parquet shards.")
    ap.add_argument("--output", required=True, help="Output .smi path (one canonical SMILES per line).")
    ap.add_argument("--column", default="SMILES_Canonical")
    ap.add_argument("--keep-parquet", action="store_true", help="Do not delete parquet after extraction.")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_dropped = 0
    t_start = time.time()
    with out.open("w", encoding="utf-8") as fout:
        for i in range(args.start, args.start + args.n_shards):
            shard = f"data/shard_{i:03d}.parquet"
            t0 = time.time()
            local = hf_hub_download(
                repo_id=REPO_ID,
                filename=shard,
                repo_type="dataset",
                local_dir=str(cache),
            )
            tbl = pq.read_table(local, columns=[args.column])
            col = tbl.column(args.column).to_pylist()
            kept = 0
            for smi in col:
                if smi is None:
                    n_dropped += 1
                    continue
                s = smi.strip()
                if not s:
                    n_dropped += 1
                    continue
                fout.write(s + "\n")
                kept += 1
            n_written += kept
            print(
                f"[{i:03d}/{args.start + args.n_shards - 1}] {Path(local).name}: "
                f"kept={kept} dropped={len(col) - kept} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
            if not args.keep_parquet:
                Path(local).unlink(missing_ok=True)

    print(
        f"\nDone. wrote={n_written} dropped={n_dropped} "
        f"total={time.time() - t_start:.1f}s -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
