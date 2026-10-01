#!/usr/bin/env python3
"""Compute the full set of RDKit Descriptors._descList (~210) for a SMILES corpus.

Output:
    <output-dir>/descriptors.parquet  (n_mols, 210) float32, plus a 'smiles' column
    <output-dir>/desc_stats.json      per-descriptor mean/std/n_finite over the corpus
    <output-dir>/desc_names.json      ordered descriptor names
    <output-dir>/summary.json         counts + elapsed
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


_W_DESCS = None
_W_NAMES = None


def _init_worker() -> None:
    global _W_DESCS, _W_NAMES
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem import Descriptors
    _W_DESCS = list(Descriptors._descList)
    _W_NAMES = [n for n, _ in _W_DESCS]


def _compute_one(smi: str) -> Tuple[str, List[float]]:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi, [float("nan")] * len(_W_DESCS)
    out = []
    for _, fn in _W_DESCS:
        try:
            v = float(fn(mol))
            if not math.isfinite(v):
                v = float("nan")
        except Exception:
            v = float("nan")
        out.append(v)
    return smi, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help=".smi file (one SMILES per line)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--chunksize", type=int, default=2048)
    ap.add_argument("--max-mols", type=int, default=0, help="0 = all")
    ap.add_argument("--rows-per-shard", type=int, default=1_000_000,
                    help="Number of rows per parquet shard (memory-friendly).")
    ap.add_argument("--progress-every", type=int, default=500_000)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # discover descriptor names from a single init
    _init_worker()
    desc_names = list(_W_NAMES)
    n_desc = len(desc_names)
    (out_dir / "desc_names.json").write_text(json.dumps(desc_names, indent=2))
    print(f"n_descriptors = {n_desc}", flush=True)

    in_path = Path(args.input)

    def _iter_smiles():
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

    t0 = time.time()
    n_processed = 0
    n_invalid = 0
    sums = np.zeros(n_desc, dtype=np.float64)
    sum_sqs = np.zeros(n_desc, dtype=np.float64)
    n_finite = np.zeros(n_desc, dtype=np.int64)

    # Use streamed parquet writer with sharding for memory friendliness.
    schema = pa.schema(
        [pa.field("smiles", pa.string())]
        + [pa.field(name, pa.float32()) for name in desc_names]
    )
    shard_idx = 0
    rows_in_shard = 0
    writer = None

    def _open_writer():
        nonlocal writer, shard_idx, rows_in_shard
        if writer is not None:
            writer.close()
            shard_idx += 1
            rows_in_shard = 0
        path = out_dir / f"descriptors_shard_{shard_idx:03d}.parquet"
        writer = pq.ParquetWriter(path, schema, compression="zstd")
        return path

    _open_writer()

    smis_buf: List[str] = []
    vals_buf: List[List[float]] = []
    BUF_FLUSH = 50_000

    def _flush():
        nonlocal smis_buf, vals_buf, rows_in_shard
        if not smis_buf:
            return
        cols = {"smiles": pa.array(smis_buf, type=pa.string())}
        arr = np.asarray(vals_buf, dtype=np.float32)
        for j, name in enumerate(desc_names):
            cols[name] = pa.array(arr[:, j], type=pa.float32())
        tbl = pa.table(cols, schema=schema)
        writer.write_table(tbl)
        rows_in_shard += len(smis_buf)
        smis_buf = []
        vals_buf = []
        if rows_in_shard >= args.rows_per_shard:
            _open_writer()

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, initializer=_init_worker) as pool:
        for smi, vec in pool.imap_unordered(
            _compute_one, _iter_smiles(), chunksize=args.chunksize
        ):
            n_processed += 1
            arr = np.asarray(vec, dtype=np.float64)
            finite = np.isfinite(arr)
            if not finite.any():
                n_invalid += 1
            sums[finite] += arr[finite]
            sum_sqs[finite] += arr[finite] ** 2
            n_finite[finite] += 1
            smis_buf.append(smi)
            vals_buf.append(vec)
            if len(smis_buf) >= BUF_FLUSH:
                _flush()
            if n_processed % args.progress_every == 0:
                rate = n_processed / max(1.0, (time.time() - t0))
                print(
                    f"[{n_processed:>10}] invalid={n_invalid} rate={rate:.0f} mols/s "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
    _flush()
    if writer is not None:
        writer.close()

    # per-descriptor stats
    means = np.where(n_finite > 0, sums / np.maximum(n_finite, 1), 0.0)
    var = np.where(
        n_finite > 1,
        (sum_sqs - n_finite * means ** 2) / np.maximum(n_finite - 1, 1),
        1.0,
    )
    var = np.where(var < 1e-12, 1.0, var)
    stds = np.sqrt(var)

    stats = {
        "names": desc_names,
        "n_finite": n_finite.tolist(),
        "mean": means.tolist(),
        "std": stds.tolist(),
        "n_total": n_processed,
    }
    (out_dir / "desc_stats.json").write_text(json.dumps(stats))

    # Identify constant / near-useless descriptors (zero std after filtering)
    bad = [
        (name, int(n_finite[i]), float(stds[i]))
        for i, name in enumerate(desc_names)
        if n_finite[i] < 0.5 * n_processed or stds[i] < 1e-6
    ]
    summary = {
        "input": str(in_path),
        "n_processed": n_processed,
        "n_all_invalid": n_invalid,
        "n_descriptors": n_desc,
        "n_low_quality_desc": len(bad),
        "low_quality_desc_examples": bad[:20],
        "elapsed_seconds": time.time() - t0,
        "shards": shard_idx + 1,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
