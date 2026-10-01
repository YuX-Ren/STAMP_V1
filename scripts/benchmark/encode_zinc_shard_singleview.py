#!/usr/bin/env python3
"""Single-view STAMP encoding of one ZINC15 SMILES parquet shard.

Reads a raw parquet with a `smiles` column, applies non-ring-aware
`STAMPRepresentation` guided by a legacy motif vocab, drops rows with
<3 or >100 heavy atoms, writes {out_parquet} with columns [smiles, stamp].

Multi-processing pool inside a single PBS task; use PBS job arrays to
parallelize across shards.
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger


def init_worker(stamp_vocab_path: str) -> None:
    sys.path.insert(0, "/home/svu/yuxuan.ren/STAMP_V1.5")
    RDLogger.DisableLog("rdApp.*")
    from stamp.benchmark.representations import STAMPRepresentation
    global REP
    REP = STAMPRepresentation(vocab_path=stamp_vocab_path)


def encode_smi(smi: str) -> str:
    if not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if n_heavy < 3 or n_heavy > 100:
        return ""
    try:
        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        text = REP.encode(canon)
        return text or ""
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-parquet", required=True, help="Input parquet with a smiles column.")
    ap.add_argument("--out-parquet", required=True, help="Output parquet path.")
    ap.add_argument("--stamp-vocab", required=True)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--n-workers", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=20_000)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all rows.")
    args = ap.parse_args()

    RDLogger.DisableLog("rdApp.*")
    in_path = Path(args.in_parquet).resolve()
    out_path = Path(args.out_parquet).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(str(in_path))
    print(f"[in]  {in_path}  rows={pf.metadata.num_rows:,} rg={pf.metadata.num_row_groups}", flush=True)
    print(f"[out] {out_path}", flush=True)
    print(f"[cfg] n_workers={args.n_workers}  batch_size={args.batch_size}  vocab={args.stamp_vocab}", flush=True)

    schema = pa.schema([("smiles", pa.string()), ("stamp", pa.string())])
    writer = pq.ParquetWriter(str(out_path), schema, compression="snappy")

    t0 = time.time()
    total = 0
    ok = 0
    with Pool(args.n_workers, initializer=init_worker, initargs=(args.stamp_vocab,)) as pool:
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=[args.smiles_col]):
            smis = batch.column(args.smiles_col).to_pylist()
            if not smis:
                continue
            chunksize = max(1, len(smis) // (args.n_workers * 4))
            stamps = list(pool.imap(encode_smi, smis, chunksize=chunksize))
            keep_smis = [s for s, t in zip(smis, stamps) if t]
            keep_st = [t for t in stamps if t]
            if keep_smis:
                writer.write_table(pa.Table.from_pydict({"smiles": keep_smis, "stamp": keep_st}, schema=schema))
            total += len(smis)
            ok += len(keep_smis)
            if total % (args.batch_size * 5) == 0:
                dt = max(1e-6, time.time() - t0)
                print(f"  {total:>10,}  ok={ok:>10,} ({ok/total*100:.1f}%)  rate={total/dt:.0f}/s", flush=True)
            if args.max_rows > 0 and total >= args.max_rows:
                break
    writer.close()
    dt = time.time() - t0
    print(f"[done] rows_in={total:,}  rows_out={ok:,} ({ok/max(1,total)*100:.1f}%)  wall={dt:.1f}s  rate={total/max(1e-6,dt):.0f}/s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
