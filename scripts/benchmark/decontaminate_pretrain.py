"""Decontaminate the STAMP pretrain parquet: drop rows whose InChIKey skeleton
(first 14 chars) matches any GenMol fragments.csv target. Original parquet
is preserved; output is a new file.

Usage:
  python scripts/benchmark/decontaminate_pretrain.py \\
    --in /scratch/yuxuan.ren/data/dataset_stamp.parquet \\
    --targets /home/svu/yuxuan.ren/genmol_fragments.csv \\
    --out /scratch/yuxuan.ren/data/dataset_stamp_decontam.parquet
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=200_000)
    args = ap.parse_args()

    targets = pd.read_csv(args.targets)
    skeleton_set = set()
    full_set = set()
    for _, row in targets.iterrows():
        smi = row["smiles"]
        m = Chem.MolFromSmiles(smi)
        if m is None: continue
        ik = MolToInchiKey(m)
        full_set.add(ik)
        skeleton_set.add(ik[:14])
    print(f"Target skeletons: {len(skeleton_set)}  full keys: {len(full_set)}")

    pf = pq.ParquetFile(args.inp)
    schema = pf.schema_arrow
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(args.out, schema, compression="snappy")

    t0 = time.time()
    total = 0; kept = 0; dropped = 0
    drop_log = []
    for batch in pf.iter_batches(batch_size=args.batch_size):
        df = batch.to_pandas()
        # Filter on InChIKey skeleton
        mask = df["inchi_key"].fillna("").str[:14].isin(skeleton_set)
        n_drop = int(mask.sum())
        if n_drop > 0:
            for _, drow in df[mask].iterrows():
                drop_log.append((drow.get("source_db", "?"), drow["inchi_key"],
                                  drow.get("std_smiles", "")))
            df = df[~mask]
        kept += len(df); dropped += n_drop; total += len(batch)
        writer.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False))
        if total % (args.batch_size * 5) == 0:
            print(f"  scanned {total:>10,}  kept {kept:>10,}  dropped {dropped:>3} ({time.time()-t0:.1f}s)", flush=True)

    writer.close()
    print(f"\nDone. kept {kept:,} / scanned {total:,}  dropped {dropped} ({time.time()-t0:.1f}s)")
    print(f"Wrote {args.out}")
    if drop_log:
        print(f"\nDropped rows ({len(drop_log)}):")
        for db, ik, smi in drop_log[:60]:
            print(f"  [{db}] {ik}  {smi[:80]}")


if __name__ == "__main__":
    main()
