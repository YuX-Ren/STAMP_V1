"""STAMP-encode a SMILES parquet, sharded for parallel processing across PBS jobs.

Each invocation processes one shard (--shard-idx of --n-shards) and writes
output to {out_prefix}_shard{idx:03d}.parquet. After all shards finish, use
pyarrow concat to produce a single parquet."""
import argparse, json, time, sys, os
from pathlib import Path
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')


def init_worker(stamp_vocab_path):
    sys.path.insert(0, '/home/svu/yuxuan.ren/STAMP_V1.5')
    from stamp.benchmark.representations import build_representation
    global REP
    REP = build_representation('stamp', stamp_vocab=stamp_vocab_path)


def encode_smi(smi):
    if not smi: return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return ""
    n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if n_heavy < 3 or n_heavy > 100: return ""
    try:
        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        text = REP.encode(canon)
        return text
    except Exception:
        return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--stamp-vocab", required=True)
    p.add_argument("--shard-idx", type=int, required=True)
    p.add_argument("--n-shards", type=int, required=True)
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--n-workers", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=10_000)
    p.add_argument("--max-rows", type=int, default=0)
    args = p.parse_args()

    out_path = f"{args.out_prefix}_shard{args.shard_idx:03d}.parquet"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(args.inp)
    schema = pa.schema([("smiles", pa.string()), ("stamp", pa.string())])
    writer = pq.ParquetWriter(out_path, schema, compression="snappy")

    t0 = time.time()
    total = 0; ok = 0; row_idx = 0
    with Pool(args.n_workers, initializer=init_worker, initargs=(args.stamp_vocab,)) as pool:
        for batch in pf.iter_batches(batch_size=args.batch_size):
            df = batch.to_pandas()
            smis = df[args.smiles_col].fillna("").tolist()
            # Filter to this shard: take rows where row_idx % n_shards == shard_idx
            this_shard_mask = [(row_idx + i) % args.n_shards == args.shard_idx for i in range(len(smis))]
            row_idx += len(smis)
            shard_smis = [s for s, m in zip(smis, this_shard_mask) if m]
            if not shard_smis: continue
            chunksize = max(1, len(shard_smis) // (args.n_workers * 4))
            stamps = list(pool.imap(encode_smi, shard_smis, chunksize=chunksize))
            valid_smi = [s for s, t in zip(shard_smis, stamps) if t]
            valid_st = [t for t in stamps if t]
            if valid_smi:
                writer.write_table(pa.Table.from_pydict({"smiles": valid_smi, "stamp": valid_st}, schema=schema))
            total += len(shard_smis); ok += len(valid_smi)
            if total % (args.batch_size * 10) == 0:
                rate = total / max(1, time.time()-t0)
                print(f"  shard{args.shard_idx} {total:>10,}  ok={ok:>10,} ({ok/total*100:.1f}%)  rate={rate:.0f}/s", flush=True)
            if args.max_rows > 0 and total >= args.max_rows: break

    writer.close()
    print(f"\nshard{args.shard_idx} done. {total:,} rows, {ok:,} ok ({ok/max(1,total)*100:.1f}%) in {time.time()-t0:.0f}s")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
