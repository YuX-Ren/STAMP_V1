"""STAMP-encode one SMILES parquet shard → output STAMP parquet.

For parallel invocation: each call processes one input shard."""
import argparse, time, sys
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
        return REP.encode(canon)
    except Exception:
        return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-parquet", required=True)
    p.add_argument("--out-parquet", required=True)
    p.add_argument("--stamp-vocab", required=True)
    p.add_argument("--n-workers", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=20000)
    args = p.parse_args()

    schema = pa.schema([("smiles", pa.string()), ("stamp", pa.string())])
    writer = pq.ParquetWriter(args.out_parquet, schema, compression="snappy")

    try: from tqdm import tqdm
    except ImportError: tqdm = None

    pf = pq.ParquetFile(args.in_parquet)
    n_total_rows = pf.metadata.num_rows if pf.metadata else None
    t0 = time.time(); total = 0; ok = 0
    shard_name = args.in_parquet.split('/')[-1]
    bar = tqdm(total=n_total_rows, desc=shard_name, unit='mol', mininterval=2.0) if tqdm and n_total_rows else None
    with Pool(args.n_workers, initializer=init_worker, initargs=(args.stamp_vocab,)) as pool:
        for batch in pf.iter_batches(batch_size=args.batch_size):
            df = batch.to_pandas()
            smis = df["smiles"].fillna("").tolist()
            chunksize = max(1, len(smis) // (args.n_workers * 4))
            stamps = list(pool.imap(encode_smi, smis, chunksize=chunksize))
            valid_mask = [bool(s) for s in stamps]
            valid_smi = [s for s, m in zip(smis, valid_mask) if m]
            valid_st = [s for s, m in zip(stamps, valid_mask) if m]
            if valid_smi:
                writer.write_table(pa.Table.from_pydict({"smiles": valid_smi, "stamp": valid_st}, schema=schema))
            total += len(smis); ok += len(valid_smi)
            if bar: bar.update(len(smis))
            else:
                pct = total / max(1, n_total_rows or total) * 100
                print(f"  [{shard_name}] {total:,}/{n_total_rows or '?'} ({pct:.0f}%) ok={ok:,} ({ok/total*100:.1f}%) rate={total/(time.time()-t0):.0f}/s", flush=True)
    if bar: bar.close()

    writer.close()
    print(f"DONE {args.in_parquet} → {args.out_parquet}  rows={total:,} ok={ok:,} ({ok/max(1,total)*100:.1f}%) {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
