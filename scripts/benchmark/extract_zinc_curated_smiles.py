"""Extract SMILES from ZINC-Curated HF arrow shards into one parquet (smiles only).

Reads the whole arrow shard at once via RecordBatchFileReader.read_all()
(much faster than per-batch iteration), selects the `input` column, writes to
the output parquet with snappy compression."""
import argparse, time, glob
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.ipc as ipc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out-train", required=True)
    p.add_argument("--out-valid", required=True)
    p.add_argument("--max-rows-train", type=int, default=0)
    args = p.parse_args()

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("smiles", pa.string())])

    for split, out_path in [("train", args.out_train), ("validation", args.out_valid)]:
        print(f"\n=== {split} ===", flush=True)
        shards = sorted(glob.glob(f"{args.src}/{split}/data-*.arrow"))
        print(f"  {len(shards)} shards", flush=True)
        writer = pq.ParquetWriter(out_path, schema, compression="snappy")
        t0 = time.time(); n_total = 0
        for i, shard in enumerate(shards):
            try:
                tbl = ipc.RecordBatchFileReader(pa.OSFile(shard, "rb")).read_all()
            except pa.ArrowInvalid:
                tbl = ipc.RecordBatchStreamReader(pa.OSFile(shard, "rb")).read_all()
            # Just keep the SMILES column, rename to 'smiles'
            smi_arr = tbl.column("input")
            out_tbl = pa.table([smi_arr], schema=schema)
            writer.write_table(out_tbl)
            n_total += tbl.num_rows
            if (i + 1) % 5 == 0 or (i + 1) == len(shards):
                rate = n_total / max(1, time.time()-t0)
                print(f"  shard {i+1:>3}/{len(shards)}  cum={n_total:>12,}  rate={rate:,.0f}/s", flush=True)
            if split == "train" and args.max_rows_train > 0 and n_total >= args.max_rows_train:
                break
        writer.close()
        print(f"  done: {n_total:,} rows  ({time.time()-t0:.0f}s)  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
