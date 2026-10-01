"""Extract SMILES from one arrow shard → one parquet (smiles only).

Designed for parallel invocation: each call processes one shard independently."""
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.ipc as ipc
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-arrow", required=True)
    p.add_argument("--out-parquet", required=True)
    p.add_argument("--smiles-col", default="input")
    args = p.parse_args()

    t0 = time.time()
    try:
        with pa.OSFile(args.in_arrow, "rb") as f:
            tbl = ipc.RecordBatchFileReader(f).read_all()
    except pa.ArrowInvalid:
        with pa.OSFile(args.in_arrow, "rb") as f:
            tbl = ipc.RecordBatchStreamReader(f).read_all()
    n = tbl.num_rows
    smi_arr = tbl.column(args.smiles_col)
    out_tbl = pa.table([smi_arr], schema=pa.schema([("smiles", pa.string())]))
    pq.write_table(out_tbl, args.out_parquet, compression="snappy")
    print(f"  {args.in_arrow} → {args.out_parquet}  rows={n:,}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
