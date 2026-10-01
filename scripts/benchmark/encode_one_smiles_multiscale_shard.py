#!/usr/bin/env python3
"""Encode one SMILES parquet shard into multi-scale STAMP parquet.

Each output row stores:
  - smiles
  - a CaMS-compatible ``stamp`` field with [CONCAT] separators
  - one STAMP column per scale
  - a concatenated ``stamp_multiscale`` text field
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def _parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def init_worker(scale_vocabs_csv: str, scale_names_csv: str, force_ring_aware_encode: bool):
    sys.path.insert(0, str(__file__.split("/scripts/benchmark/")[0]))
    from stamp.benchmark.multiscale_stamp import MultiScaleSTAMPEncoder

    vocab_paths = _parse_csv(scale_vocabs_csv)
    scale_names = _parse_csv(scale_names_csv)
    global MS_ENCODER
    MS_ENCODER = MultiScaleSTAMPEncoder(
        vocab_paths,
        scale_names=scale_names,
        force_ring_aware_encode=bool(force_ring_aware_encode),
    )


def encode_smi(smi: str):
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if n_heavy < 3 or n_heavy > 100:
        return None
    try:
        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        scale_texts = MS_ENCODER.encode_scales(canon)
        concat_tokens = ["[BOS]"]
        for i, text in enumerate(scale_texts):
            concat_tokens.extend(str(text).split())
            if i != len(scale_texts) - 1:
                concat_tokens.append("[CONCAT]")
        concat_tokens.append("[EOS]")
        joined = []
        for scale_tok, text in zip(MS_ENCODER.scale_tokens, scale_texts):
            joined.append(scale_tok)
            joined.append(text)
        return {
            "smiles": canon,
            "stamp": " ".join(concat_tokens),
            "stamp_multiscale": " ".join(joined),
            **{f"stamp_{name}": text for name, text in zip(MS_ENCODER.scale_names, scale_texts)},
        }
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-parquet", required=True)
    p.add_argument("--out-parquet", required=True)
    p.add_argument("--smiles-column", default="smiles")
    p.add_argument("--scale-vocabs", required=True, help="Comma-separated vocab paths, ordered fine-to-coarse.")
    p.add_argument("--scale-names", required=True, help="Comma-separated scale names, same length as --scale-vocabs.")
    p.add_argument("--force-ring-aware-encode", action="store_true", help="Override vocab header and emit [Ring] closures.")
    p.add_argument("--n-workers", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=20000)
    args = p.parse_args()

    scale_vocabs = _parse_csv(args.scale_vocabs)
    scale_names = _parse_csv(args.scale_names)
    if len(scale_vocabs) != len(scale_names):
        raise ValueError("--scale-vocabs and --scale-names must have the same length.")
    if not scale_vocabs:
        raise ValueError("At least one scale vocab is required.")

    schema = pa.schema(
        [("smiles", pa.string()), ("stamp", pa.string()), ("stamp_multiscale", pa.string())]
        + [(f"stamp_{name}", pa.string()) for name in scale_names]
    )
    writer = pq.ParquetWriter(args.out_parquet, schema, compression="snappy")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    pf = pq.ParquetFile(args.in_parquet)
    n_total_rows = pf.metadata.num_rows if pf.metadata else None
    t0 = time.time()
    total = 0
    ok = 0
    shard_name = args.in_parquet.split("/")[-1]
    bar = tqdm(total=n_total_rows, desc=shard_name, unit="mol", mininterval=2.0) if tqdm and n_total_rows else None
    with Pool(
        args.n_workers,
        initializer=init_worker,
        initargs=(",".join(scale_vocabs), ",".join(scale_names), bool(args.force_ring_aware_encode)),
    ) as pool:
        for batch in pf.iter_batches(batch_size=args.batch_size):
            df = batch.to_pandas()
            smis = df[args.smiles_column].fillna("").tolist()
            chunksize = max(1, len(smis) // max(1, args.n_workers * 4))
            rows = [r for r in pool.imap(encode_smi, smis, chunksize=chunksize) if r is not None]
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            total += len(smis)
            ok += len(rows)
            if bar:
                bar.update(len(smis))
            else:
                pct = total / max(1, n_total_rows or total) * 100
                print(
                    f"  [{shard_name}] {total:,}/{n_total_rows or '?'} ({pct:.0f}%) "
                    f"ok={ok:,} ({ok/max(1,total)*100:.1f}%) rate={total/max(1e-9,time.time()-t0):.0f}/s",
                    flush=True,
                )
    if bar:
        bar.close()
    writer.close()
    print(
        f"DONE {args.in_parquet} -> {args.out_parquet} rows={total:,} ok={ok:,} "
        f"({ok/max(1,total)*100:.1f}%) {time.time()-t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
