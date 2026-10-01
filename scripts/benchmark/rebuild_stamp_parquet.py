#!/usr/bin/env python3
"""Rebuild STAMP parquet shards from SMILES parquet inputs.

The output schema is intentionally minimal:
  smiles, stamp[, split_random]

Condition columns can be added afterwards with
``precompute_condition_tokens_standalone.py``. Keeping these two stages
separate makes codec failures easier to audit.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


REP = None


def _init_worker(repo_root: str, stamp_vocab: str) -> None:
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from stamp.benchmark.representations import build_representation

    global REP
    REP = build_representation("stamp", stamp_vocab=stamp_vocab)


def _encode_one(smiles: Any) -> tuple[str, str, str]:
    smi = str(smiles or "").strip()
    if not smi:
        return "", "", "empty"
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi, "", "rdkit_invalid"
    n_heavy = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    if n_heavy < 3 or n_heavy > 100:
        return smi, "", "heavy_atom_filter"
    try:
        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        stamp = REP.encode(canon)
        return canon, stamp, ""
    except Exception as exc:
        return smi, "", f"{type(exc).__name__}: {exc}"


def _split_for_smiles(smiles: str, val_fraction: float) -> str:
    if val_fraction <= 0.0:
        return "train"
    h = hashlib.blake2b(smiles.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(h, "big") / float(1 << 64)
    return "val" if value < val_fraction else "train"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stamp-vocab", required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--add-split-random",
        action="store_true",
        help="Add a stable train/val split_random column for non train_/val_ source files.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        print(f"[skip] exists: {output_path}", flush=True)
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(input_path)
    if args.smiles_column not in pf.schema.names:
        raise KeyError(f"{input_path} missing {args.smiles_column!r}; columns={pf.schema.names}")

    fields = [("smiles", pa.string()), ("stamp", pa.string())]
    if args.add_split_random:
        fields.append(("split_random", pa.string()))
    schema = pa.schema(fields)

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    writer = pq.ParquetWriter(tmp_path, schema, compression=str(args.compression))
    total = 0
    ok = 0
    failures: dict[str, int] = {}
    remaining = int(args.max_rows) if int(args.max_rows) > 0 else None
    t0 = time.time()
    try:
        with Pool(
            int(args.n_workers),
            initializer=_init_worker,
            initargs=(str(args.repo_root), str(args.stamp_vocab)),
        ) as pool:
            for batch in pf.iter_batches(batch_size=int(args.batch_size), columns=[args.smiles_column]):
                if remaining is not None and remaining <= 0:
                    break
                smiles = batch.column(0).to_pylist()
                if remaining is not None and len(smiles) > remaining:
                    smiles = smiles[:remaining]
                chunksize = max(1, len(smiles) // max(1, int(args.n_workers) * 4))
                rows = pool.imap(_encode_one, smiles, chunksize=chunksize)
                out_smiles: list[str] = []
                out_stamp: list[str] = []
                out_split: list[str] = []
                for canon, stamp, error in rows:
                    total += 1
                    if error:
                        failures[error] = failures.get(error, 0) + 1
                        continue
                    ok += 1
                    out_smiles.append(canon)
                    out_stamp.append(stamp)
                    if args.add_split_random:
                        out_split.append(_split_for_smiles(canon, float(args.val_fraction)))
                if out_smiles:
                    payload: dict[str, Any] = {"smiles": out_smiles, "stamp": out_stamp}
                    if args.add_split_random:
                        payload["split_random"] = out_split
                    writer.write_table(pa.Table.from_pydict(payload, schema=schema))
                if remaining is not None:
                    remaining -= len(smiles)
                if total and total % max(1, int(args.batch_size) * 10) == 0:
                    rate = total / max(1e-6, time.time() - t0)
                    print(
                        f"[rebuild] {input_path.name} rows={total:,} ok={ok:,} "
                        f"({ok / max(1, total) * 100:.2f}%) rate={rate:.0f}/s",
                        flush=True,
                    )
        writer.close()
        writer = None
        os.replace(tmp_path, output_path)
    finally:
        if writer is not None:
            writer.close()
        if tmp_path.exists():
            tmp_path.unlink()

    print(
        f"[done] {input_path} -> {output_path} rows={total:,} ok={ok:,} "
        f"({ok / max(1, total) * 100:.2f}%) failures={failures}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
