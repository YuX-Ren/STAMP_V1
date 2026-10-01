#!/usr/bin/env python3
"""Encode IBM MoLFormer PubChem/ZINC SMILES into single-scale STAMP shards.

The data source stays aligned with MolFormer: PubChem canonical SMILES plus
ZINC15 SMILES, with the original max-SMILES-length filter. Only the molecular
representation is changed from SMILES tokens to STAMP tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REP = None
WORKER_CFG: dict[str, object] = {}


def _init_worker(stamp_vocab: str, worker_cfg_json: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from stamp.benchmark.representations import STAMPRepresentation

    global REP, WORKER_CFG
    REP = STAMPRepresentation(vocab_path=stamp_vocab, force_ring_aware_encode=True)
    WORKER_CFG = json.loads(worker_cfg_json)


def _parse_smiles_from_line(raw: bytes | str, source: str) -> str | None:
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
    text = text.strip()
    if not text:
        return None
    parts = text.split()
    if not parts:
        return None
    if source == "zinc":
        if parts[0].lower() == "smiles":
            return None
        return parts[0]
    if source == "pubchem":
        if parts[0].lower() in {"cid", "smiles"}:
            return None
        if len(parts) >= 2 and parts[0].isdigit():
            return parts[1]
        return parts[0]
    return parts[0]


def _largest_fragment(mol: Chem.Mol) -> tuple[Chem.Mol, int]:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) <= 1:
        return mol, 0

    def score(frag: Chem.Mol) -> tuple[int, int]:
        heavy = sum(1 for atom in frag.GetAtoms() if atom.GetAtomicNum() > 1)
        return heavy, frag.GetNumAtoms()

    return max(frags, key=score), len(frags) - 1


def _encode_record(record: tuple[str, str]) -> tuple[dict[str, object] | None, str]:
    source, smiles = record
    max_smiles_len = int(WORKER_CFG["max_smiles_len"])
    min_heavy_atoms = int(WORKER_CFG["min_heavy_atoms"])
    max_heavy_atoms = int(WORKER_CFG["max_heavy_atoms"])
    fragment_policy = str(WORKER_CFG["fragment_policy"])

    if not smiles:
        return None, "empty"
    if max_smiles_len > 0 and len(smiles) > max_smiles_len:
        return None, "over_smiles_len"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid_smiles"

    dropped_fragments = 0
    n_fragments = len(Chem.GetMolFrags(mol))
    if n_fragments > 1:
        if fragment_policy == "drop":
            return None, "disconnected"
        if fragment_policy == "largest":
            mol, dropped_fragments = _largest_fragment(mol)
        elif fragment_policy != "keep":
            return None, "bad_fragment_policy"

    heavy = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    if min_heavy_atoms > 0 and heavy < min_heavy_atoms:
        return None, "too_few_heavy_atoms"
    if max_heavy_atoms > 0 and heavy > max_heavy_atoms:
        return None, "too_many_heavy_atoms"

    try:
        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        stamp = REP.encode(canon)
    except Exception:
        return None, "stamp_encode_fail"

    return (
        {
            "smiles": canon,
            "source": source,
            "stamp": stamp,
            "n_fragments": n_fragments,
            "dropped_fragments": dropped_fragments,
            "heavy_atoms": heavy,
        },
        "ok",
    )


def _byte_range_lines(path: Path, shard_index: int, num_shards: int) -> Iterator[bytes]:
    size = path.stat().st_size
    start = size * shard_index // num_shards
    end = size * (shard_index + 1) // num_shards
    with path.open("rb") as f:
        if start > 0:
            f.seek(start - 1)
            f.readline()
        else:
            f.seek(0)
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            yield line


def _pubchem_zip_lines(path: Path, shard_index: int, num_shards: int) -> Iterator[bytes]:
    with zipfile.ZipFile(path) as zf:
        names = [info.filename for info in zf.infolist() if info.filename.endswith(".smi")]
        if len(names) != 1:
            raise ValueError(f"Expected one PubChem .smi member in {path}, found {len(names)}")
        with zf.open(names[0]) as f:
            for row_idx, line in enumerate(f):
                if row_idx % num_shards == shard_index:
                    yield line


def _balanced_zinc_members(path: Path, shard_index: int, num_shards: int) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        infos = [
            info
            for info in zf.infolist()
            if info.filename.startswith("ZINC/") and info.filename.endswith(".smi") and info.file_size > 0
        ]
    loads = [0 for _ in range(num_shards)]
    bins: list[list[str]] = [[] for _ in range(num_shards)]
    for info in sorted(infos, key=lambda x: x.file_size, reverse=True):
        idx = min(range(num_shards), key=lambda i: loads[i])
        bins[idx].append(info.filename)
        loads[idx] += int(info.file_size)
    return sorted(bins[shard_index])


def _zinc_zip_lines(path: Path, members: list[str]) -> Iterator[bytes]:
    with zipfile.ZipFile(path) as zf:
        for name in members:
            with zf.open(name) as f:
                yield from f


def _iter_records(args: argparse.Namespace) -> Iterator[tuple[str, str]]:
    if args.source in {"all", "pubchem"}:
        pubchem_smi = Path(args.pubchem_smi) if args.pubchem_smi else None
        if pubchem_smi and pubchem_smi.exists():
            pubchem_lines = _byte_range_lines(pubchem_smi, args.shard_index, args.num_shards)
        elif args.pubchem_zip:
            pubchem_lines = _pubchem_zip_lines(Path(args.pubchem_zip), args.shard_index, args.num_shards)
        else:
            pubchem_lines = iter(())
        for line in pubchem_lines:
            smiles = _parse_smiles_from_line(line, "pubchem")
            if smiles:
                yield "pubchem", smiles

    if args.source in {"all", "zinc"} and args.zinc_zip:
        members = _balanced_zinc_members(Path(args.zinc_zip), args.shard_index, args.num_shards)
        for line in _zinc_zip_lines(Path(args.zinc_zip), members):
            smiles = _parse_smiles_from_line(line, "zinc")
            if smiles:
                yield "zinc", smiles


def _schema() -> pa.Schema:
    return pa.schema(
        [
            ("smiles", pa.string()),
            ("source", pa.string()),
            ("stamp", pa.string()),
            ("n_fragments", pa.int16()),
            ("dropped_fragments", pa.int16()),
            ("heavy_atoms", pa.int16()),
        ]
    )


def _write_batch(
    pool,
    batch: list[tuple[str, str]],
    writer: pq.ParquetWriter,
    schema: pa.Schema,
    stats: Counter[str],
    source_counts: Counter[str],
    args: argparse.Namespace,
) -> int:
    chunksize = max(1, len(batch) // max(1, args.n_workers * 4))
    rows: list[dict[str, object]] = []
    for row, status in pool.imap(_encode_record, batch, chunksize=chunksize):
        stats[status] += 1
        if row is not None:
            rows.append(row)
            source_counts[str(row["source"])] += 1
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    return len(batch)


def _log_progress(total: int, stats: Counter[str], source_counts: Counter[str], t0: float, shard_index: int) -> None:
    elapsed = max(1e-9, time.time() - t0)
    ok = stats.get("ok", 0)
    drops = {k: v for k, v in stats.items() if k != "ok"}
    print(
        f"[shard {shard_index:05d}] seen={total:,} ok={ok:,} "
        f"ok_rate={ok/max(1,total)*100:.1f}% rate={total/elapsed:.0f}/s "
        f"sources={dict(source_counts)} drops={drops}",
        flush=True,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pubchem-smi", default="")
    p.add_argument("--pubchem-zip", default="/scratch/yuxuan.ren/data/molformer_original/downloads/pubchem_smiles_canonical.zip")
    p.add_argument("--zinc-zip", default="/scratch/yuxuan.ren/data/molformer_original/downloads/ZINC15_smiles.zip")
    p.add_argument("--source", choices=["all", "pubchem", "zinc"], default="all")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--stamp-vocab", required=True)
    p.add_argument("--n-workers", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=10000)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--max-smiles-len", type=int, default=211)
    p.add_argument("--min-heavy-atoms", type=int, default=1)
    p.add_argument("--max-heavy-atoms", type=int, default=0)
    p.add_argument("--fragment-policy", choices=["largest", "drop", "keep"], default="largest")
    args = p.parse_args()

    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"train_{args.shard_index:05d}.parquet"
    stats_path = out_dir / f"train_{args.shard_index:05d}.stats.json"
    schema = _schema()

    worker_cfg = {
        "max_smiles_len": args.max_smiles_len,
        "min_heavy_atoms": args.min_heavy_atoms,
        "max_heavy_atoms": args.max_heavy_atoms,
        "fragment_policy": args.fragment_policy,
    }
    stats: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    total = 0
    t0 = time.time()

    writer = pq.ParquetWriter(out_path, schema, compression="snappy")
    try:
        from multiprocessing import Pool

        with Pool(
            max(1, args.n_workers),
            initializer=_init_worker,
            initargs=(args.stamp_vocab, json.dumps(worker_cfg)),
        ) as pool:
            batch: list[tuple[str, str]] = []
            for record in _iter_records(args):
                batch.append(record)
                if len(batch) >= args.batch_size:
                    total += _write_batch(pool, batch, writer, schema, stats, source_counts, args)
                    batch = []
                    _log_progress(total, stats, source_counts, t0, args.shard_index)
                    if args.max_rows > 0 and total >= args.max_rows:
                        break
            if batch and not (args.max_rows > 0 and total >= args.max_rows):
                total += _write_batch(pool, batch, writer, schema, stats, source_counts, args)
                _log_progress(total, stats, source_counts, t0, args.shard_index)
    finally:
        writer.close()

    payload = {
        "out": str(out_path),
        "elapsed_sec": time.time() - t0,
        "total_input_records": total,
        "stats": dict(stats),
        "source_counts": dict(source_counts),
        "args": vars(args),
    }
    stats_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
