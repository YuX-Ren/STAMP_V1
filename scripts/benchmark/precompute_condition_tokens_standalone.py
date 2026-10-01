#!/usr/bin/env python3
"""Standalone parquet QED/SA condition precompute.

This script does not depend on the STAMP repository. It reads existing parquet
shards with a SMILES column and writes copies with added condition columns:

  qed, sa, condition_qed, condition_sa, condition_tokens
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import QED

RDLogger.DisableLog("rdApp.*")


def _load_sa_scorer():
    sa_dir = Path(RDConfig.RDContribDir) / "SA_Score"
    if str(sa_dir) not in sys.path:
        sys.path.insert(0, str(sa_dir))
    import sascorer  # type: ignore

    return sascorer


def _resolve_paths(spec: str) -> list[str]:
    paths = sorted(glob.glob(str(spec)))
    if not paths and Path(spec).exists():
        paths = [str(spec)]
    return [str(Path(p).resolve()) for p in paths]


def _qed_token(value: float) -> str:
    if value < 0.4:
        return "[C_QED_LOW]"
    if value < 0.6:
        return "[C_QED_MID]"
    if value < 0.8:
        return "[C_QED_HIGH]"
    return "[C_QED_VHIGH]"


def _sa_token(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "[C_SA_UNK]"
    if float(value) <= 3.0:
        return "[C_SA_LOW]"
    if float(value) <= 4.0:
        return "[C_SA_MID]"
    return "[C_SA_HIGH]"


def _compute_condition(smiles: Any, sa_scorer) -> tuple[float, float, str, str, str]:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return float("nan"), float("nan"), "", "", ""
    try:
        qed_value = float(QED.qed(mol))
    except Exception:
        return float("nan"), float("nan"), "", "", ""
    sa_value: float | None = None
    try:
        sa_value = float(sa_scorer.calculateScore(mol))
    except Exception:
        sa_value = None
    qed_tok = _qed_token(qed_value)
    sa_tok = _sa_token(sa_value)
    return qed_value, (float(sa_value) if sa_value is not None else float("nan")), qed_tok, sa_tok, f"{qed_tok} {sa_tok}"


def _drop_existing(table: pa.Table) -> pa.Table:
    cols = ["qed", "sa", "condition_qed", "condition_sa", "condition_tokens"]
    existing = [c for c in cols if c in table.column_names]
    return table.drop(existing) if existing else table


def _process_file(
    *,
    input_path: str,
    output_path: Path,
    smiles_column: str,
    compression: str,
    overwrite: bool,
    max_rows: int,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        return {"input": input_path, "output": str(output_path), "skipped": True}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(input_path)
    if smiles_column not in pf.schema.names:
        raise KeyError(f"{input_path} missing {smiles_column!r}; columns={pf.schema.names}")

    sa_scorer = _load_sa_scorer()
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    valid_conditions = 0
    t0 = time.time()
    remaining = int(max_rows) if int(max_rows) > 0 else None
    try:
        for rg in range(pf.num_row_groups):
            if remaining is not None and remaining <= 0:
                break
            table = pf.read_row_group(rg)
            if remaining is not None and table.num_rows > remaining:
                table = table.slice(0, remaining)
            smiles = table.column(smiles_column).to_pylist()
            qed_values: list[float] = []
            sa_values: list[float] = []
            qed_tokens: list[str] = []
            sa_tokens: list[str] = []
            condition_tokens: list[str] = []
            for smi in smiles:
                qed, sa, qtok, stok, cond = _compute_condition(smi, sa_scorer)
                qed_values.append(qed)
                sa_values.append(sa)
                qed_tokens.append(qtok)
                sa_tokens.append(stok)
                condition_tokens.append(cond)
                if cond:
                    valid_conditions += 1
            table = _drop_existing(table)
            table = table.append_column("qed", pa.array(qed_values, type=pa.float32()))
            table = table.append_column("sa", pa.array(sa_values, type=pa.float32()))
            table = table.append_column("condition_qed", pa.array(qed_tokens, type=pa.string()))
            table = table.append_column("condition_sa", pa.array(sa_tokens, type=pa.string()))
            table = table.append_column("condition_tokens", pa.array(condition_tokens, type=pa.string()))
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema, compression=compression)
            writer.write_table(table)
            total_rows += table.num_rows
            if remaining is not None:
                remaining -= table.num_rows
        if writer is not None:
            writer.close()
            writer = None
        os.replace(tmp_path, output_path)
    finally:
        if writer is not None:
            writer.close()
        if tmp_path.exists():
            tmp_path.unlink()
    return {
        "input": input_path,
        "output": str(output_path),
        "rows": total_rows,
        "valid_conditions": valid_conditions,
        "seconds": time.time() - t0,
        "skipped": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = _resolve_paths(args.parquet)
    if int(args.max_files) > 0:
        paths = paths[: int(args.max_files)]
    if not paths:
        raise FileNotFoundError(args.parquet)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, path in enumerate(paths, start=1):
        print(f"[precompute] {i}/{len(paths)} {path}", flush=True)
        result = _process_file(
            input_path=path,
            output_path=out_dir / Path(path).name,
            smiles_column=str(args.smiles_column),
            compression=str(args.compression),
            overwrite=bool(args.overwrite),
            max_rows=int(args.max_rows_per_file),
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    manifest = {
        "input": args.parquet,
        "output_dir": str(out_dir),
        "smiles_column": str(args.smiles_column),
        "files": results,
    }
    (out_dir / "condition_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
