#!/usr/bin/env python3
"""Add discrete QED/SA condition-token columns to STAMP parquet shards."""
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
from rdkit import Chem, RDLogger
from rdkit.Chem import QED

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.common import iterable_with_progress  # noqa: E402
from stamp.benchmark.metrics import _load_sa_scorer  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def _resolve_path_specs(specs: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        expanded = sorted(glob.glob(str(spec)))
        if not expanded and os.path.exists(str(spec)):
            expanded = [str(spec)]
        for path in expanded:
            resolved = str(Path(path).resolve())
            if resolved not in seen:
                out.append(resolved)
                seen.add(resolved)
    return out


def _qed_token(qed: float) -> str:
    if qed < 0.4:
        return "[C_QED_LOW]"
    if qed < 0.6:
        return "[C_QED_MID]"
    if qed < 0.8:
        return "[C_QED_HIGH]"
    return "[C_QED_VHIGH]"


def _sa_token(sa_value: float | None) -> str:
    if sa_value is None or not math.isfinite(float(sa_value)):
        return "[C_SA_UNK]"
    if float(sa_value) <= 3.0:
        return "[C_SA_LOW]"
    if float(sa_value) <= 4.0:
        return "[C_SA_MID]"
    return "[C_SA_HIGH]"


def _compute_one(smiles: Any, *, condition_mode: str, sa_scorer: Any) -> tuple[float, float, str, str, str]:
    mode = str(condition_mode or "qed_sa").strip().lower()
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return float("nan"), float("nan"), "", "", ""
    try:
        qed_value = float(QED.qed(mol))
    except Exception:
        return float("nan"), float("nan"), "", "", ""
    qed_tok = _qed_token(qed_value)
    if mode == "qed":
        return qed_value, float("nan"), qed_tok, "", qed_tok
    if mode != "qed_sa":
        raise ValueError(f"Unsupported condition_mode: {condition_mode!r}")
    sa_value: float | None = None
    if sa_scorer is not None:
        try:
            sa_value = float(sa_scorer.calculateScore(mol))
        except Exception:
            sa_value = None
    sa_tok = _sa_token(sa_value)
    return qed_value, (float(sa_value) if sa_value is not None else float("nan")), qed_tok, sa_tok, f"{qed_tok} {sa_tok}"


def _drop_existing(table: pa.Table, names: list[str]) -> pa.Table:
    existing = [name for name in names if name in table.column_names]
    return table.drop(existing) if existing else table


def _write_one_file(
    *,
    input_path: str,
    output_path: Path,
    smiles_column: str,
    condition_mode: str,
    compression: str,
    overwrite: bool,
    max_rows: int,
    show_progress: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        return {"input": input_path, "output": str(output_path), "skipped": True}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(input_path)
    if smiles_column not in pf.schema.names:
        raise KeyError(f"{input_path} missing smiles column {smiles_column!r}; columns={pf.schema.names}")

    sa_scorer = _load_sa_scorer() if str(condition_mode).strip().lower() == "qed_sa" else None
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    valid_conditions = 0
    t0 = time.time()
    try:
        rg_iter = iterable_with_progress(
            range(pf.num_row_groups),
            show_progress=show_progress,
            total=pf.num_row_groups,
            desc=f"condition {Path(input_path).name}",
        )
        remaining = int(max_rows) if int(max_rows) > 0 else None
        for rg in rg_iter:
            table = pf.read_row_group(rg)
            if remaining is not None and remaining <= 0:
                break
            if remaining is not None and table.num_rows > remaining:
                table = table.slice(0, remaining)
            smiles_values = table.column(smiles_column).to_pylist()
            qed_values: list[float] = []
            sa_values: list[float] = []
            qed_tokens: list[str] = []
            sa_tokens: list[str] = []
            condition_tokens: list[str] = []
            for smi in smiles_values:
                qed_value, sa_value, qed_tok, sa_tok, cond = _compute_one(
                    smi,
                    condition_mode=condition_mode,
                    sa_scorer=sa_scorer,
                )
                qed_values.append(qed_value)
                sa_values.append(sa_value)
                qed_tokens.append(qed_tok)
                sa_tokens.append(sa_tok)
                condition_tokens.append(cond)
                if cond:
                    valid_conditions += 1
            table = _drop_existing(table, ["qed", "sa", "condition_qed", "condition_sa", "condition_tokens"])
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
    parser.add_argument("--parquet", required=True, help="Input parquet path or glob.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--condition-mode", default="qed_sa", choices=["qed", "qed_sa"])
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    args = parser.parse_args()

    paths = _resolve_path_specs([args.parquet])
    if int(args.max_files) > 0:
        paths = paths[: int(args.max_files)]
    if not paths:
        raise FileNotFoundError(args.parquet)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for path in paths:
        result = _write_one_file(
            input_path=path,
            output_path=out_dir / Path(path).name,
            smiles_column=args.smiles_column,
            condition_mode=args.condition_mode,
            compression=str(args.compression),
            overwrite=bool(args.overwrite),
            max_rows=int(args.max_rows_per_file),
            show_progress=not bool(args.no_tqdm),
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    manifest = {
        "input": args.parquet,
        "output_dir": str(out_dir),
        "condition_mode": str(args.condition_mode),
        "smiles_column": str(args.smiles_column),
        "files": results,
    }
    (out_dir / "condition_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "n_files": len(results)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
