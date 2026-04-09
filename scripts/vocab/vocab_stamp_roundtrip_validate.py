#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger

from stamp.core.mol_normalize import normalize_mol
from stamp.core.stamp_codec import STAMPEncoder
from stamp.vocab.candidate_generation import generate_candidates
from stamp.vocab.vocab_builder import iter_smiles_from_file

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


WORKER_ENCODER: Optional[STAMPEncoder] = None
WORKER_CORE_HASHES: Optional[set[str]] = None
WORKER_SCHEMA_HASHES: Optional[set[str]] = None


@dataclass
class ItemResult:
    row_index: int
    input_smiles: str
    canonical_input: str
    stamp: str
    recovered_smiles: str
    strict_ok: bool
    achiral_ok: bool
    num_candidates: int
    num_in_vocab: int
    num_fallback: int
    error: str = ""


def _init_worker(core_hashes: List[str], schema_hashes: List[str]) -> None:
    global WORKER_ENCODER, WORKER_CORE_HASHES, WORKER_SCHEMA_HASHES
    RDLogger.DisableLog("rdApp.*")
    WORKER_ENCODER = STAMPEncoder()
    WORKER_CORE_HASHES = set(core_hashes)
    WORKER_SCHEMA_HASHES = set(schema_hashes)


def _canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _achiral(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _process_item(payload: Tuple[int, str]) -> ItemResult:
    global WORKER_ENCODER, WORKER_CORE_HASHES, WORKER_SCHEMA_HASHES
    row_index, smiles = payload

    if WORKER_ENCODER is None or WORKER_CORE_HASHES is None or WORKER_SCHEMA_HASHES is None:
        raise RuntimeError("Worker not initialized.")

    try:
        canonical_input = _canonical(smiles)
        stamp = WORKER_ENCODER.encode_smiles(smiles)
        recovered = WORKER_ENCODER.decode_stamp(stamp, canonical=True)

        strict_ok = canonical_input == recovered
        achiral_ok = _achiral(canonical_input) == _achiral(recovered)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES in dataset: {smiles}")
        normalized = normalize_mol(mol)
        candidates, _ = generate_candidates(normalized, WORKER_ENCODER)

        in_vocab = 0
        for cand in candidates:
            if (
                cand.internal_subgraph_hash in WORKER_CORE_HASHES
                and cand.schema_hash in WORKER_SCHEMA_HASHES
            ):
                in_vocab += 1

        return ItemResult(
            row_index=row_index,
            input_smiles=smiles,
            canonical_input=canonical_input,
            stamp=stamp,
            recovered_smiles=recovered,
            strict_ok=strict_ok,
            achiral_ok=achiral_ok,
            num_candidates=len(candidates),
            num_in_vocab=in_vocab,
            num_fallback=max(0, len(candidates) - in_vocab),
        )
    except Exception as exc:
        return ItemResult(
            row_index=row_index,
            input_smiles=smiles,
            canonical_input="",
            stamp="",
            recovered_smiles="",
            strict_ok=False,
            achiral_ok=False,
            num_candidates=0,
            num_in_vocab=0,
            num_fallback=0,
            error=str(exc),
        )


def _estimate_total(input_path: str, smiles_column: str, max_mols: Optional[int]) -> Optional[int]:
    path = Path(input_path)
    total: Optional[int] = None

    try:
        if path.suffix.lower() in {".smi", ".smiles", ".txt"}:
            with path.open("r", encoding="utf-8") as f:
                total = sum(1 for line in f if line.strip())
        else:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                total = 0
                for row in reader:
                    smi = (row.get(smiles_column) or "").strip()
                    if smi:
                        total += 1
    except Exception:
        total = None

    if total is not None and max_mols is not None:
        total = min(total, max_mols)
    return total


def _load_vocab_hashes(vocab_dir: Path) -> Tuple[List[str], List[str]]:
    core_vocab = json.loads((vocab_dir / "core_vocab.json").read_text(encoding="utf-8"))
    schema_vocab = json.loads((vocab_dir / "schema_vocab.json").read_text(encoding="utf-8"))
    core_hashes = [x["core_hash"] for x in core_vocab]
    schema_hashes = [x["schema_hash"] for x in schema_vocab]
    return core_hashes, schema_hashes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Use a built STAMP vocab to generate STAMP and validate STAMP->SMILES roundtrip."
    )
    p.add_argument("--input", required=True, help="Input dataset (.csv/.smiles).")
    p.add_argument("--vocab-dir", required=True, help="Directory containing core_vocab.json/schema_vocab.json.")
    p.add_argument("--smiles-column", default="smiles", help="SMILES column for CSV input.")
    p.add_argument("--max-mols", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--chunksize", type=int, default=64)
    p.add_argument("--out-prefix", default="results/roundtrip/zinc250k/vocab/latest")
    p.add_argument("--no-tqdm", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")

    vocab_dir = Path(args.vocab_dir)
    core_hashes, schema_hashes = _load_vocab_hashes(vocab_dir)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fail_path = Path(str(out_prefix) + "_failures.tsv")
    summary_path = Path(str(out_prefix) + "_summary.json")
    report_path = Path(str(out_prefix) + "_report.txt")

    total_est = _estimate_total(args.input, args.smiles_column, args.max_mols)
    smiles_iter = iter_smiles_from_file(
        args.input,
        smiles_column=args.smiles_column,
        max_mols=args.max_mols,
    )

    started = datetime.now(timezone.utc)

    total = 0
    strict_passed = 0
    achiral_passed = 0
    errors = 0
    fallback_total = 0
    candidates_total = 0

    with fail_path.open("w", encoding="utf-8", newline="") as f_fail:
        writer = csv.writer(f_fail, delimiter="\t")
        writer.writerow(
            [
                "row_index",
                "status",
                "failure_type",
                "smiles",
                "canonical_input",
                "stamp",
                "recovered_smiles",
                "achiral_input",
                "achiral_recovered",
                "num_candidates",
                "num_in_vocab",
                "num_fallback",
                "error",
            ]
        )

        with ProcessPoolExecutor(
            max_workers=max(1, args.num_workers),
            initializer=_init_worker,
            initargs=(core_hashes, schema_hashes),
        ) as pool:
            mapped = pool.map(
                _process_item,
                enumerate(smiles_iter),
                chunksize=max(1, args.chunksize),
            )

            if not args.no_tqdm and tqdm is not None:
                mapped = tqdm(mapped, total=total_est, desc="STAMP roundtrip")

            for res in mapped:
                total += 1
                candidates_total += res.num_candidates
                fallback_total += res.num_fallback

                if res.error:
                    errors += 1
                    writer.writerow(
                        [
                            res.row_index,
                            "ERROR",
                            "exception",
                            res.input_smiles,
                            "",
                            "",
                            "",
                            "",
                            "",
                            0,
                            0,
                            0,
                            res.error,
                        ]
                    )
                    continue

                if res.strict_ok:
                    strict_passed += 1
                if res.achiral_ok:
                    achiral_passed += 1

                if not res.strict_ok:
                    failure_type = "stereo_only" if res.achiral_ok else "structure_or_valence"
                    writer.writerow(
                        [
                            res.row_index,
                            "FAIL",
                            failure_type,
                            res.input_smiles,
                            res.canonical_input,
                            res.stamp,
                            res.recovered_smiles,
                            _achiral(res.canonical_input),
                            _achiral(res.recovered_smiles),
                            res.num_candidates,
                            res.num_in_vocab,
                            res.num_fallback,
                            "",
                        ]
                    )

    finished = datetime.now(timezone.utc)

    strict_failed = total - strict_passed
    achiral_failed = total - achiral_passed
    vocab_coverage = 1.0
    if candidates_total > 0:
        vocab_coverage = (candidates_total - fallback_total) / candidates_total

    summary = {
        "input": args.input,
        "vocab_dir": str(vocab_dir),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "total": total,
        "strict_passed": strict_passed,
        "strict_failed": strict_failed,
        "strict_pass_rate": strict_passed / total if total else 0.0,
        "achiral_passed": achiral_passed,
        "achiral_failed": achiral_failed,
        "achiral_pass_rate": achiral_passed / total if total else 0.0,
        "errors": errors,
        "candidates_total": candidates_total,
        "fallback_total": fallback_total,
        "vocab_coverage": vocab_coverage,
        "failures_file": str(fail_path),
        "summary_file": str(summary_path),
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "Vocab-based STAMP Roundtrip Report",
        f"input: {args.input}",
        f"vocab_dir: {vocab_dir}",
        f"started_at: {started.isoformat()}",
        f"finished_at: {finished.isoformat()}",
        f"total: {total}",
        f"strict_passed: {strict_passed}",
        f"strict_failed: {strict_failed}",
        f"strict_pass_rate: {summary['strict_pass_rate']:.6f}",
        f"achiral_passed: {achiral_passed}",
        f"achiral_failed: {achiral_failed}",
        f"achiral_pass_rate: {summary['achiral_pass_rate']:.6f}",
        f"errors: {errors}",
        f"candidates_total: {candidates_total}",
        f"fallback_total: {fallback_total}",
        f"vocab_coverage: {vocab_coverage:.6f}",
        f"failures_file: {fail_path}",
        f"summary_file: {summary_path}",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Saved failures: {fail_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
