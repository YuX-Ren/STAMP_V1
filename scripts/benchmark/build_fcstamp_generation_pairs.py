#!/usr/bin/env python
"""Extract source-to-generated molecule pairs from FC-STAMP eval JSON files."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from rdkit import Chem, RDLogger

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
RDLogger.DisableLog("rdApp.*")

from stamp.benchmark.edit_mechanism import write_csv, write_json  # noqa: E402


def _canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0 or len(Chem.GetMolFrags(mol)) != 1:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _expand_paths(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in sorted(matches))
        else:
            paths.append(Path(pattern))
    return paths


def extract_pairs(paths: List[Path], max_examples_per_case: int) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n_cases = 0
    n_examples_seen = 0
    n_invalid = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = payload.get("seed", "")
        for task, task_payload in payload.get("tasks", {}).items():
            for case_idx, case in enumerate(task_payload.get("cases", [])):
                n_cases += 1
                source = _canonical(str(case.get("source_smiles", "")))
                if source is None:
                    n_invalid += 1
                    continue
                examples = list(case.get("examples", []))
                if max_examples_per_case > 0:
                    examples = examples[:max_examples_per_case]
                for example_idx, example in enumerate(examples):
                    n_examples_seen += 1
                    generated = _canonical(str(example))
                    if generated is None or generated == source:
                        if generated is None:
                            n_invalid += 1
                        continue
                    rows.append(
                        {
                            "pair_id": (
                                f"{task}:seed{seed}:case{case_idx}:example{example_idx}"
                            ),
                            "edit_class": str(task),
                            "smiles_a": source,
                            "smiles_b": generated,
                            "case_name": str(case.get("name", "")),
                            "seed": seed,
                            "eval_json": str(path),
                        }
                    )
    metadata = {
        "n_eval_jsons": len(paths),
        "n_cases": n_cases,
        "n_examples_seen": n_examples_seen,
        "n_invalid_or_disconnected": n_invalid,
        "n_pairs": len(rows),
        "max_examples_per_case": max_examples_per_case,
        "eval_jsons": [str(path) for path in paths],
    }
    return rows, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FC-STAMP source/generated-example pair CSV."
    )
    parser.add_argument("--eval-json", nargs="+", required=True, help="Eval JSON path(s) or globs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples-per-case", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _expand_paths(args.eval_json)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing eval JSON files: {missing}")
    rows, metadata = extract_pairs(paths, int(args.max_examples_per_case))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, rows)
    write_json(output.with_suffix(output.suffix + ".meta.json"), metadata)
    print(f"[fcstamp-pairs] pairs={len(rows)} eval_jsons={len(paths)}")
    print(f"[fcstamp-pairs] wrote {output}")


if __name__ == "__main__":
    main()
