#!/usr/bin/env python
"""Build a sampled ChEMBL matched-molecular-pair CSV.

This script mines single-cut MMP-style pairs from a SMILES table.  For each
molecule, RDKit MMPA cuts acyclic single bonds; the larger fragment is treated
as the shared context and the smaller fragment as the variable substituent.
Molecules with the same context and different variables form MMP pairs.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMMPA

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
RDLogger.DisableLog("rdApp.*")

from stamp.benchmark.edit_mechanism import write_csv, write_json  # noqa: E402


def _read_smiles(
    path: Path,
    *,
    smiles_col: str,
    max_mols: int,
) -> Iterable[tuple[int, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(fh, dialect=dialect)
        for idx, row in enumerate(reader):
            smiles = (row.get(smiles_col) or "").strip()
            if smiles:
                yield idx, smiles
            if max_mols > 0 and idx + 1 >= max_mols:
                break


def _canonical_fragment(fragment: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(fragment)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _heavy_atoms(fragment: str) -> int:
    mol = Chem.MolFromSmiles(fragment)
    if mol is None:
        return 0
    return int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1))


def _classify_variable_pair(variable_a_heavy: int, variable_b_heavy: int) -> str:
    low = min(variable_a_heavy, variable_b_heavy)
    high = max(variable_a_heavy, variable_b_heavy)
    if high <= 2:
        return "atom_substitution"
    if low <= 1 < high:
        return "scaffold_decoration"
    if abs(variable_a_heavy - variable_b_heavy) <= 2:
        return "substituent_replacement"
    return "motif_extension"


def _single_cut_context_records(
    smiles: str,
    *,
    min_context_heavy_atoms: int,
    max_variable_heavy_atoms: int,
) -> List[Dict[str, Any]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0 or len(Chem.GetMolFrags(mol)) != 1:
        return []
    canonical_parent = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    records: List[Dict[str, Any]] = []
    try:
        fragments = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False)
    except Exception:
        return records
    for core, chains in fragments:
        if core:
            continue
        parts = [part for part in str(chains).split(".") if part]
        if len(parts) != 2:
            continue
        part_a = _canonical_fragment(parts[0])
        part_b = _canonical_fragment(parts[1])
        if part_a is None or part_b is None:
            continue
        heavy_a = _heavy_atoms(part_a)
        heavy_b = _heavy_atoms(part_b)
        if heavy_a >= heavy_b:
            context, variable = part_a, part_b
            context_heavy, variable_heavy = heavy_a, heavy_b
        else:
            context, variable = part_b, part_a
            context_heavy, variable_heavy = heavy_b, heavy_a
        if context_heavy < min_context_heavy_atoms:
            continue
        if variable_heavy <= 0 or variable_heavy > max_variable_heavy_atoms:
            continue
        records.append(
            {
                "smiles": canonical_parent,
                "context": context,
                "variable": variable,
                "context_heavy_atoms": context_heavy,
                "variable_heavy_atoms": variable_heavy,
            }
        )
    return records


def build_pairs(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(int(args.seed))
    by_context: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_context_variable: Dict[str, set[str]] = defaultdict(set)
    n_seen = 0
    n_valid_parent = 0
    n_fragment_records = 0

    for source_idx, smiles in _read_smiles(
        Path(args.input),
        smiles_col=args.smiles_col,
        max_mols=int(args.max_mols),
    ):
        n_seen += 1
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0 or len(Chem.GetMolFrags(mol)) != 1:
            continue
        heavy_atoms = int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1))
        if heavy_atoms < int(args.min_parent_heavy_atoms):
            continue
        if int(args.max_parent_heavy_atoms) > 0 and heavy_atoms > int(args.max_parent_heavy_atoms):
            continue
        n_valid_parent += 1
        records = _single_cut_context_records(
            smiles,
            min_context_heavy_atoms=int(args.min_context_heavy_atoms),
            max_variable_heavy_atoms=int(args.max_variable_heavy_atoms),
        )
        for record in records:
            n_fragment_records += 1
            context = str(record["context"])
            variable = str(record["variable"])
            if variable in seen_context_variable[context]:
                continue
            if len(by_context[context]) >= int(args.max_records_per_context):
                continue
            record["source_index"] = source_idx
            by_context[context].append(record)
            seen_context_variable[context].add(variable)

    contexts = [context for context, records in by_context.items() if len(records) >= 2]
    rng.shuffle(contexts)
    pairs: List[Dict[str, Any]] = []
    for context in contexts:
        records = list(by_context[context])
        rng.shuffle(records)
        n_for_context = 0
        for rec_a, rec_b in combinations(records, 2):
            if rec_a["smiles"] == rec_b["smiles"]:
                continue
            if rec_a["variable"] == rec_b["variable"]:
                continue
            edit_class = _classify_variable_pair(
                int(rec_a["variable_heavy_atoms"]),
                int(rec_b["variable_heavy_atoms"]),
            )
            pairs.append(
                {
                    "pair_id": f"chembl_mmp_{len(pairs)}",
                    "edit_class": edit_class,
                    "smiles_a": rec_a["smiles"],
                    "smiles_b": rec_b["smiles"],
                    "context": context,
                    "variable_a": rec_a["variable"],
                    "variable_b": rec_b["variable"],
                    "context_heavy_atoms": rec_a["context_heavy_atoms"],
                    "variable_a_heavy_atoms": rec_a["variable_heavy_atoms"],
                    "variable_b_heavy_atoms": rec_b["variable_heavy_atoms"],
                    "source_index_a": rec_a["source_index"],
                    "source_index_b": rec_b["source_index"],
                }
            )
            n_for_context += 1
            if n_for_context >= int(args.max_pairs_per_context):
                break
            if int(args.max_pairs) > 0 and len(pairs) >= int(args.max_pairs):
                break
        if int(args.max_pairs) > 0 and len(pairs) >= int(args.max_pairs):
            break

    metadata = {
        "input": str(Path(args.input)),
        "n_input_rows_seen": n_seen,
        "n_valid_parent_molecules": n_valid_parent,
        "n_fragment_records": n_fragment_records,
        "n_contexts_total": len(by_context),
        "n_contexts_with_pairs": len(contexts),
        "n_pairs": len(pairs),
        "max_mols": int(args.max_mols),
        "max_pairs": int(args.max_pairs),
        "max_pairs_per_context": int(args.max_pairs_per_context),
        "max_records_per_context": int(args.max_records_per_context),
        "min_parent_heavy_atoms": int(args.min_parent_heavy_atoms),
        "max_parent_heavy_atoms": int(args.max_parent_heavy_atoms),
        "min_context_heavy_atoms": int(args.min_context_heavy_atoms),
        "max_variable_heavy_atoms": int(args.max_variable_heavy_atoms),
        "seed": int(args.seed),
    }
    return pairs, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sampled ChEMBL single-cut MMP pairs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--max-mols", type=int, default=50000)
    parser.add_argument("--max-pairs", type=int, default=10000)
    parser.add_argument("--max-pairs-per-context", type=int, default=20)
    parser.add_argument("--max-records-per-context", type=int, default=64)
    parser.add_argument("--min-parent-heavy-atoms", type=int, default=6)
    parser.add_argument("--max-parent-heavy-atoms", type=int, default=80)
    parser.add_argument("--min-context-heavy-atoms", type=int, default=5)
    parser.add_argument("--max-variable-heavy-atoms", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs, metadata = build_pairs(args)
    write_csv(output, pairs)
    write_json(output.with_suffix(output.suffix + ".meta.json"), metadata)
    print(f"[chembl-mmp] pairs={len(pairs)} contexts={metadata['n_contexts_with_pairs']}")
    print(f"[chembl-mmp] wrote {output}")


if __name__ == "__main__":
    main()
