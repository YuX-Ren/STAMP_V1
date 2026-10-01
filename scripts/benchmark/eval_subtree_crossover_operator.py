#!/usr/bin/env python
"""Evaluate STAMP subtree crossover against simple token/string controls."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from stamp.benchmark.edit_mechanism import (  # noqa: E402
    build_stamp_tree,
    load_smiles_rows,
    sample_operator_trials,
    summarize_operator_rows,
    write_csv,
    write_json,
)
from stamp.core.stamp_codec import STAMPEncoder  # noqa: E402


def _build_stamp_encoder(stamp_vocab: str | None, force_ring_aware: bool) -> STAMPEncoder:
    if not stamp_vocab:
        return STAMPEncoder()
    from stamp.vocab.principal_subgraph import PrincipalSubgraphTokenizer

    tokenizer = PrincipalSubgraphTokenizer(stamp_vocab)
    if force_ring_aware:
        tokenizer.allow_ring_single_edges = True
        tokenizer.blessed_rings = set(tokenizer.vocab.keys())
    return STAMPEncoder(motif_tokenizer=tokenizer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Operator-level test for the claim that STAMP subtree contiguity "
            "enables valid local crossover."
        )
    )
    parser.add_argument("--input", required=True, help="SMILES file or CSV/TSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--max-mols", type=int, default=5000)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stamp-vocab", default=None)
    parser.add_argument("--stamp-force-ring-aware-encode", action="store_true")
    parser.add_argument("--max-span-len", type=int, default=16)
    parser.add_argument("--size-delta", type=int, default=4)
    parser.add_argument(
        "--operators",
        default="stamp-subtree,stamp-random-span,smiles-substring",
        help=(
            "Comma-separated operators. Supported: stamp-subtree, "
            "stamp-random-span, safe-fragment-splice, smiles-substring."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = _build_stamp_encoder(args.stamp_vocab, args.stamp_force_ring_aware_encode)
    smiles_values = load_smiles_rows(
        Path(args.input),
        smiles_col=args.smiles_col,
        max_mols=int(args.max_mols),
    )

    trees = []
    errors: List[dict] = []
    for idx, smiles in enumerate(smiles_values):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0 or len(Chem.GetMolFrags(mol)) != 1:
            errors.append({"row": idx, "smiles": smiles, "error": "invalid_or_disconnected"})
            continue
        try:
            trees.append(build_stamp_tree(smiles, encoder))
        except Exception as exc:
            errors.append({"row": idx, "smiles": smiles, "error": str(exc)})

    operators = [op.strip() for op in args.operators.split(",") if op.strip()]
    rows = sample_operator_trials(
        trees,
        encoder=encoder,
        n_trials=int(args.n_trials),
        seed=int(args.seed),
        max_span_len=int(args.max_span_len) if int(args.max_span_len) > 0 else None,
        size_delta=int(args.size_delta),
        operators=operators,
    )
    summary = summarize_operator_rows(rows)

    write_csv(output_dir / "operator_trials.csv", rows)
    write_csv(output_dir / "operator_summary.csv", summary)
    write_csv(output_dir / "input_errors.csv", errors)
    write_json(
        output_dir / "operator_config.json",
        {
            "input": str(Path(args.input)),
            "n_input_smiles": len(smiles_values),
            "n_parsed_molecules": len(trees),
            "n_input_errors": len(errors),
            "n_trials_per_operator": int(args.n_trials),
            "operators": operators,
            "stamp_vocab": args.stamp_vocab,
            "max_span_len": int(args.max_span_len),
            "size_delta": int(args.size_delta),
            "seed": int(args.seed),
        },
    )

    print(f"[subtree-operator] parsed_molecules={len(trees)} errors={len(errors)}")
    print(f"[subtree-operator] wrote {output_dir / 'operator_trials.csv'}")
    print(f"[subtree-operator] wrote {output_dir / 'operator_summary.csv'}")


if __name__ == "__main__":
    main()
