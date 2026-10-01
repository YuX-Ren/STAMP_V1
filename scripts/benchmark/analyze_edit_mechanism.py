#!/usr/bin/env python
"""Run STAMP edit decomposition and cross-language edit-locality analysis."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from stamp.benchmark.edit_mechanism import (  # noqa: E402
    analyze_cross_language_pair,
    build_representations,
    decompose_stamp_pair,
    load_pair_rows,
    summarize_numeric_rows,
    write_csv,
    write_json,
)
from stamp.core.stamp_codec import STAMPEncoder  # noqa: E402


DECOMPOSITION_METRICS = (
    "delta_motif",
    "delta_joint",
    "delta_branch",
    "motif_insertions",
    "motif_deletions",
    "motif_substitutions",
    "joint_signature_changes",
    "parent_joint_changes",
    "branch_parent_changes",
    "branch_degree_changes",
    "branch_token_changes",
    "inserted_subtree_roots",
    "deleted_subtree_roots",
    "stamp_edit_script_length",
    "stamp_changed_token_fraction",
    "stamp_normalized_token_edit_distance",
    "stamp_changed_span_count",
    "stamp_disconnected_edit_regions",
    "stamp_largest_changed_span",
    "stamp_minimal_contiguous_edit_span",
    "stamp_one_local_span",
    "stamp_subtree_contiguous_edit",
    "stamp_executable_as_one_typed_patch",
    "stamp_transferable_to_compatible_scaffold",
    "stamp_source_patch_span",
    "stamp_target_patch_span",
    "stamp_executable_patch_span",
)

CROSS_LANGUAGE_METRICS = (
    "edit_script_length",
    "changed_token_fraction",
    "normalized_token_edit_distance",
    "changed_span_count",
    "disconnected_edit_regions",
    "largest_changed_span",
    "minimal_contiguous_edit_span",
    "one_local_span",
    "subtree_contiguous_edit",
)


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
            "Mechanistic evidence tables for STAMP: motif/joint/branch edit "
            "decomposition plus cross-language token locality."
        )
    )
    parser.add_argument("--pairs", required=True, help="CSV/TSV with paired molecules.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smiles-a-col", default="smiles_a")
    parser.add_argument("--smiles-b-col", default="smiles_b")
    parser.add_argument("--edit-class-col", default="edit_class")
    parser.add_argument("--pair-id-col", default="pair_id")
    parser.add_argument("--max-pairs", type=int, default=0, help="0 = all pairs.")
    parser.add_argument("--stamp-vocab", default=None, help="Optional PS-VAE/STAMP motif vocab.")
    parser.add_argument(
        "--stamp-force-ring-aware-encode",
        action="store_true",
        help="Use ring-aware PS-VAE encode mode when --stamp-vocab is supplied.",
    )
    parser.add_argument(
        "--representations",
        default="smiles,stamp,selfies,safe,fragsmiles,group-selfies",
        help="Comma-separated complete molecular languages for locality comparison.",
    )
    parser.add_argument("--skip-stamp-decomposition", action="store_true")
    parser.add_argument("--skip-cross-language", action="store_true")
    parser.add_argument(
        "--strict-representations",
        action="store_true",
        help="Fail if an optional representation dependency is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pair_rows(
        Path(args.pairs),
        smiles_a_col=args.smiles_a_col,
        smiles_b_col=args.smiles_b_col,
        edit_class_col=args.edit_class_col,
        pair_id_col=args.pair_id_col,
        max_pairs=int(args.max_pairs),
    )
    encoder = _build_stamp_encoder(args.stamp_vocab, args.stamp_force_ring_aware_encode)

    all_errors: List[Dict[str, Any]] = []
    decomposition_rows: List[Dict[str, Any]] = []
    if not args.skip_stamp_decomposition:
        for pair in pairs:
            try:
                decomposition_rows.append(
                    decompose_stamp_pair(
                        pair["smiles_a"],
                        pair["smiles_b"],
                        encoder=encoder,
                        edit_class=pair["edit_class"],
                        pair_id=pair["pair_id"],
                    )
                )
            except Exception as exc:
                all_errors.append(
                    {
                        "stage": "stamp_decomposition",
                        "pair_id": pair["pair_id"],
                        "edit_class": pair["edit_class"],
                        "smiles_a": pair["smiles_a"],
                        "smiles_b": pair["smiles_b"],
                        "error": str(exc),
                    }
                )

        write_csv(output_dir / "stamp_edit_decomposition_pairs.csv", decomposition_rows)
        decomposition_summary = summarize_numeric_rows(
            decomposition_rows,
            group_key="edit_class",
            metric_keys=DECOMPOSITION_METRICS,
        )
        write_csv(output_dir / "stamp_edit_decomposition_by_edit_class.csv", decomposition_summary)

    cross_rows: List[Dict[str, Any]] = []
    if not args.skip_cross_language:
        rep_names = [name.strip() for name in args.representations.split(",") if name.strip()]
        representations, rep_errors = build_representations(
            rep_names,
            stamp_vocab=args.stamp_vocab,
            stamp_force_ring_aware_encode=bool(args.stamp_force_ring_aware_encode),
            strict=bool(args.strict_representations),
        )
        all_errors.extend({"stage": "build_representation", **err} for err in rep_errors)
        for pair in pairs:
            for representation in representations:
                try:
                    cross_rows.append(
                        analyze_cross_language_pair(
                            pair["smiles_a"],
                            pair["smiles_b"],
                            representation=representation,
                            pair_id=pair["pair_id"],
                            edit_class=pair["edit_class"],
                            stamp_encoder=encoder,
                        )
                    )
                except Exception as exc:
                    all_errors.append(
                        {
                            "stage": "cross_language",
                            "representation": representation.name,
                            "pair_id": pair["pair_id"],
                            "edit_class": pair["edit_class"],
                            "smiles_a": pair["smiles_a"],
                            "smiles_b": pair["smiles_b"],
                            "error": str(exc),
                        }
                    )
        write_csv(output_dir / "cross_language_edit_locality_pairs.csv", cross_rows)
        cross_summary = summarize_numeric_rows(
            cross_rows,
            group_key="representation",
            metric_keys=CROSS_LANGUAGE_METRICS,
        )
        write_csv(output_dir / "cross_language_edit_locality_by_representation.csv", cross_summary)

    write_csv(output_dir / "errors.csv", all_errors)
    write_json(
        output_dir / "analysis_config.json",
        {
            "pairs": str(Path(args.pairs)),
            "n_input_pairs": len(pairs),
            "n_decomposition_rows": len(decomposition_rows),
            "n_cross_language_rows": len(cross_rows),
            "n_errors": len(all_errors),
            "stamp_vocab": args.stamp_vocab,
            "representations": args.representations,
        },
    )

    print(f"[edit-mechanism] pairs={len(pairs)}")
    if not args.skip_stamp_decomposition:
        print(f"[edit-mechanism] wrote {output_dir / 'stamp_edit_decomposition_pairs.csv'}")
        print(f"[edit-mechanism] wrote {output_dir / 'stamp_edit_decomposition_by_edit_class.csv'}")
    if not args.skip_cross_language:
        print(f"[edit-mechanism] wrote {output_dir / 'cross_language_edit_locality_pairs.csv'}")
        print(f"[edit-mechanism] wrote {output_dir / 'cross_language_edit_locality_by_representation.csv'}")
    if all_errors:
        print(f"[edit-mechanism] warnings/errors={len(all_errors)}; see {output_dir / 'errors.csv'}")


if __name__ == "__main__":
    main()
