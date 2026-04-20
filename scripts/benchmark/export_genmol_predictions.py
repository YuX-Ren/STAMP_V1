#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark.eval_ao_genmol_fragments import _build_genmol_prompt_row  # noqa: E402
from stamp.benchmark.lm import load_trained_model  # noqa: E402
from stamp.benchmark.common import canonicalize_smiles  # noqa: E402
from stamp.benchmark.tasks import _encode_completion_prompt_tokens, _sample_ao_infill_sequences  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Export AO-GPT predictions on GenMol original fragment tasks.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--fragments-csv", default="/tmp/genmol_repo/data/fragments.csv")
    p.add_argument("--output", required=True)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    RDLogger.DisableLog("rdApp.*")
    model, vocab, rep, dev = load_trained_model(args.model_dir, device=args.device)
    df = pd.read_csv(args.fragments_csv)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    tasks = ["linker_design", "motif_extension", "scaffold_decoration", "superstructure_generation"]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "task",
                "name",
                "source_smiles",
                "query_smiles",
                "sample_id",
                "prompt_repr",
                "pred_full_repr",
                "pred_smiles",
                "valid",
            ]
        )
        for task in tasks:
            for _, row in df.iterrows():
                built = _build_genmol_prompt_row(rep, row["smiles"], task, row[task])
                if built is None:
                    writer.writerow([task, row["name"], row["smiles"], row[task], -1, "", "", "", 0])
                    continue
                prompt_tokens = _encode_completion_prompt_tokens(rep, built["masked_repr"], is_repr=True)
                seqs = _sample_ao_infill_sequences(
                    model,
                    vocab,
                    prompt_tokens=prompt_tokens,
                    n_samples=int(args.num_samples),
                    temperature=float(args.temperature),
                    top_k=int(args.top_k),
                    device=dev,
                )
                for i, toks in enumerate(seqs):
                    full_repr = rep.detokenize(toks)
                    try:
                        pred_smiles = rep.decode(full_repr)
                    except Exception:
                        pred_smiles = ""
                    valid = int(canonicalize_smiles(pred_smiles, require_single_component=True) is not None) if pred_smiles else 0
                    writer.writerow(
                        [
                            task,
                            row["name"],
                            row["smiles"],
                            row[task],
                            i,
                            built["prompt_repr"],
                            full_repr,
                            pred_smiles,
                            valid,
                        ]
                    )
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
