#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.data import load_smiles_list  # noqa: E402
from stamp.benchmark.denovo import run_denovo_sampling_eval  # noqa: E402
from stamp.benchmark.representations import build_representation  # noqa: E402
from stamp.vocab.branch_paths import resolve_branch_output_dir  # noqa: E402


def _resolve_length_histogram(
    model_dir: str | Path,
    train_smiles: list[str],
) -> dict[int, int]:
    model_path = Path(model_dir)
    meta_path = model_path / "train_meta.json"
    if not meta_path.exists():
        payload = {}
    else:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    hist_payload = payload.get("train_token_length_histogram") or {}
    hist = {int(k): int(v) for k, v in hist_payload.items() if int(v) > 0}
    if hist:
        return hist
    stamp_vocab_path = None
    vocab_candidate = model_path / "stamp_vocab.txt"
    if vocab_candidate.exists():
        stamp_vocab_path = str(vocab_candidate)
    rep = build_representation("stamp", stamp_vocab=stamp_vocab_path)
    inferred: dict[int, int] = {}
    for s in train_smiles:
        try:
            length = len(rep.tokenize(rep.encode(s)))
        except Exception:
            continue
        inferred[length] = inferred.get(length, 0) + 1
    return inferred


def _sample_lengths_from_histogram(hist: dict[int, int], n_samples: int, max_cap: int) -> list[int]:
    items = sorted((int(k), int(v)) for k, v in hist.items() if int(v) > 0)
    if not items:
        raise ValueError("Empty length histogram.")
    lengths = [min(max_cap, max(1, k)) for k, _ in items]
    weights = [v for _, v in items]
    rng = random.Random(42)
    return rng.choices(lengths, weights=weights, k=n_samples)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained STAMP diffusion LM without optional benchmark deps.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--train", required=True, help="Train SMILES file for novelty.")
    p.add_argument("--reference", required=True, help="Reference/test SMILES file.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-smiles-column", default="smiles")
    p.add_argument("--reference-smiles-column", default="smiles")
    p.add_argument("--n-samples", type=int, default=10_000)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--diffusion-steps", type=int, default=None)
    p.add_argument("--diffusion-tokens-per-step", type=int, default=None)
    p.add_argument("--diffusion-block-size", type=int, default=None)
    p.add_argument("--diffusion-remasking", default=None, choices=["low_confidence", "random"])
    p.add_argument("--diffusion-stochastic-transfer", action="store_true")
    p.add_argument("--decode-batch-size", type=int, default=256)
    p.add_argument("--reference-max-samples", type=int, default=None)
    p.add_argument("--assume-train-canonical", action="store_true")
    p.add_argument("--eval-workers", type=int, default=1)
    p.add_argument("--eval-chunksize", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-tqdm", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")

    train_smiles = load_smiles_list(
        args.train,
        smiles_column=args.train_smiles_column,
        canonicalize=True,
        deduplicate=False,
        show_progress=not args.no_tqdm,
    )
    reference_smiles = load_smiles_list(
        args.reference,
        smiles_column=args.reference_smiles_column,
        canonicalize=True,
        deduplicate=False,
        show_progress=not args.no_tqdm,
    )

    sample_lengths = None
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        meta_path = Path(args.model_dir) / "train_meta.json"
        cfg_cap = 255
        if meta_path.exists():
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            cfg = payload.get("cfg") or {}
            cfg_cap = max(1, int(cfg.get("max_seq_len", 256)) - 1)
        hist = _resolve_length_histogram(args.model_dir, train_smiles)
        if not hist:
            raise ValueError(
                "Could not infer training token length histogram from train_meta.json or the provided train SMILES."
            )
        sample_lengths = _sample_lengths_from_histogram(hist, n_samples=args.n_samples, max_cap=cfg_cap)
        print(f"[eval-stamp-diffusion] sampled_lengths_from_empirical_histogram n={len(sample_lengths)}", flush=True)
    else:
        print(f"[eval-stamp-diffusion] fixed_max_new_tokens={max_new_tokens}", flush=True)

    output_dir = resolve_branch_output_dir(
        args.output_dir,
        model_dir=args.model_dir,
    )

    run_denovo_sampling_eval(
        model_dir=args.model_dir,
        train_smiles=train_smiles,
        reference_smiles=reference_smiles,
        output_dir=output_dir,
        n_samples=args.n_samples,
        max_new_tokens=(max_new_tokens if max_new_tokens is not None else 1),
        temperature=args.temperature,
        top_k=args.top_k,
        diffusion_steps=args.diffusion_steps,
        diffusion_tokens_per_step=args.diffusion_tokens_per_step,
        diffusion_block_size=args.diffusion_block_size,
        diffusion_remasking=args.diffusion_remasking,
        diffusion_stochastic_transfer=(True if args.diffusion_stochastic_transfer else None),
        decode_batch_size=args.decode_batch_size,
        reference_max_samples=args.reference_max_samples,
        assume_train_canonical=(True if args.assume_train_canonical else None),
        eval_workers=args.eval_workers,
        eval_chunksize=args.eval_chunksize,
        show_progress=not args.no_tqdm,
        device=args.device,
        sample_lengths=sample_lengths,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
