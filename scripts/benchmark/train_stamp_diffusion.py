#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.data import load_smiles_list  # noqa: E402
from stamp.benchmark.lm import LMConfig, train_language_model  # noqa: E402
from stamp.vocab.branch_paths import resolve_branch_output_dir  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a STAMP diffusion LM without optional benchmark deps.")
    p.add_argument("--train", required=True, help="Train SMILES file.")
    p.add_argument("--valid", required=True, help="Valid SMILES file.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--representation", default="stamp")
    p.add_argument("--train-smiles-column", default="smiles")
    p.add_argument("--valid-smiles-column", default="smiles")
    p.add_argument(
        "--vocab-fit",
        action="append",
        default=[],
        help="Additional SMILES/CSV file used to fit the LM token vocabulary. Repeat for train/valid/test files.",
    )
    p.add_argument("--vocab-fit-smiles-column", default="smiles")
    p.add_argument("--stamp-vocab", default=None, help="PS-VAE vocab file for vocab-aware STAMP encoding.")
    p.add_argument("--encode-workers", type=int, default=1)
    p.add_argument("--encode-chunksize", type=int, default=128)
    p.add_argument("--no-tqdm", action="store_true")

    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-data-workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "none"])
    p.add_argument("--torch-compile", action="store_true")
    p.add_argument("--no-tf32", action="store_true")
    p.add_argument("--diffusion-steps", type=int, default=128)
    p.add_argument("--diffusion-time-epsilon", type=float, default=1e-3)
    p.add_argument("--diffusion-scheduler", default="linear", choices=["linear", "cosine"])
    p.add_argument("--diffusion-loss-weight", default="scheduler", choices=["scheduler", "uniform"])
    p.add_argument("--diffusion-remasking", default="low_confidence", choices=["low_confidence", "random"])
    p.add_argument("--diffusion-stochastic-transfer", action="store_true")
    p.add_argument("--diffusion-block-size", type=int, default=0)
    p.add_argument("--diffusion-tokens-per-step", type=int, default=0)
    p.add_argument("--diffusion-chemical-role-masking", action="store_true")
    p.add_argument("--init-from", default=None, help="Init weights from a trained model directory (e.g. AR checkpoint).")
    p.add_argument("--device", default="cuda")
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
    valid_smiles = load_smiles_list(
        args.valid,
        smiles_column=args.valid_smiles_column,
        canonicalize=True,
        deduplicate=False,
        show_progress=not args.no_tqdm,
    )
    vocab_fit_smiles = None
    if args.vocab_fit:
        merged: list[str] = []
        for path in args.vocab_fit:
            merged.extend(
                load_smiles_list(
                    path,
                    smiles_column=args.vocab_fit_smiles_column,
                    canonicalize=True,
                    deduplicate=False,
                    show_progress=not args.no_tqdm,
                )
            )
        vocab_fit_smiles = merged

    cfg = LMConfig(
        model_type="diffusion",
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        seed=args.seed,
        num_data_workers=args.num_data_workers,
        prefetch_factor=args.prefetch_factor,
        amp=args.amp,
        torch_compile=bool(args.torch_compile),
        tf32=not bool(args.no_tf32),
        diffusion_steps=args.diffusion_steps,
        diffusion_time_epsilon=args.diffusion_time_epsilon,
        diffusion_scheduler=args.diffusion_scheduler,
        diffusion_loss_weight=args.diffusion_loss_weight,
        diffusion_remasking=args.diffusion_remasking,
        diffusion_stochastic_transfer=bool(args.diffusion_stochastic_transfer),
        diffusion_block_size=args.diffusion_block_size,
        diffusion_tokens_per_step=args.diffusion_tokens_per_step,
        diffusion_chemical_role_masking=bool(args.diffusion_chemical_role_masking),
    )

    output_dir = resolve_branch_output_dir(
        args.output_dir,
        stamp_vocab=args.stamp_vocab,
    )

    train_language_model(
        train_smiles=train_smiles,
        valid_smiles=valid_smiles,
        representation=args.representation,
        output_dir=output_dir,
        vocab_fit_smiles=vocab_fit_smiles,
        cfg=cfg,
        show_progress=not args.no_tqdm,
        device=args.device,
        encode_workers=args.encode_workers,
        encode_chunksize=args.encode_chunksize,
        stamp_vocab=args.stamp_vocab,
        init_from=args.init_from,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
