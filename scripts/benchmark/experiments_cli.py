#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.experiments import (  # noqa: E402
    LMConfig,
    RoBERTaConfig,
    build_linker_design_dataset,
    build_fragment_infill_dataset,
    build_motif_extension_dataset,
    build_scaffold_completion_dataset,
    build_scaffold_decoration_dataset,
    build_scaffold_morphing_dataset,
    build_superstructure_generation_dataset,
    build_linker_infill_dataset,
    evaluate_fragment_infill,
    evaluate_linker_design,
    evaluate_scaffold_completion,
    finetune_roberta,
    load_smiles_list,
    prepare_dataset_splits,
    pretrain_roberta,
    run_denovo_sampling_eval,
    run_low_resource_experiments,
    run_property_prediction_benchmark,
    run_representation_benchmark,
    run_roberta_property_benchmark,
    save_benchmark_plots,
    save_experiment_markdown_tables,
    train_ao_fragment_infill,
    train_completion_model,
    train_language_model,
)
from stamp.vocab.branch_paths import resolve_branch_output_dir  # noqa: E402


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--preset",
        default="none",
        choices=["none", "moses_gpt20m"],
        help="Hyper-parameter preset. `moses_gpt20m` => 6 layers, 8 heads, d_model=768, epochs=10, lr=5e-4.",
    )
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
    p.add_argument(
        "--model-type",
        default="decoder",
        choices=["decoder", "diffusion", "ao_gpt", "ao_gpt_pp"],
        help="Language model type used by train-lm. decoder=AR, diffusion=MDLM, ao_gpt=any-order AR, ao_gpt_pp=AO with type-aware masking for property-prediction pretraining.",
    )
    p.add_argument("--ao-pp-frag-mask-rate", type=float, default=0.20, help="[ao_gpt_pp] fragment token mask rate.")
    p.add_argument("--ao-pp-topo-mask-rate", type=float, default=0.05, help="[ao_gpt_pp] topology token mask rate.")
    p.add_argument("--ao-pp-mix-uniform", type=float, default=0.50, help="[ao_gpt_pp] per-batch probability of uniform AO.")
    p.add_argument("--ao-pp-mix-type-aware", type=float, default=0.40, help="[ao_gpt_pp] per-batch probability of type-aware AO.")
    p.add_argument("--ao-pp-mix-identity", type=float, default=0.10, help="[ao_gpt_pp] per-batch probability of identity (L->R) order.")
    p.add_argument("--num-data-workers", type=int, default=0, help="DataLoader worker processes.")
    p.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor when num-data-workers > 0.")
    p.add_argument(
        "--amp",
        default="auto",
        choices=["auto", "bf16", "fp16", "none"],
        help="Mixed precision mode for CUDA training.",
    )
    p.add_argument("--torch-compile", action="store_true", help="Enable torch.compile for model forward.")
    p.add_argument("--no-tf32", action="store_true", help="Disable TF32 matmul/cudnn on CUDA.")
    p.add_argument(
        "--diffusion-steps",
        type=int,
        default=128,
        help="Diffusion denoising steps for training/sampling (diffusion mode).",
    )
    p.add_argument(
        "--diffusion-time-epsilon",
        type=float,
        default=1e-3,
        help="Lower bound of sampled timesteps t in diffusion training.",
    )
    p.add_argument(
        "--diffusion-scheduler",
        default="linear",
        choices=["linear", "cosine"],
        help="Alpha scheduler for diffusion masking.",
    )
    p.add_argument(
        "--diffusion-loss-weight",
        default="scheduler",
        choices=["scheduler", "uniform"],
        help="Per-token loss weighting in diffusion training.",
    )
    p.add_argument(
        "--diffusion-remasking",
        default="low_confidence",
        choices=["low_confidence", "random"],
        help="Remasking strategy during diffusion sampling.",
    )
    p.add_argument(
        "--diffusion-stochastic-transfer",
        action="store_true",
        help="Use stochastic binomial token transfer counts in diffusion sampling.",
    )
    p.add_argument(
        "--diffusion-block-size",
        type=int,
        default=0,
        help="Block size for diffusion sampling. <=0 means full generation window.",
    )
    p.add_argument(
        "--diffusion-tokens-per-step",
        type=int,
        default=0,
        help="Optional fixed token budget per diffusion denoising step. 0 keeps scheduler-driven counts.",
    )
    p.add_argument(
        "--diffusion-chemical-role-masking",
        action="store_true",
        help="Enable chemical-role-aware masking in diffusion training.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device, e.g. cuda / cuda:0 / cpu / auto. Default: cuda.",
    )


def _cfg_from_args(args: argparse.Namespace) -> LMConfig:
    cfg = LMConfig(
        model_type=args.model_type,
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
        ao_pp_frag_mask_rate=float(getattr(args, "ao_pp_frag_mask_rate", 0.20)),
        ao_pp_topo_mask_rate=float(getattr(args, "ao_pp_topo_mask_rate", 0.05)),
        ao_pp_mix_uniform=float(getattr(args, "ao_pp_mix_uniform", 0.50)),
        ao_pp_mix_type_aware=float(getattr(args, "ao_pp_mix_type_aware", 0.40)),
        ao_pp_mix_identity=float(getattr(args, "ao_pp_mix_identity", 0.10)),
    )
    if getattr(args, "preset", "none") == "moses_gpt20m":
        cfg.d_model = 768
        cfg.n_heads = 8
        cfg.n_layers = 6
        cfg.epochs = 10
        cfg.lr = 5e-4
    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="STAMP lightweight benchmark CLI (Experiments A-E)."
    )
    sp = p.add_subparsers(dest="cmd", required=True)

    p_prepare = sp.add_parser("prepare-data", help="Prepare train/valid/test + low-resource splits.")
    p_prepare.add_argument("--input", required=True)
    p_prepare.add_argument("--smiles-column", default="smiles")
    p_prepare.add_argument("--output-dir", default="results/benchmark")
    p_prepare.add_argument("--train-size", type=int, default=100000)
    p_prepare.add_argument("--valid-size", type=int, default=10000)
    p_prepare.add_argument("--test-size", type=int, default=10000)
    p_prepare.add_argument("--low-resource-sizes", default="10000,50000,100000")
    p_prepare.add_argument("--seed", type=int, default=42)
    p_prepare.add_argument("--max-mols", type=int, default=None)
    p_prepare.add_argument("--no-tqdm", action="store_true")

    p_stats = sp.add_parser("repr-stats", help="Run Experiment A representation statistics.")
    p_stats.add_argument("--input", required=True)
    p_stats.add_argument("--smiles-column", default="smiles")
    p_stats.add_argument("--output-dir", default="results/benchmark/expA_repr_stats")
    p_stats.add_argument(
        "--representations",
        default="smiles,stamp,selfies,safe,t-smiles,group-selfies,fragsmiles",
        help="Comma-separated representation names.",
    )
    p_stats.add_argument("--max-mols", type=int, default=20000)
    p_stats.add_argument("--workers", type=int, default=1, help="Parallel worker processes across representations.")
    p_stats.add_argument("--no-tqdm", action="store_true")

    p_train = sp.add_parser("train-lm", help="Train decoder-only LM for one representation (Experiment B).")
    p_train.add_argument("--train", required=True, help="Train SMILES file.")
    p_train.add_argument("--valid", required=True, help="Valid SMILES file.")
    p_train.add_argument("--representation", required=True)
    p_train.add_argument("--output-dir", required=True)
    p_train.add_argument(
        "--encode-workers",
        type=int,
        default=1,
        help="Number of CPU workers for representation tokenization. Use >1 to speed up STAMP encoding.",
    )
    p_train.add_argument(
        "--encode-chunksize",
        type=int,
        default=128,
        help="Chunksize for multiprocessing tokenization map.",
    )
    p_train.add_argument("--no-tqdm", action="store_true")
    p_train.add_argument(
        "--stamp-vocab",
        default=None,
        help="PS-VAE vocab file for STAMP representation. If set, enables vocab-aware encoding.",
    )
    p_train.add_argument(
        "--stamp-force-ring-aware-encode",
        action="store_true",
        help="[STAMP] Force ring-aware encode: open ring single bonds, treat full vocab as blessed rings. Dramatically reduces OOV on unseen scaffolds by decomposing rare rings into in-vocab sub-fragments. Recommended for property-prediction pretraining.",
    )
    _add_model_args(p_train)

    p_denovo = sp.add_parser("sample-eval", help="Sample and evaluate de novo generation (Experiment B).")
    p_denovo.add_argument("--model-dir", required=True)
    p_denovo.add_argument("--train", required=True, help="Train SMILES file (for novelty).")
    p_denovo.add_argument("--reference", required=True, help="Reference/test SMILES file.")
    p_denovo.add_argument("--output-dir", required=True)
    p_denovo.add_argument("--n-samples", type=int, default=10000)
    p_denovo.add_argument("--max-new-tokens", type=int, default=256)
    p_denovo.add_argument("--temperature", type=float, default=1.0)
    p_denovo.add_argument("--top-k", type=int, default=50)
    p_denovo.add_argument(
        "--ao-decode-mode",
        default="ar",
        choices=["ar", "ao_canvas_confidence"],
        help="For ao_gpt checkpoints: standard left-to-right AR or AO canvas confidence filling.",
    )
    p_denovo.add_argument(
        "--ao-canvas-placeholder",
        default="unk",
        choices=["unk", "pad"],
        help="Placeholder token used by ao_canvas_confidence sampler for unfilled positions.",
    )
    p_denovo.add_argument(
        "--ao-refine-steps",
        type=int,
        default=0,
        help="For STAMP ao_gpt checkpoints, run this many self-confidence span-refine passes after draft generation.",
    )
    p_denovo.add_argument(
        "--ao-refine-top-spans",
        type=int,
        default=1,
        help="At each refine step, consider this many lowest-confidence STAMP subtrees.",
    )
    p_denovo.add_argument(
        "--ao-refine-max-span-len",
        type=int,
        default=12,
        help="Maximum STAMP subtree token span length eligible for AO refine.",
    )
    p_denovo.add_argument(
        "--ao-refine-samples-per-span",
        type=int,
        default=4,
        help="Number of AO bidirectional resamples to try per selected span.",
    )
    p_denovo.add_argument(
        "--ao-refine-temperature",
        type=float,
        default=None,
        help="Optional temperature override for AO span-refine resampling.",
    )
    p_denovo.add_argument(
        "--ao-refine-top-k",
        type=int,
        default=None,
        help="Optional top-k override for AO span-refine resampling.",
    )
    p_denovo.add_argument(
        "--ao-refine-min-gain",
        type=float,
        default=0.0,
        help="Only accept AO span edits whose normalized span log-prob improves by more than this margin.",
    )
    p_denovo.add_argument("--diffusion-steps", type=int, default=None, help="Optional override for diffusion sampling steps.")
    p_denovo.add_argument(
        "--diffusion-tokens-per-step",
        type=int,
        default=None,
        help="Optional override for fixed diffusion token budget per denoising step.",
    )
    p_denovo.add_argument(
        "--diffusion-block-size",
        type=int,
        default=None,
        help="Optional override for diffusion sampling block size.",
    )
    p_denovo.add_argument(
        "--diffusion-remasking",
        default=None,
        choices=["low_confidence", "random"],
        help="Optional override for diffusion remasking strategy.",
    )
    p_denovo.add_argument(
        "--diffusion-stochastic-transfer",
        action="store_true",
        help="Enable stochastic token transfer counts in diffusion sampling.",
    )
    p_denovo.add_argument("--decode-batch-size", type=int, default=256)
    p_denovo.add_argument(
        "--eval-workers",
        type=int,
        default=1,
        help="CPU worker processes for evaluate_generated_smiles(). Use >1 for faster post-sampling validation/metrics.",
    )
    p_denovo.add_argument(
        "--eval-chunksize",
        type=int,
        default=256,
        help="Chunksize used by evaluation process pools.",
    )
    p_denovo.add_argument(
        "--reference-max-samples",
        type=int,
        default=None,
        help="Optional cap for reference set size in distribution-property evaluation (speed/accuracy tradeoff).",
    )
    p_denovo.add_argument(
        "--assume-train-canonical",
        action="store_true",
        help="Skip canonicalization of full train set when computing novelty.",
    )
    p_denovo.add_argument("--device", default="cuda")
    p_denovo.add_argument("--no-tqdm", action="store_true")

    p_scaffold = sp.add_parser("build-scaffold", help="Build scaffold-completion dataset (Experiment C).")
    p_scaffold.add_argument("--input", required=True)
    p_scaffold.add_argument("--smiles-column", default="smiles")
    p_scaffold.add_argument("--output", required=True)
    p_scaffold.add_argument("--min-scaffold-heavy-atoms", type=int, default=5)
    p_scaffold.add_argument("--max-mols", type=int, default=None)
    p_scaffold.add_argument("--workers", type=int, default=1, help="CPU workers for scaffold extraction.")
    p_scaffold.add_argument("--chunksize", type=int, default=128, help="Multiprocessing chunksize.")
    p_scaffold.add_argument("--no-tqdm", action="store_true")

    p_linker = sp.add_parser("build-linker", help="Build linker-design dataset (Experiment D).")
    p_linker.add_argument("--input", required=True)
    p_linker.add_argument("--smiles-column", default="smiles")
    p_linker.add_argument("--output", required=True)
    p_linker.add_argument("--max-mols", type=int, default=None)
    p_linker.add_argument("--workers", type=int, default=1, help="CPU workers for linker extraction.")
    p_linker.add_argument("--chunksize", type=int, default=128, help="Multiprocessing chunksize.")
    p_linker.add_argument("--no-tqdm", action="store_true")

    p_morph = sp.add_parser("build-scaffold-morphing", help="Build scaffold-morphing dataset.")
    p_morph.add_argument("--input", required=True)
    p_morph.add_argument("--smiles-column", default="smiles")
    p_morph.add_argument("--output", required=True)
    p_morph.add_argument("--max-mols", type=int, default=None)
    p_morph.add_argument("--workers", type=int, default=1, help="CPU workers for scaffold-morphing extraction.")
    p_morph.add_argument("--chunksize", type=int, default=128, help="Multiprocessing chunksize.")
    p_morph.add_argument("--no-tqdm", action="store_true")

    p_linker_infill = sp.add_parser("build-linker-infill", help="Build linker-design infill dataset.")
    p_linker_infill.add_argument("--input", required=True)
    p_linker_infill.add_argument("--smiles-column", default="smiles")
    p_linker_infill.add_argument("--output", required=True)
    p_linker_infill.add_argument("--representation", default="stamp")
    p_linker_infill.add_argument("--stamp-vocab", default=None)
    p_linker_infill.add_argument("--max-mols", type=int, default=None)
    p_linker_infill.add_argument("--workers", type=int, default=1)
    p_linker_infill.add_argument("--chunksize", type=int, default=128)
    p_linker_infill.add_argument("--no-tqdm", action="store_true")

    p_morph_infill = sp.add_parser("build-scaffold-morphing-infill", help="Build scaffold-morphing infill dataset.")
    p_morph_infill.add_argument("--input", required=True)
    p_morph_infill.add_argument("--smiles-column", default="smiles")
    p_morph_infill.add_argument("--output", required=True)
    p_morph_infill.add_argument("--representation", default="stamp")
    p_morph_infill.add_argument("--stamp-vocab", default=None)
    p_morph_infill.add_argument("--max-mols", type=int, default=None)
    p_morph_infill.add_argument("--workers", type=int, default=1)
    p_morph_infill.add_argument("--chunksize", type=int, default=128)
    p_morph_infill.add_argument("--no-tqdm", action="store_true")

    p_motif_ext = sp.add_parser("build-motif-extension", help="Build motif-extension infill dataset.")
    p_motif_ext.add_argument("--input", required=True)
    p_motif_ext.add_argument("--smiles-column", default="smiles")
    p_motif_ext.add_argument("--output", required=True)
    p_motif_ext.add_argument("--representation", default="stamp")
    p_motif_ext.add_argument("--stamp-vocab", default=None)
    p_motif_ext.add_argument("--max-mols", type=int, default=None)
    p_motif_ext.add_argument("--seed", type=int, default=42)
    p_motif_ext.add_argument("--no-tqdm", action="store_true")

    p_scaf_dec = sp.add_parser("build-scaffold-decoration", help="Build scaffold-decoration infill dataset.")
    p_scaf_dec.add_argument("--input", required=True)
    p_scaf_dec.add_argument("--smiles-column", default="smiles")
    p_scaf_dec.add_argument("--output", required=True)
    p_scaf_dec.add_argument("--representation", default="stamp")
    p_scaf_dec.add_argument("--stamp-vocab", default=None)
    p_scaf_dec.add_argument("--max-mols", type=int, default=None)
    p_scaf_dec.add_argument("--min-scaffold-heavy-atoms", type=int, default=5)
    p_scaf_dec.add_argument("--no-tqdm", action="store_true")

    p_super = sp.add_parser("build-superstructure-generation", help="Build superstructure-generation infill dataset.")
    p_super.add_argument("--input", required=True)
    p_super.add_argument("--smiles-column", default="smiles")
    p_super.add_argument("--output", required=True)
    p_super.add_argument("--representation", default="stamp")
    p_super.add_argument("--stamp-vocab", default=None)
    p_super.add_argument("--max-mols", type=int, default=None)
    p_super.add_argument("--min-keep-groups", type=int, default=1)
    p_super.add_argument("--max-keep-groups", type=int, default=3)
    p_super.add_argument("--seed", type=int, default=42)
    p_super.add_argument("--no-tqdm", action="store_true")

    p_train_comp = sp.add_parser("train-completion", help="Train prefix completion model for scaffold/linker task.")
    p_train_comp.add_argument("--csv", required=True)
    p_train_comp.add_argument("--representation", required=True)
    p_train_comp.add_argument("--stamp-vocab", default=None)
    p_train_comp.add_argument("--prompt-key", required=True)
    p_train_comp.add_argument("--target-key", required=True)
    p_train_comp.add_argument("--prompt-is-repr", action="store_true")
    p_train_comp.add_argument("--target-is-repr", action="store_true")
    p_train_comp.add_argument(
        "--prompt-template",
        default=None,
        help="Optional format template (e.g. '{fragment_a}.{fragment_b}') to build prompt.",
    )
    p_train_comp.add_argument(
        "--target-template",
        default=None,
        help="Optional format template to build target string.",
    )
    p_train_comp.add_argument(
        "--encode-workers",
        type=int,
        default=1,
        help="CPU workers for representation tokenization in completion training.",
    )
    p_train_comp.add_argument(
        "--encode-chunksize",
        type=int,
        default=128,
        help="Chunksize for multiprocessing tokenization map.",
    )
    p_train_comp.add_argument("--output-dir", required=True)
    p_train_comp.add_argument("--no-tqdm", action="store_true")
    _add_model_args(p_train_comp)

    p_eval_scaf = sp.add_parser("eval-scaffold", help="Evaluate scaffold completion model (Experiment C).")
    p_eval_scaf.add_argument("--model-dir", required=True)
    p_eval_scaf.add_argument("--csv", required=True)
    p_eval_scaf.add_argument("--output-dir", required=True)
    p_eval_scaf.add_argument("--max-eval", type=int, default=1000)
    p_eval_scaf.add_argument("--n-samples-per-prompt", type=int, default=1)
    p_eval_scaf.add_argument("--max-new-tokens", type=int, default=256)
    p_eval_scaf.add_argument("--temperature", type=float, default=1.0)
    p_eval_scaf.add_argument("--top-k", type=int, default=50)
    p_eval_scaf.add_argument("--eval-workers", type=int, default=1, help="CPU workers for post-decode evaluation.")
    p_eval_scaf.add_argument("--eval-chunksize", type=int, default=256, help="Multiprocessing chunksize.")
    p_eval_scaf.add_argument("--device", default="cuda")
    p_eval_scaf.add_argument("--no-tqdm", action="store_true")

    p_eval_link = sp.add_parser("eval-linker", help="Evaluate linker design model (Experiment D).")
    p_eval_link.add_argument("--model-dir", required=True)
    p_eval_link.add_argument("--csv", required=True)
    p_eval_link.add_argument("--output-dir", required=True)
    p_eval_link.add_argument("--max-eval", type=int, default=1000)
    p_eval_link.add_argument("--n-samples-per-prompt", type=int, default=1)
    p_eval_link.add_argument("--max-new-tokens", type=int, default=256)
    p_eval_link.add_argument("--temperature", type=float, default=1.0)
    p_eval_link.add_argument("--top-k", type=int, default=50)
    p_eval_link.add_argument("--eval-workers", type=int, default=1, help="CPU workers for post-decode evaluation.")
    p_eval_link.add_argument("--eval-chunksize", type=int, default=256, help="Multiprocessing chunksize.")
    p_eval_link.add_argument("--device", default="cuda")
    p_eval_link.add_argument("--no-tqdm", action="store_true")

    p_infill = sp.add_parser("build-fragment-infill", help="Build random fragment-span infill dataset.")
    p_infill.add_argument("--input", required=True)
    p_infill.add_argument("--smiles-column", default="smiles")
    p_infill.add_argument("--output", required=True)
    p_infill.add_argument("--representation", default="stamp")
    p_infill.add_argument("--stamp-vocab", default=None)
    p_infill.add_argument("--max-mols", type=int, default=None)
    p_infill.add_argument("--min-mask-groups", type=int, default=1)
    p_infill.add_argument("--max-mask-groups", type=int, default=3)
    p_infill.add_argument("--seed", type=int, default=42)
    p_infill.add_argument("--no-tqdm", action="store_true")

    p_eval_infill = sp.add_parser("eval-fragment-infill", help="Evaluate fragment-span infill model.")
    p_eval_infill.add_argument("--model-dir", required=True)
    p_eval_infill.add_argument("--csv", required=True)
    p_eval_infill.add_argument("--output-dir", required=True)
    p_eval_infill.add_argument("--max-eval", type=int, default=1000)
    p_eval_infill.add_argument("--n-samples-per-prompt", type=int, default=1)
    p_eval_infill.add_argument("--max-new-tokens", type=int, default=128)
    p_eval_infill.add_argument("--temperature", type=float, default=1.0)
    p_eval_infill.add_argument("--top-k", type=int, default=50)
    p_eval_infill.add_argument("--eval-workers", type=int, default=1)
    p_eval_infill.add_argument("--eval-chunksize", type=int, default=256)
    p_eval_infill.add_argument("--device", default="cuda")
    p_eval_infill.add_argument("--no-tqdm", action="store_true")

    p_train_ao_infill = sp.add_parser("train-ao-fragment-infill", help="Train AO-GPT with in-place masked fragment reconstruction.")
    p_train_ao_infill.add_argument("--csv", required=True)
    p_train_ao_infill.add_argument("--representation", default="stamp")
    p_train_ao_infill.add_argument("--stamp-vocab", default=None)
    p_train_ao_infill.add_argument("--output-dir", required=True)
    p_train_ao_infill.add_argument("--no-tqdm", action="store_true")
    _add_model_args(p_train_ao_infill)

    p_low = sp.add_parser("run-low-resource", help="Run low-resource sweep (Experiment E).")
    p_low.add_argument("--split-dir", required=True, help="Directory containing train/valid/test + low_resource subdir.")
    p_low.add_argument("--output-dir", required=True)
    p_low.add_argument("--representations", default="smiles,stamp,selfies")
    p_low.add_argument("--sizes", default="10000,50000,100000")
    p_low.add_argument("--epochs", type=int, default=5)
    p_low.add_argument("--batch-size", type=int, default=64)
    p_low.add_argument("--max-seq-len", type=int, default=256)
    p_low.add_argument("--n-samples", type=int, default=2000)
    p_low.add_argument("--decode-batch-size", type=int, default=256)
    p_low.add_argument("--encode-workers", type=int, default=1)
    p_low.add_argument("--encode-chunksize", type=int, default=128)
    p_low.add_argument("--eval-workers", type=int, default=1)
    p_low.add_argument("--eval-chunksize", type=int, default=256)
    p_low.add_argument("--device", default="cuda")
    p_low.add_argument("--num-data-workers", type=int, default=0)
    p_low.add_argument("--prefetch-factor", type=int, default=2)
    p_low.add_argument(
        "--amp",
        default="auto",
        choices=["auto", "bf16", "fp16", "none"],
        help="Mixed precision mode for CUDA training.",
    )
    p_low.add_argument("--torch-compile", action="store_true", help="Enable torch.compile for model forward.")
    p_low.add_argument("--no-tf32", action="store_true", help="Disable TF32 matmul/cudnn on CUDA.")
    p_low.add_argument("--model-type", default="decoder", choices=["decoder", "diffusion"])
    p_low.add_argument("--diffusion-steps", type=int, default=128)
    p_low.add_argument("--diffusion-time-epsilon", type=float, default=1e-3)
    p_low.add_argument("--diffusion-scheduler", default="linear", choices=["linear", "cosine"])
    p_low.add_argument("--diffusion-loss-weight", default="scheduler", choices=["scheduler", "uniform"])
    p_low.add_argument("--diffusion-remasking", default="low_confidence", choices=["low_confidence", "random"])
    p_low.add_argument("--diffusion-stochastic-transfer", action="store_true")
    p_low.add_argument("--diffusion-block-size", type=int, default=0)
    p_low.add_argument("--no-tqdm", action="store_true")

    p_table = sp.add_parser("make-tables", help="Generate markdown tables for paper/report.")
    p_table.add_argument("--repr-stats-json", required=True)
    p_table.add_argument("--denovo-metrics-json", required=True, nargs="+")
    p_table.add_argument("--output-md", required=True)

    p_plot = sp.add_parser("make-plots", help="Generate PNG plots from benchmark outputs.")
    p_plot.add_argument("--run-root", required=True, help="Benchmark run root directory.")
    p_plot.add_argument("--output-dir", default=None, help="Optional output dir (default: <run-root>/plots).")

    p_prop = sp.add_parser("property-pred", help="Run property prediction benchmark (Experiment F).")
    p_prop.add_argument("--input", required=True, help="CSV containing smiles + target property column.")
    p_prop.add_argument("--output-dir", required=True)
    p_prop.add_argument(
        "--task-type",
        default="regression",
        choices=["regression", "classification"],
        help="Task type for metric computation.",
    )
    p_prop.add_argument("--target-column", required=True, help="Regression target column name.")
    p_prop.add_argument("--smiles-column", default="smiles")
    p_prop.add_argument(
        "--representations",
        default="smiles,stamp,selfies,safe,group-selfies,fragsmiles",
        help="Comma-separated representation names.",
    )
    p_prop.add_argument("--test-size", type=float, default=0.2)
    p_prop.add_argument("--seed", type=int, default=42)
    p_prop.add_argument("--max-mols", type=int, default=None)
    p_prop.add_argument("--n-features", type=int, default=2048, help="Hashed feature dimension.")
    p_prop.add_argument("--ridge-alpha", type=float, default=1e-3, help="L2 regularization strength.")
    p_prop.add_argument("--no-deduplicate", action="store_true", help="Keep duplicate molecules.")
    p_prop.add_argument("--no-tqdm", action="store_true")

    # --- RoBERTa pre-train + fine-tune (Experiment G) ---

    p_pt = sp.add_parser(
        "pretrain-roberta",
        help="Pre-train a RoBERTa MLM encoder for one representation (Experiment G step 1).",
    )
    p_pt.add_argument("--train", required=True, help="Train SMILES file.")
    p_pt.add_argument("--valid", required=True, help="Valid SMILES file.")
    p_pt.add_argument("--representation", required=True)
    p_pt.add_argument("--output-dir", required=True)
    p_pt.add_argument("--d-model", type=int, default=768)
    p_pt.add_argument("--n-heads", type=int, default=12)
    p_pt.add_argument("--n-layers", type=int, default=6)
    p_pt.add_argument("--d-ff", type=int, default=1536)
    p_pt.add_argument("--dropout", type=float, default=0.1)
    p_pt.add_argument("--max-seq-len", type=int, default=512)
    p_pt.add_argument("--mask-prob", type=float, default=0.15)
    p_pt.add_argument(
        "--type-aware-masking",
        action="store_true",
        help="[STAMP] Mask fragment vs topology tokens at different rates (frag=0.20, topo=0.05 by default).",
    )
    p_pt.add_argument("--frag-mask-prob", type=float, default=0.20, help="[type-aware] mask rate for fragment (motif) tokens.")
    p_pt.add_argument("--topo-mask-prob", type=float, default=0.05, help="[type-aware] mask rate for topology tokens.")
    p_pt.add_argument(
        "--stamp-force-ring-aware-encode",
        action="store_true",
        help="[STAMP] Force ring-aware encode for the pretraining tokenizer.",
    )
    p_pt.add_argument("--stamp-vocab", default=None, help="[STAMP] PS-VAE vocab file for STAMP representation.")
    p_pt.add_argument("--batch-size", type=int, default=32)
    p_pt.add_argument("--lr", type=float, default=5e-4)
    p_pt.add_argument("--weight-decay", type=float, default=0.01)
    p_pt.add_argument("--pretrain-epochs", type=int, default=20)
    p_pt.add_argument("--seed", type=int, default=42)
    p_pt.add_argument("--num-data-workers", type=int, default=0)
    p_pt.add_argument("--prefetch-factor", type=int, default=2)
    p_pt.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "none"])
    p_pt.add_argument("--torch-compile", action="store_true")
    p_pt.add_argument("--no-tf32", action="store_true")
    p_pt.add_argument("--device", default="cuda")
    p_pt.add_argument("--encode-workers", type=int, default=1)
    p_pt.add_argument("--encode-chunksize", type=int, default=128)
    p_pt.add_argument("--no-tqdm", action="store_true")

    p_ft = sp.add_parser(
        "finetune-roberta",
        help="Fine-tune a pre-trained RoBERTa on a MoleculeNet task (Experiment G step 2).",
    )
    p_ft.add_argument("--pretrained-dir", required=True, help="Directory with pretrained_roberta.pt.")
    p_ft.add_argument("--task", required=True, choices=["bbbp", "hiv", "tox21", "bace", "clintox", "sider", "muv", "esol", "freesolv", "lipophilicity"], help="MoleculeNet task name.")
    p_ft.add_argument("--output-dir", required=True)
    p_ft.add_argument("--data-dir", default=None, help="Directory for MoleculeNet data (DeepChem or local CSV).")
    p_ft.add_argument("--split-seed", type=int, default=42)
    p_ft.add_argument("--finetune-epochs", type=int, default=20)
    p_ft.add_argument("--finetune-lr", type=float, default=2e-5)
    p_ft.add_argument("--finetune-batch-size", type=int, default=32)
    p_ft.add_argument("--dropout", type=float, default=0.1)
    p_ft.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "none"])
    p_ft.add_argument("--no-tf32", action="store_true")
    p_ft.add_argument("--device", default="cuda")
    p_ft.add_argument("--encode-workers", type=int, default=1)
    p_ft.add_argument("--encode-chunksize", type=int, default=128)
    p_ft.add_argument("--num-data-workers", type=int, default=0)
    p_ft.add_argument("--prefetch-factor", type=int, default=2)
    p_ft.add_argument("--no-tqdm", action="store_true")

    p_rb = sp.add_parser(
        "run-roberta-benchmark",
        help="Full RoBERTa pre-train + fine-tune benchmark across representations and tasks (Experiment G).",
    )
    p_rb.add_argument("--train", required=True, help="Train SMILES file for pre-training.")
    p_rb.add_argument("--valid", required=True, help="Valid SMILES file for pre-training.")
    p_rb.add_argument("--output-dir", required=True)
    p_rb.add_argument(
        "--representations",
        default="smiles,stamp,selfies,safe,group-selfies,fragsmiles",
        help="Comma-separated representation names.",
    )
    p_rb.add_argument("--tasks", default="bbbp,hiv,tox21,bace,clintox,sider,muv,esol,freesolv,lipophilicity", help="Comma-separated MoleculeNet tasks.")
    p_rb.add_argument("--data-dir", default=None, help="Directory for MoleculeNet data.")
    p_rb.add_argument("--split-seed", type=int, default=42)
    p_rb.add_argument("--d-model", type=int, default=768)
    p_rb.add_argument("--n-heads", type=int, default=12)
    p_rb.add_argument("--n-layers", type=int, default=6)
    p_rb.add_argument("--d-ff", type=int, default=1536)
    p_rb.add_argument("--dropout", type=float, default=0.1)
    p_rb.add_argument("--max-seq-len", type=int, default=512)
    p_rb.add_argument("--mask-prob", type=float, default=0.15)
    p_rb.add_argument("--batch-size", type=int, default=32)
    p_rb.add_argument("--lr", type=float, default=5e-4)
    p_rb.add_argument("--weight-decay", type=float, default=0.01)
    p_rb.add_argument("--pretrain-epochs", type=int, default=20)
    p_rb.add_argument("--finetune-epochs", type=int, default=20)
    p_rb.add_argument("--finetune-lr", type=float, default=2e-5)
    p_rb.add_argument("--finetune-batch-size", type=int, default=32)
    p_rb.add_argument("--seed", type=int, default=42)
    p_rb.add_argument("--num-data-workers", type=int, default=0)
    p_rb.add_argument("--prefetch-factor", type=int, default=2)
    p_rb.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "none"])
    p_rb.add_argument("--torch-compile", action="store_true")
    p_rb.add_argument("--no-tf32", action="store_true")
    p_rb.add_argument("--device", default="cuda")
    p_rb.add_argument("--encode-workers", type=int, default=1)
    p_rb.add_argument("--encode-chunksize", type=int, default=128)
    p_rb.add_argument("--no-tqdm", action="store_true")

    def _add_mine_vocab_args(parser: argparse.ArgumentParser, *, ring_default: bool, include_toggle: bool) -> None:
        parser.add_argument("--input", required=True, help="Input CSV/TXT file with SMILES.")
        parser.add_argument("--smiles-column", default="smiles")
        parser.add_argument("--output", required=True, help="Output vocab file path (*.txt).")
        parser.add_argument("--vocab-size", type=int, default=500, help="Target vocabulary size (including seed atoms).")
        parser.add_argument("--max-mols", type=int, default=None, help="Max molecules to read from input file.")
        parser.add_argument(
            "--max-mining-mols",
            type=int,
            default=50000,
            help="Subsample this many mols for BPE mining (0=use all). Default 50000 is fast and statistically robust.",
        )
        parser.add_argument("--kekulize", action="store_true", default=True, help="Kekulize molecules before mining (default: True).")
        parser.add_argument("--no-kekulize", dest="kekulize", action="store_false", help="Disable kekulization.")
        if include_toggle:
            parser.add_argument(
                "--allow-ring-single-edges",
                action="store_true",
                help="Enable the ring-aware mining/tokenization branch: keep ring single bonds open so vocab motifs may require [Ring] closures at encode time.",
            )
        parser.add_argument("--workers", type=int, default=1, help="Parallel workers for BPE frequency counting.")
        parser.add_argument(
            "--ring-blessed-freq",
            type=int,
            default=0,
            help="Frequency-gated ring pre-merge. When >0 and ring-aware enabled, ring systems appearing in >= this many mols are pre-merged as seeds (benzene, pyridine); rarer rings are opened for BPE. 0 disables gating.",
        )
        parser.add_argument(
            "--min-bpe-merge-freq",
            type=int,
            default=0,
            help="Early-stop BPE when the best merge candidate's freq falls below this threshold. Useful to cap long-tail rare merges. 0 disables.",
        )
        parser.add_argument(
            "--blessed-rings-vocab",
            type=str,
            default=None,
            help="Path to an existing PS-VAE vocab.txt whose entries are used directly as blessed rings for ring-aware mining (skips auto-frequency precompute).",
        )
        parser.add_argument("--no-tqdm", action="store_true")
        parser.set_defaults(allow_ring_single_edges=ring_default)

    p_mine = sp.add_parser(
        "mine-vocab",
        help="Mine the legacy tree-only PS-VAE principal subgraph vocabulary. Add --allow-ring-single-edges to switch to the ring-aware branch.",
    )
    _add_mine_vocab_args(p_mine, ring_default=False, include_toggle=True)

    p_mine_ring = sp.add_parser(
        "mine-vocab-ring",
        help="Mine the ring-aware PS-VAE vocabulary branch that may emit [Ring] closures at encode time.",
    )
    _add_mine_vocab_args(p_mine_ring, ring_default=True, include_toggle=False)

    return p


def _load_smiles_txt(path: str | Path) -> list[str]:
    return [s.strip() for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def _load_csv_records(path: str | Path) -> list[dict[str, str]]:
    import csv

    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v or "") for k, v in row.items()})
    return rows


def _branch_output_from_vocab(output_dir: str | Path, stamp_vocab: str | None) -> str:
    return str(resolve_branch_output_dir(output_dir, stamp_vocab=stamp_vocab))


def _branch_output_from_model(output_dir: str | Path, model_dir: str | Path) -> str:
    return str(resolve_branch_output_dir(output_dir, model_dir=model_dir))


def main() -> int:
    args = build_parser().parse_args()
    RDLogger.DisableLog("rdApp.*")

    if args.cmd == "prepare-data":
        sizes = [int(x) for x in args.low_resource_sizes.split(",") if x.strip()]
        res = prepare_dataset_splits(
            input_path=args.input,
            output_dir=args.output_dir,
            smiles_column=args.smiles_column,
            train_size=args.train_size,
            valid_size=args.valid_size,
            test_size=args.test_size,
            low_resource_sizes=sizes,
            seed=args.seed,
            max_mols=args.max_mols,
            show_progress=not args.no_tqdm,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "repr-stats":
        smiles = load_smiles_list(
            args.input,
            smiles_column=args.smiles_column,
            max_mols=args.max_mols,
            canonicalize=True,
            deduplicate=True,
            show_progress=not args.no_tqdm,
        )
        reps = [x.strip() for x in args.representations.split(",") if x.strip()]
        res = run_representation_benchmark(
            smiles_list=smiles,
            representations=reps,
            output_dir=args.output_dir,
            max_items=args.max_mols,
            show_progress=not args.no_tqdm,
            num_workers=args.workers,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd in {"mine-vocab", "mine-vocab-ring"}:
        from stamp.vocab.principal_subgraph import mine_principal_subgraphs, PSVAEMineConfig
        from stamp.vocab.vocab_builder import iter_smiles_from_file

        smis = list(iter_smiles_from_file(
            args.input,
            smiles_column=args.smiles_column,
            max_mols=args.max_mols,
        ))
        blessed_preloaded = None
        if args.blessed_rings_vocab:
            import json as _json
            from pathlib import Path as _P
            vp = _P(args.blessed_rings_vocab)
            with vp.open("r", encoding="utf-8") as f:
                lines = f.read().strip().split("\n")
            # Skip header line; each vocab row is "smi\tatom_num\tfreq"
            items = []
            for line in lines[1:]:
                parts = line.strip().split("\t")
                if len(parts) >= 3 and int(parts[1]) >= 3:
                    items.append(parts[0])
            blessed_preloaded = frozenset(items)
            print(f"[cli] loaded {len(blessed_preloaded)} blessed rings from {vp}", flush=True)
        cfg = PSVAEMineConfig(
            vocab_size=args.vocab_size,
            kekulize=args.kekulize,
            num_workers=args.workers,
            show_progress=not args.no_tqdm,
            max_mining_mols=args.max_mining_mols,
            allow_ring_single_edges=bool(args.allow_ring_single_edges),
            ring_blessed_freq_threshold=int(args.ring_blessed_freq),
            min_bpe_merge_freq=int(args.min_bpe_merge_freq),
            blessed_rings_preloaded=blessed_preloaded,
        )
        selected, details = mine_principal_subgraphs(smis, cfg, args.output)
        print(json.dumps({
            "vocab_path": str(args.output),
            "vocab_size": len(selected),
            "kekulize": args.kekulize,
            "allow_ring_single_edges": bool(args.allow_ring_single_edges),
            "branch_mode": ("ring_single_edges" if bool(args.allow_ring_single_edges) else "legacy_tree"),
            "input_mols": len(smis),
        }, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "train-lm":
        train = _load_smiles_txt(args.train)
        valid = _load_smiles_txt(args.valid)
        cfg = _cfg_from_args(args)
        res = train_language_model(
            train_smiles=train,
            valid_smiles=valid,
            representation=args.representation,
            output_dir=_branch_output_from_vocab(args.output_dir, args.stamp_vocab),
            cfg=cfg,
            show_progress=not args.no_tqdm,
            device=args.device,
            encode_workers=args.encode_workers,
            encode_chunksize=args.encode_chunksize,
            stamp_vocab=args.stamp_vocab,
            stamp_force_ring_aware_encode=bool(getattr(args, "stamp_force_ring_aware_encode", False)),
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "sample-eval":
        train = _load_smiles_txt(args.train)
        ref = _load_smiles_txt(args.reference)
        res = run_denovo_sampling_eval(
            model_dir=args.model_dir,
            train_smiles=train,
            reference_smiles=ref,
            output_dir=_branch_output_from_model(args.output_dir, args.model_dir),
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            ao_decode_mode=args.ao_decode_mode,
            ao_canvas_placeholder=args.ao_canvas_placeholder,
            diffusion_steps=args.diffusion_steps,
            diffusion_tokens_per_step=args.diffusion_tokens_per_step,
            diffusion_block_size=args.diffusion_block_size,
            diffusion_remasking=args.diffusion_remasking,
            diffusion_stochastic_transfer=(True if args.diffusion_stochastic_transfer else None),
            decode_batch_size=args.decode_batch_size,
            eval_workers=args.eval_workers,
            eval_chunksize=args.eval_chunksize,
            reference_max_samples=args.reference_max_samples,
            assume_train_canonical=(True if args.assume_train_canonical else None),
            show_progress=not args.no_tqdm,
            device=args.device,
            ao_refine_steps=args.ao_refine_steps,
            ao_refine_top_spans=args.ao_refine_top_spans,
            ao_refine_max_span_len=args.ao_refine_max_span_len,
            ao_refine_samples_per_span=args.ao_refine_samples_per_span,
            ao_refine_temperature=args.ao_refine_temperature,
            ao_refine_top_k=args.ao_refine_top_k,
            ao_refine_min_gain=args.ao_refine_min_gain,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "build-scaffold":
        smiles = load_smiles_list(
            args.input,
            smiles_column=args.smiles_column,
            max_mols=args.max_mols,
            canonicalize=True,
            deduplicate=True,
            show_progress=not args.no_tqdm,
        )
        res = build_scaffold_completion_dataset(
            smiles_list=smiles,
            output_path=args.output,
            min_scaffold_heavy_atoms=args.min_scaffold_heavy_atoms,
            max_items=args.max_mols,
            show_progress=not args.no_tqdm,
            num_workers=args.workers,
            chunksize=args.chunksize,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "build-linker":
        smiles = load_smiles_list(
            args.input,
            smiles_column=args.smiles_column,
            max_mols=args.max_mols,
            canonicalize=True,
            deduplicate=True,
            show_progress=not args.no_tqdm,
        )
        res = build_linker_design_dataset(
            smiles_list=smiles,
            output_path=args.output,
            max_items=args.max_mols,
            show_progress=not args.no_tqdm,
            num_workers=args.workers,
            chunksize=args.chunksize,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "build-scaffold-morphing":
        smiles = load_smiles_list(
            args.input,
            smiles_column=args.smiles_column,
            max_mols=args.max_mols,
            canonicalize=True,
            deduplicate=True,
            show_progress=not args.no_tqdm,
        )
        res = build_scaffold_morphing_dataset(
            smiles_list=smiles,
            output_path=args.output,
            max_items=args.max_mols,
            show_progress=not args.no_tqdm,
            num_workers=args.workers,
            chunksize=args.chunksize,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "build-linker-infill":
        smiles = load_smiles_list(
            args.input,
            smiles_column=args.smiles_column,
            max_mols=args.max_mols,
            canonicalize=True,
            deduplicate=True,
            show_progress=not args.no_tqdm,
        )
        res = build_linker_infill_dataset(
            smiles_list=smiles,
            output_path=args.output,
            representation=args.representation,
            stamp_vocab=args.stamp_vocab,
            max_items=args.max_mols,
            show_progress=not args.no_tqdm,
            num_workers=args.workers,
            chunksize=args.chunksize,
            ring_only=False,
            task_name="linker_design",
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "build-scaffold-morphing-infill":
        smiles = load_smiles_list(
            args.input,
            smiles_column=args.smiles_column,
            max_mols=args.max_mols,
            canonicalize=True,
            deduplicate=True,
            show_progress=not args.no_tqdm,
        )
        res = build_linker_infill_dataset(
            smiles_list=smiles,
            output_path=args.output,
            representation=args.representation,
            stamp_vocab=args.stamp_vocab,
            max_items=args.max_mols,
            show_progress=not args.no_tqdm,
            num_workers=args.workers,
            chunksize=args.chunksize,
            ring_only=True,
            task_name="scaffold_morphing",
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "build-motif-extension":
        smiles = load_smiles_list(
            args.input,
            smiles_column=args.smiles_column,
            max_mols=args.max_mols,
            canonicalize=True,
            deduplicate=True,
            show_progress=not args.no_tqdm,
        )
        res = build_motif_extension_dataset(
            smiles_list=smiles,
            output_path=args.output,
            representation=args.representation,
            stamp_vocab=args.stamp_vocab,
            max_items=args.max_mols,
            seed=args.seed,
            show_progress=not args.no_tqdm,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "build-scaffold-decoration":
        smiles = load_smiles_list(
            args.input,
            smiles_column=args.smiles_column,
            max_mols=args.max_mols,
            canonicalize=True,
            deduplicate=True,
            show_progress=not args.no_tqdm,
        )
        res = build_scaffold_decoration_dataset(
            smiles_list=smiles,
            output_path=args.output,
            representation=args.representation,
            stamp_vocab=args.stamp_vocab,
            min_scaffold_heavy_atoms=args.min_scaffold_heavy_atoms,
            max_items=args.max_mols,
            show_progress=not args.no_tqdm,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "build-superstructure-generation":
        smiles = load_smiles_list(
            args.input,
            smiles_column=args.smiles_column,
            max_mols=args.max_mols,
            canonicalize=True,
            deduplicate=True,
            show_progress=not args.no_tqdm,
        )
        res = build_superstructure_generation_dataset(
            smiles_list=smiles,
            output_path=args.output,
            representation=args.representation,
            stamp_vocab=args.stamp_vocab,
            max_items=args.max_mols,
            min_keep_groups=args.min_keep_groups,
            max_keep_groups=args.max_keep_groups,
            seed=args.seed,
            show_progress=not args.no_tqdm,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "build-fragment-infill":
        smiles = load_smiles_list(
            args.input,
            smiles_column=args.smiles_column,
            max_mols=args.max_mols,
            canonicalize=True,
            deduplicate=True,
            show_progress=not args.no_tqdm,
        )
        res = build_fragment_infill_dataset(
            smiles_list=smiles,
            output_path=args.output,
            representation=args.representation,
            stamp_vocab=args.stamp_vocab,
            min_mask_groups=args.min_mask_groups,
            max_mask_groups=args.max_mask_groups,
            max_items=args.max_mols,
            seed=args.seed,
            show_progress=not args.no_tqdm,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "train-completion":
        records = _load_csv_records(args.csv)
        cfg = _cfg_from_args(args)
        res = train_completion_model(
            records=records,
            prompt_key=args.prompt_key,
            target_key=args.target_key,
            representation=args.representation,
            output_dir=_branch_output_from_vocab(args.output_dir, args.stamp_vocab),
            prompt_template=args.prompt_template,
            target_template=args.target_template,
            cfg=cfg,
            show_progress=not args.no_tqdm,
            device=args.device,
            encode_workers=args.encode_workers,
            encode_chunksize=args.encode_chunksize,
            stamp_vocab=args.stamp_vocab,
            prompt_is_repr=bool(args.prompt_is_repr),
            target_is_repr=bool(args.target_is_repr),
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "train-ao-fragment-infill":
        records = _load_csv_records(args.csv)
        cfg = _cfg_from_args(args)
        cfg.model_type = "ao_gpt"
        res = train_ao_fragment_infill(
            records=records,
            representation=args.representation,
            output_dir=_branch_output_from_vocab(args.output_dir, args.stamp_vocab),
            cfg=cfg,
            show_progress=not args.no_tqdm,
            device=args.device,
            stamp_vocab=args.stamp_vocab,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "eval-fragment-infill":
        res = evaluate_fragment_infill(
            model_dir=args.model_dir,
            infill_csv=args.csv,
            output_dir=_branch_output_from_model(args.output_dir, args.model_dir),
            max_eval=args.max_eval,
            n_samples_per_prompt=args.n_samples_per_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            show_progress=not args.no_tqdm,
            device=args.device,
            eval_workers=args.eval_workers,
            eval_chunksize=args.eval_chunksize,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "eval-scaffold":
        res = evaluate_scaffold_completion(
            model_dir=args.model_dir,
            scaffold_csv=args.csv,
            output_dir=_branch_output_from_model(args.output_dir, args.model_dir),
            max_eval=args.max_eval,
            n_samples_per_prompt=args.n_samples_per_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            show_progress=not args.no_tqdm,
            device=args.device,
            eval_workers=args.eval_workers,
            eval_chunksize=args.eval_chunksize,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "eval-linker":
        res = evaluate_linker_design(
            model_dir=args.model_dir,
            linker_csv=args.csv,
            output_dir=_branch_output_from_model(args.output_dir, args.model_dir),
            max_eval=args.max_eval,
            n_samples_per_prompt=args.n_samples_per_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            show_progress=not args.no_tqdm,
            device=args.device,
            eval_workers=args.eval_workers,
            eval_chunksize=args.eval_chunksize,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "run-low-resource":
        reps = [x.strip() for x in args.representations.split(",") if x.strip()]
        sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
        res = run_low_resource_experiments(
            split_dir=args.split_dir,
            output_dir=args.output_dir,
            representations=reps,
            sizes=sizes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            n_samples=args.n_samples,
            show_progress=not args.no_tqdm,
            device=args.device,
            decode_batch_size=args.decode_batch_size,
            encode_workers=args.encode_workers,
            encode_chunksize=args.encode_chunksize,
            eval_workers=args.eval_workers,
            eval_chunksize=args.eval_chunksize,
            num_data_workers=args.num_data_workers,
            prefetch_factor=args.prefetch_factor,
            amp=args.amp,
            torch_compile=bool(args.torch_compile),
            tf32=not bool(args.no_tf32),
            model_type=args.model_type,
            diffusion_steps=args.diffusion_steps,
            diffusion_time_epsilon=args.diffusion_time_epsilon,
            diffusion_scheduler=args.diffusion_scheduler,
            diffusion_loss_weight=args.diffusion_loss_weight,
            diffusion_remasking=args.diffusion_remasking,
            diffusion_stochastic_transfer=bool(args.diffusion_stochastic_transfer),
            diffusion_block_size=args.diffusion_block_size,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "make-tables":
        save_experiment_markdown_tables(
            representation_stats_json=args.repr_stats_json,
            denovo_metrics_jsons=args.denovo_metrics_json,
            output_md=args.output_md,
        )
        print(f"Saved tables: {args.output_md}")
        return 0

    if args.cmd == "make-plots":
        res = save_benchmark_plots(
            run_root=args.run_root,
            output_dir=args.output_dir,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "property-pred":
        reps = [x.strip() for x in args.representations.split(",") if x.strip()]
        res = run_property_prediction_benchmark(
            input_path=args.input,
            output_dir=args.output_dir,
            target_column=args.target_column,
            representations=reps,
            task_type=args.task_type,
            smiles_column=args.smiles_column,
            test_size=args.test_size,
            seed=args.seed,
            max_mols=args.max_mols,
            n_features=args.n_features,
            ridge_alpha=args.ridge_alpha,
            deduplicate=not bool(args.no_deduplicate),
            show_progress=not args.no_tqdm,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "pretrain-roberta":
        train = _load_smiles_txt(args.train)
        valid = _load_smiles_txt(args.valid)
        cfg = RoBERTaConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            max_seq_len=args.max_seq_len,
            mask_prob=args.mask_prob,
            type_aware_masking=bool(getattr(args, "type_aware_masking", False)),
            frag_mask_prob=float(getattr(args, "frag_mask_prob", 0.20)),
            topo_mask_prob=float(getattr(args, "topo_mask_prob", 0.05)),
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            pretrain_epochs=args.pretrain_epochs,
            seed=args.seed,
            num_data_workers=args.num_data_workers,
            prefetch_factor=args.prefetch_factor,
            amp=args.amp,
            torch_compile=bool(args.torch_compile),
            tf32=not bool(args.no_tf32),
        )
        res = pretrain_roberta(
            train_smiles=train,
            valid_smiles=valid,
            representation=args.representation,
            output_dir=args.output_dir,
            cfg=cfg,
            show_progress=not args.no_tqdm,
            device=args.device,
            encode_workers=args.encode_workers,
            encode_chunksize=args.encode_chunksize,
            stamp_vocab=getattr(args, "stamp_vocab", None),
            stamp_force_ring_aware_encode=bool(getattr(args, "stamp_force_ring_aware_encode", False)),
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "finetune-roberta":
        cfg = RoBERTaConfig(
            finetune_epochs=args.finetune_epochs,
            finetune_lr=args.finetune_lr,
            finetune_batch_size=args.finetune_batch_size,
            dropout=args.dropout,
            amp=args.amp,
            tf32=not bool(args.no_tf32),
            num_data_workers=args.num_data_workers,
            prefetch_factor=args.prefetch_factor,
        )
        res = finetune_roberta(
            pretrained_dir=args.pretrained_dir,
            task_name=args.task,
            output_dir=args.output_dir,
            cfg=cfg,
            data_dir=args.data_dir,
            split_seed=args.split_seed,
            show_progress=not args.no_tqdm,
            device=args.device,
            encode_workers=args.encode_workers,
            encode_chunksize=args.encode_chunksize,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "run-roberta-benchmark":
        train = _load_smiles_txt(args.train)
        valid = _load_smiles_txt(args.valid)
        reps = [x.strip() for x in args.representations.split(",") if x.strip()]
        tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
        cfg = RoBERTaConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            max_seq_len=args.max_seq_len,
            mask_prob=args.mask_prob,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=args.finetune_epochs,
            finetune_lr=args.finetune_lr,
            finetune_batch_size=args.finetune_batch_size,
            seed=args.seed,
            num_data_workers=args.num_data_workers,
            prefetch_factor=args.prefetch_factor,
            amp=args.amp,
            torch_compile=bool(args.torch_compile),
            tf32=not bool(args.no_tf32),
        )
        res = run_roberta_property_benchmark(
            train_smiles=train,
            valid_smiles=valid,
            representations=reps,
            output_dir=args.output_dir,
            tasks=tasks,
            cfg=cfg,
            data_dir=args.data_dir,
            split_seed=args.split_seed,
            show_progress=not args.no_tqdm,
            device=args.device,
            encode_workers=args.encode_workers,
            encode_chunksize=args.encode_chunksize,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
