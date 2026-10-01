# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STAMP (Sequential Tree-based Attachment Motif Presentation) is a molecular representation framework for LLM-guided molecule generation. It encodes molecules as token sequences using a three-token grammar (fragment, branch, joint tokens) with a stack-machine decoder that guarantees 100% sequence validity.

## Environment

Conda env spec: `envs/stamp-train-rdkit-2025.09.4.yml` (Python 3.12, RDKit 2025.09.4, PyTorch 2.6 + CUDA 12.4).

## Common Commands

### Core codec operations
```bash
python scripts/smiles_to_stamp.py "CCO"
python scripts/stamp_to_smiles.py "C [J_0] [END]"
python scripts/validate_roundtrip.py "CCO"
```

### Vocabulary building
```bash
python scripts/build_vocab.py data/delaney-processed.csv --smiles-column smiles --output-dir results/vocab/stamp_vocab_delaney
python scripts/vocab_stamp_roundtrip_validate.py --input data/zinc250k.csv --vocab-dir results/vocab/stamp_vocab_zinc250k --smiles-column smiles --out-prefix results/roundtrip/zinc250k/vocab/v2
```

### Benchmark experiments
All experiments run through a single CLI; `--help` lists all subcommands. Commonly used:
```bash
python scripts/experiments_cli.py --help
python scripts/experiments_cli.py prepare-data ...
python scripts/experiments_cli.py repr-stats ...         # Exp A: representation statistics
python scripts/experiments_cli.py train-lm ...           # Exp B: de novo LM (supports --model-type {ar,diffusion})
python scripts/experiments_cli.py sample-eval ...        # Exp B: sample & evaluate
python scripts/experiments_cli.py build-scaffold ...     # Exp C: scaffold completion
python scripts/experiments_cli.py train-completion ...   # Exp C/D: train completion model
python scripts/experiments_cli.py eval-scaffold ...      # Exp C: evaluate
python scripts/experiments_cli.py pretrain-roberta ...   # Exp E: RoBERTa MLM pretraining
python scripts/experiments_cli.py pretrain-roberta-mtr ...  # Exp E: RoBERTa multi-task-regression (descriptor) pretraining
python scripts/experiments_cli.py finetune-roberta ...   # Exp E: fine-tune on MoleculeNet (classification + regression tasks)
python scripts/experiments_cli.py property-pred ...      # Exp F: property prediction
python scripts/experiments_cli.py mine-vocab[-ring] ...  # PS-VAE principal-subgraph vocab (ring-aware variant)
# Infill-family tasks: build-linker, build-scaffold-morphing, build-linker-infill,
# build-scaffold-morphing-infill, build-motif-extension, build-scaffold-decoration,
# build-superstructure-generation, build-fragment-infill, train-ao-fragment-infill,
# eval-fragment-infill, eval-linker, run-low-resource, make-tables, make-plots.
```
Direct scripts (not wrappers) for newer workflows live under `scripts/benchmark/`:
`train_ao_gpt_parquet.py`, `finetune_ao_gpt_pp.py`, `eval_ao_gpt_pp_linear.py` (AO-GPT);
`train_stamp_diffusion.py`, `eval_stamp_diffusion.py` (diffusion LM);
`run_mcts_guacamol.py`, `run_mcts_unified_guacamol.py` (MCTS search);
`psvae_roundtrip_test.py`, `slice_psvae_vocab.py` (PS-VAE integration).

### Large-scale sharded pretraining (parquet pipeline)
For 30M-scale corpora the pipeline is: download → encode-to-STAMP shards → train on parquet.
- Data prep: `download_pubchem_shards.py`, `download_molformer_box_parallel.py`,
  `encode_smiles_to_stamp_sharded.py` (+ `encode_one_smiles_shard.py`, multiscale variant),
  `decontaminate_pretrain.py`, `prepare_pubchem_pretrain_splits.py`.
- Decoder-only / NTP pretraining over sharded parquet:
  `train_ao_gpt_parquet.py` (any-order GPT), `train_cams_decoder_shards.py` (CamS-aligned NTP),
  `train_stamp_eso_parquet.py` / `train_stamp_eso_v2_parquet.py` / `train_stamp_eso_hybrid_parquet.py` (Eso-LM, arXiv 2506.01928),
  `train_stamp_hybrid_parquet.py` (hybrid motif+char vocab), `train_stamp_diffusion_parquet.py`.
- MLM-style encoders: `train_molformer_stamp_shards.py` (MolFormer-style), plus `finetune_molformer_stamp_pp.py`,
  `finetune_cams_decoder_pp.py`, and `finetune_moleculeace.py` / `finetune_moleculeace_chemberta.py` (MoleculeACE benchmark).
- Large checkpoints/data live on scratch, not in-repo — convention: `/scratch/<user>/data/ckpt_stamp/<run_name>/eval_epoch_NNNN`.

### PMO / oracle-guided generation (TDC)
Property-Molecule-Optimization baselines over TDC oracles (DRD2, GSK3β, JNK3, etc.):
- `build_stamp_pmo_bank.py` — offline STAMP motif/subtree bank for PMO tasks.
- `run_genmol_attaching_pmo.py` (GenMol attaching-only baseline), `run_stamp_attaching_pmo.py` (GenMol-aligned STAMP),
  `run_stamp_motif_planner_pmo.py` (lightweight STAMP motif-planner).
- Oracle prediction caches are pickled under `oracle/` (e.g. `drd2_current.pkl`); the faithful GenMol reference impl lives under `isolated/pmo-genmol-faithful/`.
- See `docs/pmo_project_storage_bundle.md` (reproduction bundle), `docs/genmol_task_mapping.md`, `docs/phase2_test_time_methods.md`.

See `docs/experiments.md` for usage, `docs/experiments_runbook.md` / `experiments_checklist.md` for full runbooks,
`docs/ao_gpt_improvement.md`, `docs/diffusion_vs_ar_experiments.md`, `docs/psvae_integration.md`,
`docs/stamp_v2_design.md`, `docs/stamp_v2_vocab_design.md` for design notes on newer components.

### HPC job submission
```bash
qsub scripts/hpc/roberta_benchmark.pbs          # RoBERTa pretraining + fine-tuning
qsub scripts/hpc/train_stamp_bert.pbs           # STAMP-tokenized BERT training
qsub scripts/hpc/expc_eval_full_t07_k1.pbs      # Exp C full eval
bash  scripts/hpc/train_stamp_diffusion_datasets.sh           # diffusion LM sweep (driver)
sbatch scripts/hpc/train_ao_gpt_parquet_ddp.slurm             # AO-GPT DDP (SLURM)
```

## Architecture

```
scripts/experiments_cli.py          ← main CLI entry point
  ↓
stamp/benchmark/                    ← experiment modules
  ├── representations.py            ← BaseRepresentation interface (SMILES, STAMP, SELFIES, SAFE, etc.)
  ├── lm.py                         ← TinyDecoderLM transformer (AR + diffusion) + training loop
  ├── roberta_property.py           ← RoBERTa MLM pretraining + MoleculeNet fine-tuning
  ├── tasks.py                      ← scaffold completion, linker design, infill-family tasks
  ├── denovo.py                     ← de novo generation evaluation
  ├── property_prediction.py        ← linear classifier on molecule representations
  ├── data.py                       ← dataset splits & low-resource subsets
  ├── low_resource.py, representation_stats.py, common.py
  ├── mcts_guacamol.py              ← Guacamol MCTS search over STAMP tokens
  ├── mcts_unified/                 ← unified MCTS: policy, edit_policy, subtree pool, port signatures, transplant
  └── metrics.py, reports.py, plots.py
  ↓
stamp/core/                         ← STAMP codec (independent of vocab & benchmarks)
  ├── stamp_codec.py                ← STAMPEncoder: SMILES↔STAMP encoding/decoding via stack machine
  ├── reconstruction.py             ← STAMP→RDKit molecule reconstruction
  └── mol_normalize.py              ← molecule sanitization & canonicalization
  ↓
stamp/vocab/                        ← offline vocabulary construction
  ├── vocab_builder.py              ← StampVocabBuilder: two-level vocab (core motifs + attachment schemas)
  ├── candidate_generation.py, branch_paths.py, canonicalize.py, _chem_utils.py
  ├── principal_subgraph.py         ← PS-VAE principal-subgraph vocab mining (ring-aware variant supported)
  ├── stamp_vocab_types.py
  └── statistics.py, tree_metrics.py
```

- `stamp/experiments.py` is a compatibility facade that re-exports from `stamp/benchmark/`.
- `scripts/*.py` at root level are thin wrappers delegating to `scripts/core/`, `scripts/vocab/`, or `scripts/benchmark/`; real entrypoints live in those subdirectories (see `scripts/README.md`).
- `mcts_protype.py` at repo root is a standalone prototype — not on the benchmark path.
- Outputs conventionally land under `results/` (`results/roundtrip/…`, `results/vocab/…`, `results/benchmark/…`); data under `data/`.

## Development Conventions

- **No central test suite.** Treat `validate_roundtrip.py` and targeted script runs as smoke tests for codec/vocab changes. After edits, syntax-check with `python -m py_compile <changed files>`.
- **Don't commit generated artifacts.** Scheduler logs (`*.o<jobid>` at repo root), caches, and large `results/`/`data/` outputs are throwaway; `.gitignore` already excludes `results/`, `data/`, `logs/`, `*.csv`, `*.smiles`. Don't overwrite untracked experiment outputs or user-edited files.
- **`AGENTS.md`** at repo root carries the same project conventions (structure, style, commit/PR guidance) — keep it consistent with this file when either changes.

## Key Design Points

- **Unified representation interface**: All molecular representations (SMILES, STAMP, SELFIES, SAFE, Group-SELFIES) implement `BaseRepresentation` with `.encode()`, `.decode()`, `.tokenize()`, `.detokenize()` for fair benchmarking.
- **Stack-machine validity**: The decoder uses R4 fallback rules (see `docs/stamp.md`) to guarantee valid molecules from any token sequence.
- **Multiprocessing**: Encoding, vocabulary building, and metric computation use `ProcessPoolExecutor` / multiprocessing pools. Set `TOKENIZERS_PARALLELISM=false` to avoid OpenMP conflicts.
- **Third-party representations** live under `third_party/` (safe, group-selfies, chemicalgof) and are imported by the benchmark layer.

## Key Dependencies

- **RDKit**: molecule I/O, SMILES canonicalization, BRICS fragmentation, Murcko scaffolds
- **PyTorch**: transformer models (TinyDecoderLM, RoBERTa), supports mixed-precision (BF16/FP16)
- **NumPy**, **pandas**, **tqdm**

## Documentation

- `docs/stamp.md` — full STAMP encoding specification (grammar, stack machine, safety fallbacks)
- `docs/implement.md` — motif vocabulary construction spec
- `docs/experiments.md` — benchmark experiment guide with examples
- `docs/experiments_runbook.md`, `docs/experiments_checklist.md` — step-by-step execution guides
- `docs/stamp_v2_design.md`, `docs/stamp_v2_vocab_design.md` — v2 codec & vocab design
- `docs/ao_gpt_improvement.md` — any-order GPT training notes
- `docs/diffusion_vs_ar_experiments.md` — diffusion vs. autoregressive comparisons
- `docs/psvae_integration.md` — PS-VAE principal-subgraph integration
- `docs/genmol_task_mapping.md`, `docs/stamp_safe_task_design.md` — task design notes
- `docs/pmo_project_storage_bundle.md` — PMO reproduction bundle (faithful GenMol + STAMP/hybrid PMO)
- `docs/genmol_aonly_repro_summary.md`, `docs/genmol_aonly_repro_findings.md`, `docs/genmol_A_repro_matrix.md` — GenMol attaching-only reproduction
- `docs/phase2_test_time_methods.md` — STAMP+AO test-time generation methods (results summary)
- `docs/stamp_training_env.md` — training environment reference
