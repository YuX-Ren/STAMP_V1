# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STAMP (Sequential Tree-based Attachment Motif Presentation) is a molecular representation framework for LLM-guided molecule generation. It encodes molecules as token sequences using a three-token grammar (fragment, branch, joint tokens) with a stack-machine decoder that guarantees 100% sequence validity.

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

### Benchmark experiments (A-E)
All experiments run through a single CLI with subcommands:
```bash
python scripts/experiments_cli.py --help
python scripts/experiments_cli.py prepare-data ...
python scripts/experiments_cli.py repr-stats ...        # Exp A: representation statistics
python scripts/experiments_cli.py train-lm ...          # Exp B: de novo generation
python scripts/experiments_cli.py sample-eval ...       # Exp B: sample & evaluate
python scripts/experiments_cli.py build-scaffold ...    # Exp C: scaffold completion
python scripts/experiments_cli.py train-completion ...  # Exp C/D: train completion model
python scripts/experiments_cli.py eval-scaffold ...     # Exp C: evaluate
python scripts/experiments_cli.py pretrain-roberta ...  # Exp E: RoBERTa MLM pretraining
python scripts/experiments_cli.py finetune-roberta ...  # Exp E: fine-tune on MoleculeNet
```
See `docs/experiments.md` for detailed usage and `docs/experiments_checklist.md` for the full runbook.

### HPC (PBS) job submission
```bash
qsub scripts/hpc/roberta_benchmark.pbs  # RoBERTa pretraining + fine-tuning
```

## Architecture

```
scripts/experiments_cli.py          ← main CLI entry point
  ↓
stamp/benchmark/                    ← experiment modules
  ├── representations.py            ← BaseRepresentation interface (SMILES, STAMP, SELFIES, SAFE, etc.)
  ├── lm.py                         ← TinyDecoderLM transformer + training loop
  ├── roberta_property.py           ← RoBERTa MLM pretraining + MoleculeNet fine-tuning
  ├── tasks.py                      ← scaffold completion & linker design
  ├── denovo.py                     ← de novo generation evaluation
  ├── property_prediction.py        ← linear classifier on molecule representations
  ├── data.py                       ← dataset splits & low-resource subsets
  └── metrics.py, reports.py, plots.py
  ↓
stamp/core/                         ← STAMP codec (independent of vocab & benchmarks)
  ├── stamp_codec.py                ← STAMPEncoder: SMILES↔STAMP encoding/decoding via stack machine
  ├── reconstruction.py             ← STAMP→RDKit molecule reconstruction
  └── mol_normalize.py              ← molecule sanitization & canonicalization
  ↓
stamp/vocab/                        ← offline vocabulary construction
  ├── vocab_builder.py              ← StampVocabBuilder: two-level vocab (core motifs + attachment schemas)
  ├── candidate_generation.py       ← motif candidate extraction from fragmented molecules
  └── statistics.py, tree_metrics.py, canonicalize.py
```

- `stamp/experiments.py` is a compatibility facade that re-exports from `stamp/benchmark/`.
- `scripts/*.py` at root level are thin wrappers delegating to `scripts/core/`, `scripts/vocab/`, or `scripts/benchmark/`.

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
- `docs/experiments_runbook.md` — step-by-step execution guide
