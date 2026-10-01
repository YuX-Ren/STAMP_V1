# Repository Guidelines

## Project Structure & Module Organization

STAMP is a Python research codebase for molecular sequence representations. Core package code lives in `stamp/`: `stamp/core/` contains SMILES/STAMP codec and reconstruction logic, `stamp/vocab/` contains vocabulary mining and scoring, and `stamp/benchmark/` contains experiment, model, metric, and task code.

Command-line entry points live in `scripts/`. Prefer implementations in `scripts/core/`, `scripts/vocab/`, and `scripts/benchmark/`; root-level scripts are compatibility wrappers. HPC launch files live in `scripts/hpc/`. Design notes are in `docs/`, environment specs in `envs/`, and vendored external representations in `third_party/`. Generated data and experiment outputs should stay under `data/` and `results/`.

## Build, Test, and Development Commands

- `conda env create -f envs/stamp-train-rdkit-2025.09.4.yml`: create the Python 3.12/RDKit/PyTorch environment.
- `python scripts/smiles_to_stamp.py "CCO"`: encode a SMILES string.
- `python scripts/stamp_to_smiles.py "C [J_0] [END]"`: decode STAMP tokens.
- `python scripts/validate_roundtrip.py "CCO"`: smoke-test codec roundtripping.
- `python scripts/build_vocab.py data/delaney-processed.csv --smiles-column smiles --output-dir results/vocab/stamp_vocab_delaney`: build a vocabulary.
- `python scripts/experiments_cli.py --help`: inspect benchmark subcommands.

For syntax checks after edits, run `python -m py_compile <changed files>`.

## Coding Style & Naming Conventions

Use idiomatic Python with 4-space indentation and type hints where they clarify interfaces. Follow existing naming: modules and functions use `snake_case`, classes use `PascalCase`, constants use `UPPER_SNAKE_CASE`, and CLI options use kebab-case. Keep wrapper scripts thin; place real implementation in the appropriate domain subdirectory or `stamp/` module.

## Testing Guidelines

There is no centralized test suite yet. Treat roundtrip validation and targeted script execution as required smoke tests for codec or vocabulary changes. Add focused `test_*.py` files when introducing reusable behavior, with examples such as `CCO` plus ring/branch cases. For benchmark changes, prefer tiny input files and outputs under `results/`.

## Commit & Pull Request Guidelines

Git history uses short, imperative, lower-case summaries, for example `add diffusion llm` or `update ao-gpt...`. Keep commits scoped and avoid committing generated logs, `.o*` scheduler outputs, caches, or large result artifacts. Pull requests should describe the affected subsystem, list commands run, mention data or checkpoint assumptions, and link relevant docs or issues.

## Agent-Specific Instructions

Do not overwrite untracked experiment outputs or user-edited files. Prefer `docs/experiments.md`, `docs/experiments_runbook.md`, and `CLAUDE.md` for workflow context before changing benchmark code. Set `TOKENIZERS_PARALLELISM=false` when multiprocessing tokenization conflicts with tokenizer/OpenMP parallelism.
