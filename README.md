# STAMP Workspace Layout

## Directories

- `stamp/`: Python package (now structured by domain):
  - `stamp/core/`: codec / reconstruction / normalization.
  - `stamp/vocab/`: motif candidate generation / scoring / vocab building.
  - `stamp/benchmark/`: experiment modules (data, representation stats, LM, tasks, low-resource, reports).
  - `stamp/experiments.py`: benchmark compatibility facade (re-exports from `stamp/benchmark/`).
- `scripts/`: CLI entry points (now structured by domain):
  - `scripts/core/`: codec/roundtrip CLIs.
  - `scripts/vocab/`: vocabulary build/validation CLIs.
  - `scripts/benchmark/`: experiment CLI.
  - `scripts/*.py` at root are compatibility wrappers for old command paths.
- `docs/`: design and implementation notes.
- `data/`: datasets.
- `results/`: generated outputs (organized by `roundtrip/` and `vocab/`).

## Common Commands

- SMILES -> STAMP:
  - `python scripts/smiles_to_stamp.py "CCO"`
- STAMP -> SMILES:
  - `python scripts/stamp_to_smiles.py "C [J_0] [END]"`
- Roundtrip validation:
  - `python scripts/validate_roundtrip.py "CCO"`
- Build vocabulary:
  - `python scripts/build_vocab.py data/delaney-processed.csv --smiles-column smiles --output-dir results/vocab/stamp_vocab_delaney`
- Vocab-based roundtrip check:
  - `python scripts/vocab_stamp_roundtrip_validate.py --input data/zinc250k.csv --vocab-dir results/vocab/stamp_vocab_zinc250k --smiles-column smiles --out-prefix results/roundtrip/zinc250k/vocab/v2`
- Run benchmark experiments from checklist (A-E):
  - `python scripts/experiments_cli.py --help`
  - detailed usage: `docs/experiments.md`
