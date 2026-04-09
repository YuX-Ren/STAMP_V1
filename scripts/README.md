# Scripts Layout

## Real Entrypoints

- `scripts/core/`
  - `smiles_to_stamp.py`
  - `stamp_to_smiles.py`
  - `validate_roundtrip.py`
- `scripts/vocab/`
  - `build_vocab.py`
  - `vocab_stamp_roundtrip_validate.py`
- `scripts/benchmark/`
  - `experiments_cli.py`

## Compatibility Wrappers

Root-level files in `scripts/*.py` are thin wrappers that forward to the new subdirectory entrypoints, so existing commands keep working.
