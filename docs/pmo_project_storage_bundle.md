# PMO Reproduction Bundle

This bundle is the minimal organized snapshot needed to reproduce the PMO work
done in this repository, including:

- faithful GenMol-style PMO reproduction
- STAMP PMO bank construction
- STAMP/GenMol hybrid PMO experiments

## Included Code

- `stamp/`
- `scripts/`
- `envs/`
- `docs/`
- `oracle/`
- `mcts_protype.py`
- `README.md`
- `AGENTS.md`
- `CLAUDE.md`

## Included PMO-Specific Code Copies

- `isolated/pmo-genmol-faithful/`
  - faithful PMO baseline
  - `genmol_stamp`
  - `genmol_stamp_hybrid`
  - PMO launch scripts and logs/results subdirectories
- `third_party/genmol-pmo-clean/`
- `third_party/genmol-pmo-aonly/`

## Included Data

- `data/zinc250k.csv`

## Included Reproduction Artifacts

- `results/pmo_stamp_bank/`
  - STAMP PMO offline banks
- `results/pmo_fragment_vocab/`
  - rebuilt GenMol-style fragment vocabs
- `results/pmo_fragment_vocab_genmol_official/`
  - official GenMol PMO fragment vocabs copied into this repo
- `results/baselines/`
  - baseline comparisons, probes, logs, and reproduction evidence
- `isolated/pmo-genmol-faithful/results/pmo/`
  - faithful PMO task results
- `isolated/pmo-genmol-faithful/results/pmo_hybrid_full/`
  - full hybrid benchmark outputs collected so far
- `isolated/pmo-genmol-faithful/results/logs/`
- `isolated/pmo-genmol-faithful/results/logs_hybrid_full/`

## Excluded Artifacts

The storage copy should exclude transient files that are not needed for
reproduction:

- `__pycache__/`
- root-level scheduler outputs such as `*.o*`
- temporary files under `/tmp`
- unrelated training outputs not used by PMO

## Notes

- The faithful PMO benchmark that reached `sum AUC-10 = 17.7154` is under
  `isolated/pmo-genmol-faithful/results/pmo/`.
- The current hybrid full benchmark is still being collected under
  `isolated/pmo-genmol-faithful/results/pmo_hybrid_full/`.
- For reproducibility, prefer the isolated PMO copy instead of modifying the
  main repo entry points.
