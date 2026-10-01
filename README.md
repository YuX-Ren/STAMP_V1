# STAMP

STAMP (**Sequential Tree-based Attachment Motif Presentation**) is a molecular
sequence representation for fragment-level generation and editing. It encodes a
molecule with canonical motif tokens plus explicit joint and branch tokens, and
uses a stack-machine decoder to reconstruct molecular graphs.

This branch preserves the parallel STAMP v1.5++ research workspace, including
multiscale STAMP encodings, CAMS decoder experiments, MolFormer-STAMP models,
and the associated MoleculeNet analysis workflows.

## Quick start

Create the core environment:

```bash
conda env create -f envs/stamp-train-rdkit-2025.09.4.yml
conda activate stamp-train-rdkit-2025_09_4
```

Run the core codec:

```bash
python scripts/smiles_to_stamp.py "CCO"
python scripts/stamp_to_smiles.py "C [J_0] [END]"
python scripts/validate_roundtrip.py "CCO"
```

Inspect the benchmark CLI:

```bash
python scripts/experiments_cli.py --help
```

## Branch-specific workflows

- `stamp/benchmark/multiscale_stamp.py`: multiscale STAMP view construction.
- `stamp/benchmark/molformer_stamp.py`: MolFormer-style STAMP encoders.
- `scripts/benchmark/train_cams_decoder_shards.py`: CAMS decoder pretraining.
- `scripts/benchmark/train_molformer_stamp_shards.py`: sharded MolFormer-STAMP
  pretraining.
- `scripts/benchmark/finetune_cams_decoder_pp.py` and
  `finetune_molformer_stamp_pp.py`: downstream property-prediction workflows.

The branch also archives the AO/ESO/hybrid, GenMol/PMO, and cluster launch
recipes used as comparison baselines during the same development cycle. These
are research records with site-specific defaults, not a single turnkey pipeline.

## Repository layout

- `stamp/core/`: SMILES/STAMP codec, normalization, and reconstruction.
- `stamp/vocab/`: motif mining, canonicalization, scoring, and vocabulary tools.
- `stamp/benchmark/`: representations, models, property tasks, and metrics.
- `scripts/benchmark/`: data preparation, training, evaluation, and analysis.
- `scripts/hpc/`: site-specific PBS and shell launch recipes.
- `docs/`: representation specifications, experiment guides, and runbooks.
- `envs/`: reproducible environment specifications.

The general STAMP workflows live on `main`; this branch is intentionally kept
separate so its multiscale and property-prediction changes remain reproducible.

## Data and external dependencies

Datasets, checkpoints, scheduler logs, generated figures, and result tables are
not versioned. Keep them under `data/`, `results/`, or external storage. HPC
scripts contain cluster-specific defaults and should be configured through their
environment variables before use.

Optional representations and reference implementations can be installed as
packages or cloned under `third_party/`. See
[the experiment guide](docs/experiments.md) and
[the runbook](docs/experiments_runbook.md) for details. Optional packages,
pinned third-party revisions, and cluster-path conventions are listed in
[the reproducibility notes](docs/reproducibility.md).

For development conventions, see [AGENTS.md](AGENTS.md).

## License

No top-level software license has been selected for this repository. Third-party
projects are not vendored and remain subject to their own licenses.
