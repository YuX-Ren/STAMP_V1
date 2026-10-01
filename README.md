# STAMP

STAMP (**Sequential Tree-based Attachment Motif Presentation**) is a molecular
sequence representation for fragment-level generation and editing. It encodes a
molecule with canonical motif tokens plus explicit joint and branch tokens, and
uses a stack-machine decoder to reconstruct molecular graphs.

This repository contains the codec, vocabulary tooling, benchmark models, data
preparation utilities, and HPC launch scripts used by the STAMP research
workflows.

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

Build a motif vocabulary:

```bash
python scripts/build_vocab.py data/delaney-processed.csv \
  --smiles-column smiles \
  --output-dir results/vocab/stamp_vocab_delaney
```

Inspect the benchmark CLI:

```bash
python scripts/experiments_cli.py --help
```

## Repository layout

- `stamp/core/`: SMILES/STAMP codec, normalization, and reconstruction.
- `stamp/vocab/`: motif mining, canonicalization, scoring, and vocabulary tools.
- `stamp/benchmark/`: representations, language models, property models, tasks,
  metrics, and analysis utilities.
- `scripts/core/`, `scripts/vocab/`, `scripts/benchmark/`: primary command-line
  implementations. Root-level scripts are compatibility wrappers.
- `scripts/hpc/`: PBS, Slurm, and shell launch recipes for research runs.
- `docs/`: representation specifications, experiment guides, runbooks, and
  result notes.
- `envs/`: reproducible environment specifications.

The `main` branch contains the general STAMP training and generation workflows.
The `stamp-v1.5pp` branch preserves the parallel multiscale/CAMS and
MolFormer-STAMP experiments from the `STAMP_V1.5_pp` workspace.

## Data and external dependencies

Datasets, checkpoints, scheduler logs, and generated results are intentionally
not versioned. Keep them under `data/`, `results/`, or another external storage
location. Many HPC scripts contain site-specific defaults; override their
environment variables and paths for a different cluster.

The matching data snapshot is hosted separately in the private
[`bi0s/STAMP_V1-data`](https://huggingface.co/datasets/bi0s/STAMP_V1-data)
Hugging Face dataset. Download it and copy `main/data/` into this checkout's
`data/` directory; the dataset card includes provenance and SHA-256 checksums.

Optional representation implementations can be installed as packages or cloned
under `third_party/`. See [the experiment guide](docs/experiments.md) and
[the runbook](docs/experiments_runbook.md) for the expected layout and commands.
Optional packages, pinned third-party revisions, and cluster-path conventions are
listed in [the reproducibility notes](docs/reproducibility.md).

## Documentation

- [STAMP representation specification](docs/stamp.md)
- [Implementation and vocabulary notes](docs/implement.md)
- [Benchmark experiments](docs/experiments.md)
- [Experiment runbook](docs/experiments_runbook.md)
- [STAMP v2 design](docs/stamp_v2_design.md)
- [PS-VAE integration](docs/psvae_integration.md)

For development conventions, see [AGENTS.md](AGENTS.md).

## License

No top-level software license has been selected for this repository. Third-party
projects are not vendored and remain subject to their own licenses.
