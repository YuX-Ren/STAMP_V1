# STAMP

STAMP (**Sequential Tree-based Attachment Motif Presentation**) is a molecular
sequence representation for fragment-level generation and editing. It encodes a
molecule with canonical motif tokens plus explicit joint and branch tokens, and
uses a stack-machine decoder to reconstruct molecular graphs.

This repository contains the codec, vocabulary tooling, benchmark models, data
preparation utilities, and HPC launch scripts used by the STAMP research
workflows.

## System requirements

The CPU codec demo needs Python 3.12, RDKit 2023.09.6 and NumPy 1.26.4.
All required demo dependencies are pinned in
[`envs/requirements-demo.txt`](envs/requirements-demo.txt). It uses the Python
standard library for file I/O; `tqdm` is optional. A normal CPU computer with
2 GB available RAM is sufficient for this small demo; no GPU is required.

The demo was tested on Red Hat Enterprise Linux 9.8, x86-64, Python 3.12.12,
RDKit 2023.09.6 and NumPy 1.26.4 on an Intel Xeon Gold 6438N CPU.
Windows and macOS have not been tested. The included GitHub Actions workflow
targets Ubuntu 22.04 and Python 3.12; a passing hosted run is separate from the
local test reported here.

Research training uses PyTorch 2.6.0 with CUDA 12.4 and the environment in
[`envs/stamp-train-rdkit-2025.09.4.yml`](envs/stamp-train-rdkit-2025.09.4.yml).
That recipe targets RDKit 2025.09.4; the local demo environment above uses
2023.09.6. Keep the RDKit version associated with each vocabulary/checkpoint
when reproducing research outputs, since canonical atom ordering affects tokens.
Large training runs use NVIDIA H200 GPUs (one for encoder training; four for
the documented MolT5 pretraining run). These GPUs are not needed for the demo.
Workflow-specific optional dependencies are listed in
[the reproducibility notes](docs/reproducibility.md).

## Installation

For the self-contained CPU demo, start from the source repository:

```bash
git clone https://github.com/YuX-Ren/STAMP_V1.git
cd STAMP_V1
python3.12 -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install -r envs/requirements-demo.txt
```

Run commands from the checkout root; installing STAMP as a package is not
required. Allow approximately 2–10 minutes for installation on a normal
desktop with Python already installed and broadband access (an estimate;
package downloads and network speed dominate).

For GPU research workflows, create the separate training environment:

```bash
conda env create -f envs/stamp-train-rdkit-2025.09.4.yml
conda activate stamp-train-rdkit-2025_09_4
```

The full training environment can take approximately 10–30 minutes to install
(estimate). The CPU demo does not require this environment.

## Demo

Run the ten-molecule example bundled in this checkout:

```bash
python scripts/core/run_codec_demo.py
diff -u examples/codec/expected_roundtrips.tsv results/demo/roundtrips.tsv
```

Expected terminal output is `SUMMARY` followed by the tab-separated fields
`passed=10`, `total=10`, and `failed=0`. The output file
`results/demo/roundtrips.tsv` records SMILES, STAMP, recovered canonical SMILES
and `OK` for each molecule. `diff` produces no output on an exact match.
The measured local runtime was 0.65 seconds including interpreter startup;
allow about 1–5 seconds on a normal desktop. No private study assets are needed.
Use a new `--output` path when rerunning; existing output files are protected.
See [demo data and expected results](examples/codec/README.md).

## Instructions for use

Encode, decode, and validate individual molecules:

```bash
python scripts/smiles_to_stamp.py "CCO"
python scripts/stamp_to_smiles.py "C [J_0] [END]"
python scripts/validate_roundtrip.py "CCO"
```

Process your own file, containing one SMILES per line without a header:

```bash
python scripts/core/run_codec_demo.py --input my_molecules.txt \
  --output results/my_roundtrips.tsv
```

Build a motif vocabulary from your own CSV (with a `smiles` column):

```bash
python scripts/build_vocab.py my_molecules.csv \
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

The matching study data snapshot is hosted separately in the private
[`bi0s/STAMP_V1-data`](https://huggingface.co/datasets/bi0s/STAMP_V1-data)
Hugging Face dataset. Download it and copy `main/data/` into this checkout's
`data/` directory when your account has access. Anonymous access to the dataset
API returned HTTP 401 on 19 November 2026. The bundled demo is public source
content; exact study splits still require dataset access. Ten selected study
checkpoints and companion vocabularies/configurations are publicly available
at [bi0s/STAMP_V1](https://huggingface.co/bi0s/STAMP_V1).
See [data and model access](docs/data_and_model_access.md) for download instructions.

Optional representation implementations can be installed as packages or cloned
under `third_party/`. See [the experiment guide](docs/experiments.md) and
[the runbook](docs/experiments_runbook.md) for the expected layout and commands.
Optional packages, pinned third-party revisions, and cluster-path conventions are
listed in [the reproducibility notes](docs/reproducibility.md).

## Documentation

- [Software and machine learning checklist mapping](docs/release_checklist.md)
- [Study model card](MODEL_CARD.md)
- [Data, splits and checkpoint access](docs/data_and_model_access.md)
- [Local checkpoint inventory](docs/checkpoint_inventory.md)
- [Code functionality and pseudocode](docs/software_functionality.md)
- [STAMP representation specification](docs/stamp.md)
- [Implementation and vocabulary notes](docs/implement.md)
- [Benchmark experiments](docs/experiments.md)
- [Experiment runbook](docs/experiments_runbook.md)
- [STAMP v2 design](docs/stamp_v2_design.md)
- [PS-VAE integration](docs/psvae_integration.md)

For development conventions, see [AGENTS.md](AGENTS.md).

## License

STAMP source code and bundled demo files are distributed under the
[MIT License](LICENSE). Third-party code, upstream models and datasets remain
subject to their respective licenses and access terms.
