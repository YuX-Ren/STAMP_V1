# STAMP

**Sequential Tree Attachment-based Motif-Preserving representation**: a
topology-aware molecular language for fragment-level learning and editing.

[Paper](https://arxiv.org/abs/2610.18438) ·
[Models](https://huggingface.co/bi0s/STAMP_V1) ·
[Data](https://huggingface.co/datasets/bi0s/STAMP_V1-data)

## Requirements and installation

The CPU demo was tested on RHEL 9.8 (x86-64), Python 3.12.12,
RDKit 2023.09.6 and NumPy 1.26.4. It requires no GPU.

```bash
git clone https://github.com/YuX-Ren/STAMP_V1.git
cd STAMP_V1
python3.12 -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install -r envs/requirements-demo.txt
```

Estimated installation time on a desktop with Python installed: 2–10 minutes.
For GPU training, the environment recipe specifies Python 3.12, RDKit
2025.09.4, PyTorch 2.6.0 and CUDA 12.4:

```bash
conda env create -f envs/stamp-train-rdkit-2025.09.4.yml
conda activate stamp-train-rdkit-2025_09_4
```

Keep the RDKit version matched to the vocabulary and checkpoint when
reproducing token sequences. Study training used NVIDIA H200 GPUs.

## Demo

Run the bundled ten-molecule dataset:

```bash
python scripts/core/run_codec_demo.py
diff -u examples/codec/expected_roundtrips.tsv results/demo/roundtrips.tsv
```

Expected output:

```text
SUMMARY passed=10 total=10 failed=0
```

Fields are tab-separated. The output TSV contains input SMILES, STAMP,
recovered SMILES and status. `diff` produces no output on an exact match.
Measured runtime: 0.65 seconds on an Intel Xeon Gold 6438N; allow 1–5 seconds
on a desktop. Use a new `--output` path to repeat the demo.

## Usage

Run from the checkout root:

```bash
python scripts/smiles_to_stamp.py "CCO"
python scripts/validate_roundtrip.py "CCO"
python scripts/core/run_codec_demo.py --input my_molecules.txt --output results/my_roundtrips.tsv
python scripts/build_vocab.py my_molecules.csv --smiles-column smiles --output-dir results/vocab/my_vocab
python scripts/experiments_cli.py --help
```

Text input uses one SMILES per line; CSV input uses a `smiles` column.

## Models and data

Download a model with its vocabulary and configuration, for example:

```bash
hf download bi0s/STAMP_V1 --include 'encoder_mtr/*' --local-dir weights
```

Install `huggingface_hub` to use `hf`. See [MODEL_CARD.md](MODEL_CARD.md)
for model directories. The model repository includes a SHA-256 file manifest.
The study data repository currently requires account access.

## Code layout

| Directory | Contents |
| --- | --- |
| `stamp/core/` | Encoding and reconstruction |
| `stamp/vocab/` | Motif vocabulary construction |
| `stamp/benchmark/` | Models, tasks and evaluation |
| `scripts/` | Command-line entry points and training launchers |
| `envs/` | Environment specifications |
| `examples/codec/` | Demo data and expected output |

Training and evaluation commands expose `--help`. HPC launchers require local
data paths and scheduler settings to be adapted to the target system.

## License

[MIT](LICENSE). Upstream models and datasets retain their respective terms.
