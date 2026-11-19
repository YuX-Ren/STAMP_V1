# STAMP data and model access

The code checkout includes the ten-molecule [CPU demonstration](../examples/codec/README.md).
It runs without authentication, datasets or pretrained weights. Exact study
data and model artifacts are distributed separately.

## Study snapshot

The study data location is
[`bi0s/STAMP_V1-data`](https://huggingface.co/datasets/bi0s/STAMP_V1-data).
It is described as private in this repository; an anonymous request to
`https://huggingface.co/api/datasets/bi0s/STAMP_V1-data` returned HTTP 401 on
19 November 2026. The demo does not establish public availability of these
study train, validation and test files.

With access granted to your Hugging Face account, install the download client
and download the main data tree:

```bash
python -m pip install huggingface_hub
hf auth login
hf download bi0s/STAMP_V1-data --repo-type dataset \
  --include 'main/data/*' --local-dir /path/to/stamp-snapshot
```

Copy the contents of `/path/to/stamp-snapshot/main/data/` into a separate
checkout's `data/` directory. Use the snapshot's provenance and SHA-256
inventory to verify files before running experiments. Access must be arranged
with the corresponding author or dataset owner; authentication alone does not
grant membership. Public review access and a fixed release revision remain
release requirements.

## Required artifacts by workflow

| Workflow | Required inputs and artifacts | Entry point or protocol |
| --- | --- | --- |
| Codec demo | Bundled `examples/codec/molecules.txt` | `python scripts/core/run_codec_demo.py` |
| Controlled generation | Prepared train/validation/test SMILES; training-only motif vocabulary; trained model | [Experiment runbook](experiments_runbook.md) |
| Large AO-GPT | Encoded ZINC parquet shards, hybrid vocabulary, model and `train_meta.json` | [Recorded pretraining parameters](genmol_pretraining_parameters.md) |
| Fragment-constrained generation | Query parquet files, vocabulary and task checkpoint | [Checkpoint and task mapping](publishable_results_20260612.md) |
| Encoder properties | PubChem splits and vocabulary; pretrained encoder; MoleculeNet or MoleculeACE assignments | `scripts/hpc/pretrain_roberta_stamp_pubchem.pbs`, `scripts/benchmark/finetune_moleculeace.py` |
| Molecule-text translation | C4 and ZINC15 pretraining inputs; ChEBI-20 official splits; tokenizer and model directory | [STAMP-MolT5 protocol](molt5_stamp_chebi20_20260708.md) |

Paths beginning with `/scratch/`, `/nfs/` or `/Project_Storage/` in run reports
identify local artifacts, not download links. The data snapshot must not be
assumed to include model weights. Ten selected study checkpoints are public at
[bi0s/STAMP_V1](https://huggingface.co/bi0s/STAMP_V1), with vocabularies,
configurations and `manifest.json` containing file sizes and SHA-256 checksums.
Anonymous access, 62 artifact file sizes and all ten LFS weight hashes were
verified at revision `1830ad743012ff605f361a824b3ffcfd49d36cf2`.

Download the MTR encoder and all its companion files:

```bash
hf download bi0s/STAMP_V1 --revision 1830ad743012ff605f361a824b3ffcfd49d36cf2 \
  --include 'encoder_mtr/*' --local-dir weights
```

See the
[local checkpoint inventory](checkpoint_inventory.md) for verified paths,
sizes, task mappings and required companion files.

## Public upstream model identifiers

The supplied machine learning checklist identifies these upstream releases:

| Model | Release |
| --- | --- |
| T5.1.1 Base | [google/t5-v1_1-base](https://huggingface.co/google/t5-v1_1-base) |
| ChemBERTa-2 10M MTR | [DeepChem/ChemBERTa-10M-MTR](https://huggingface.co/DeepChem/ChemBERTa-10M-MTR) |
| Qwen2.5 7B Instruct | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |

These upstream releases do not replace STAMP-trained weights, tokenizers or
motif vocabularies. API-based evaluations also require the provider/model
identifier, prompt, decoding settings and evaluation inputs of the run.

## Data processing and leakage controls

`stamp/benchmark/data.py` implements canonicalization, invalid-input removal,
deduplication and seeded splitting. The PubChem preparation script additionally
keeps the largest fragment. Vocabulary mining should use the training partition.
`scripts/benchmark/decontaminate_pretrain.py` implements exclusion by the first
14 characters of target InChIKeys for its specified inputs; the script's
presence does not establish that every published checkpoint used this exclusion.

To release exact study artifacts, supply a fixed revision, source and license,
split assignments, preprocessing configuration, vocabulary, checkpoint selection
rule and SHA-256 checksums. Record overlap checks separately for each
pretraining corpus and evaluation benchmark. Do not mark complete independence
or public availability solely from the existence of a preparation script.
