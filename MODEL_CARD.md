# STAMP study models

STAMP is a molecular representation used by several model families. This card
describes the study workflows; weights are separate from the source checkout.
The deterministic codec demo is available without a trained model.

## Models and intended use

| Family | Implementation | Intended research use |
| --- | --- | --- |
| Causal and any-order Transformers | `stamp/benchmark/lm.py`, `scripts/benchmark/train_ao_gpt_parquet.py` | Molecular generation and fragment infilling |
| ChemBERTa-style encoders | `stamp/benchmark/roberta_property.py` | Property prediction and frozen molecular representations |
| T5.1.1 encoder-decoder | `scripts/benchmark/train_molt5_c4_zinc_mixed.py`, `scripts/benchmark/finetune_molt5_chebi20.py` | Molecule-to-text and text-to-molecule translation |

Frozen upstream language models used in editing comparisons include Qwen and
DeepSeek; they are external baselines, not STAMP-trained checkpoints.

## Training data and configurations

| Workflow | Data and configuration reference |
| --- | --- |
| Large AO-GPT and fragment infilling | ZINC-derived STAMP corpora; [recorded sample counts and infill parameters](docs/genmol_pretraining_parameters.md) |
| Encoder pretraining | PubChem-10M; preprocessing in `scripts/benchmark/prepare_pubchem_pretrain_splits.py`; launcher `scripts/hpc/pretrain_roberta_stamp_pubchem.pbs` |
| STAMP-MolT5 | C4 text and ZINC15 molecules in equal proportions; [STAMP-MolT5 training report](docs/molt5_stamp_chebi20_20260708.md) |

The documented AO-GPT infill model uses 12 layers, hidden size 768, 12 heads,
FFN size 3,072, dropout 0.1 and maximum sequence length 256. It initializes
from the 300,000-step large-corpus AO-GPT checkpoint and trains for three
epochs on 980,127 constructed training examples with global batch 256,
learning rate 1e-4, weight decay 0.1 and seed 42. The model has 112,057,459
parameters. It uses any-order autoregressive cross-entropy with random-order
probability 0.9. These parameters describe this checkpoint specifically.

Checkpoint vocabularies, token order, RDKit version and model architecture must
be kept together. Use each run's `train_meta.json` and vocabulary artifacts;
launcher defaults can differ from the settings used in a completed run.

## Evaluation

Evaluation covers molecular validity, uniqueness, quality and diversity;
fragment preservation; oracle-budget molecular optimization; property
prediction; and molecule-text translation. See
[the experiment guide](docs/experiments.md),
[generation and infill result protocols](docs/publishable_results_20260612.md),
and [translation evaluation](docs/molt5_stamp_chebi20_20260708.md).

Partitions are task-specific: generative train/validation corpora, MoleculeNet
scaffold splits, supplied MoleculeACE assignments, and official ChEBI-20
splits. Repeated random seeds are not k-fold cross-validation.

## Limitations and dataset bias

The evaluated chemical collections and benchmarks represent selected chemical
domains. Matched inputs and protocols reduce confounding between
representations, but do not remove dataset selection bias. The study does not
establish generalization to arbitrary chemical space or fully independent
testing with exclusion from every upstream pretraining corpus. Benchmark
splits and canonical deduplication do not prove such global independence.

Molecular syntax validity does not establish synthesizability, biological
activity or suitability for clinical use. Models are intended for research;
experimental validation is required for generated chemical proposals.

## Access and licensing

Ten study checkpoints and their companion files are publicly available at
[bi0s/STAMP_V1](https://huggingface.co/bi0s/STAMP_V1). The standalone 350k
MolT5 pretraining checkpoint is not included; its downstream models are included.
[Access instructions](docs/data_and_model_access.md) distinguish these weights
from the private study data snapshot and public upstream weights. The source code uses the
[MIT License](LICENSE); upstream model and dataset terms apply separately.

## Computational resources

The checklist records single-H200 encoder training times of 2 h 45 min for
MLM and 3 h for MTR. The documented STAMP-MolT5 run used four H200 GPUs;
its [run report](docs/molt5_stamp_chebi20_20260708.md) records approximately
52 hours. These are individual runs, not total study compute or carbon costs.
