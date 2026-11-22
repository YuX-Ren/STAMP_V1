# STAMP model card

**STAMP: Sequential Tree Attachment-based Motif-Preserving representation**
is a topology-aware molecular language for fragment-level learning and editing.
This release contains generative Transformers, molecular encoders and
molecule–text encoder–decoders trained with STAMP representations.

- Paper: [STAMP as a topology-aware molecular language for fragment-level learning and editing](https://arxiv.org/abs/2610.18438)
- Authors: Yuxuan Ren, Jiatong Li, Ziqiao Meng and Yatao Bian, National University of Singapore.
- Source and runnable examples: [YuX-Ren/STAMP_V1](https://github.com/YuX-Ren/STAMP_V1).
- Checkpoints: [bi0s/STAMP_V1](https://huggingface.co/bi0s/STAMP_V1).
- Data: [bi0s/STAMP_V1-data](https://huggingface.co/datasets/bi0s/STAMP_V1-data).

## Released checkpoints

The Hub repository contains fifteen model directories. Download the complete
selected directory, including its vocabulary and configuration files.

| Directory | Intended task | Model file | Version |
| --- | --- | --- | --- |
| `ao_generation` | De novo generation and frozen PMO mutation | `model.pt` | AO-GPT rerun, selected step 300,000 |
| `fragment_infill` | Linker design, motif extension and superstructure generation | `model.pt` | Balanced infill adaptation, 3 epochs |
| `scaffold_decoration` | Scaffold decoration | `model.pt` | Additional task adaptation, 10 epochs |
| `encoder_mlm` | Molecular embeddings and supervised property fine-tuning | `pretrained_roberta.pt` | PubChem MLM, final epoch 10 |
| `encoder_mtr` | Molecular embeddings and supervised property fine-tuning | `pretrained_roberta.pt` | PubChem descriptor regression, best validation checkpoint |
| `molt5_mol2text_valid_best` | Molecular captioning | `model.safetensors` | Best-validation ChEBI-20 adaptation; 350k parent |
| `molt5_text2mol_valid_best` | Text-conditioned molecular generation | `model.safetensors` | Best-validation ChEBI-20 adaptation; 350k parent |
| `controlled_stamp` | Controlled stamp generation | `model.pt` | Four-layer causal Transformer, 10 epochs |
| `controlled_smiles` | Controlled smiles generation | `model.pt` | Four-layer causal Transformer, 10 epochs |
| `controlled_selfies` | Controlled selfies generation | `model.pt` | Four-layer causal Transformer, 10 epochs |
| `controlled_group_selfies` | Controlled group-selfies generation | `model.pt` | Four-layer causal Transformer, 10 epochs |
| `controlled_safe` | Controlled safe generation | `model.pt` | Four-layer causal Transformer, 10 epochs |
| `controlled_fragsmiles` | Controlled fragsmiles generation | `model.pt` | Four-layer causal Transformer, 10 epochs |
| `controlled_tsmiles` | Controlled t-smiles generation | `model.pt` | Four-layer causal Transformer, 10 epochs |
| `controlled_molcano` | Controlled molcano generation | `model.pt` | Four-layer causal Transformer, 10 epochs |

The `.pt` files use STAMP's custom PyTorch classes. The MolT5 subdirectories
use the Transformers T5 format; the repository root is not a single
`AutoModel.from_pretrained` model. The encoder checkpoints include their MLM
or MTR pretraining heads, not the task-specific MoleculeNet/MoleculeACE heads.
AO-GPT `.pt` files also contain optimizer/scaler and training-progress state;
their file size is larger than the model tensors alone. The MolT5 release
contains two task-adapted models with tokenizer/configuration files.
Standalone MolT5 pretraining weights are not included.
Frozen language-model API systems used elsewhere in the paper are not included
in this collection.

The current
[manifest.json](https://huggingface.co/bi0s/STAMP_V1/blob/main/manifest.json)
lists artifact byte sizes and SHA-256 values. All fifteen weight files were
verified against their local source hashes after upload.

### Four-layer controlled comparison

The eight `controlled_*` directories contain the original weights from
`benchmark_20260403_193312`, with their token vocabulary, training metadata,
training history and archived evaluation metrics. These are standard causal
Transformers: 4 layers, hidden size 256, 8 attention heads, feed-forward size
1,024, GELU, post-layer normalization, learned positional embeddings, dropout
0.1 and maximum sequence length 256. They are separate from the larger
any-order models above.

All representations share the 100,000/10,000/10,000 molecule train/validation/test
split. Training uses AdamW, learning rate 3e-4, weight decay 0.01, batch size
256, 10 epochs and seed 42. MOLCANO successfully encodes 99,998 training
molecules; the others encode all 100,000. Model parameter counts differ with
the token vocabulary. The STAMP checkpoint has 19,469 model tokens; use its
bundled `vocab.json` when loading it.

The following archived results each use 10,000 generated samples, temperature
1.0 and top-k 50. They are single-run controlled comparisons, separate from
the five-seed large-model results below.

| Representation | Vocabulary size | Parameters | Validity, % | Quality, % |
| --- | --- | --- | --- | --- |
| stamp | 19,469 | 13,193,216 | 97.38 | 75.54 |
| smiles | 37 | 3,244,032 | 79.66 | 55.21 |
| selfies | 95 | 3,273,728 | 100.00 | 43.21 |
| group-selfies | 90 | 3,271,168 | 100.00 | 46.40 |
| safe | 65 | 3,258,368 | 47.81 | 32.77 |
| fragsmiles | 4,303 | 5,428,224 | 91.87 | 66.76 |
| t-smiles | 50 | 3,250,688 | 87.71 | 58.42 |
| molcano | 349 | 3,403,776 | 40.21 | 31.37 |

A standalone PyTorch loader is included at the Hub root. It loads all eight
architectures and returns token logits; converting tokens to SMILES requires
the relevant representation adapter. The public source adapters for t-SMILES
and MOLCANO still require synchronization with the original experiment code.

```python
from huggingface_hub import snapshot_download
import importlib.util
import torch

root = snapshot_download("bi0s/STAMP_V1", allow_patterns=[
    "controlled_stamp/*", "controlled_decoder.py",
])
spec = importlib.util.spec_from_file_location("controlled_decoder", f"{root}/controlled_decoder.py")
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
model, itos = loader.load_controlled_model(f"{root}/controlled_stamp")
with torch.no_grad():
    logits = model(torch.tensor([[0, 1, 2]]))
```

## Architecture and representation

| Property | AO-GPT family | Molecular encoders | STAMP-MolT5 family |
| --- | --- | --- | --- |
| Architecture | Any-order autoregressive Transformer | Bidirectional RoBERTa-style Transformer | T5.1.1-Base encoder–decoder |
| Layers | 12 | 6 | 12 encoder + 12 decoder |
| Hidden dimension | 768 | 768 | 768 |
| Attention heads | 12 | 12 | 12 |
| Feed-forward dimension | 3,072 | 1,536 | 2,048 |
| Maximum sequence length | 256 | 128 | Pretrain source/target 192/96; fine-tune 256/256 |
| Dropout | 0.1 | 0.1 | 0.1 |
| Tokenizer entries | 12,659 | 4,761 | 62,649 |
| Parameter count | 112,057,459 | MLM: 36,373,401; MTR: 32,870,608, including their respective heads | Approximately 246.34M |

AO-GPT uses adaptive layer normalization conditioned on the target position
and attention causal in the selected prediction order. Its 12,000 retained
motifs are supplemented by structural, fallback and control tokens. Training
uses the hybrid vocabulary's motif fallback with frequency cutoff 5,000 and
fallback probability bounds 0.02–0.15. Fallback operates within the codec's
fragmentation constraints; a molecule outside the supported vocabulary can
still fail encoding.

The encoders use 4,666 retained motifs plus structural and special entries.
Both checkpoint files record `stamp_force_ring_aware_encode=False`. The same
motif vocabulary must be used when extracting embeddings or fine-tuning.

MolT5 extends the T5 tokenizer with 30,549 STAMP entries. The released
configuration uses gated-GELU feed-forward layers and tied input/output
embeddings. Molecular outputs require STAMP reconstruction; they are not
ordinary SMILES strings.

## Training data

The [data file index](https://huggingface.co/datasets/bi0s/STAMP_V1-data/blob/main/main/data/manifests/release_files.tsv)
provides downloadable paths, row counts and checksums.

| Model or stage | Training inputs | Validation inputs |
| --- | --- | --- |
| Large AO-GPT | 425,457,500 ZINC-derived STAMP records, including augmentation | 51,000,653 retained records; 20,000 sampled for validation |
| Fragment infill | 980,127 constructed examples | 19,873 retained; 8,000 sampled for validation |
| Scaffold adaptation | 294,160 constructed examples | 5,840 examples |
| Encoder pretraining | 9,690,075 cleaned PubChem SMILES; 9,571,139 successfully encoded | 100,000 cleaned SMILES; 98,840 successfully encoded |
| MolT5 pretraining | C4 English text and ZINC15 STAMP examples in equal proportions | Up to 20,000 examples from each source |
| ChEBI-20 adaptation | 26,407 molecule–text pairs | 3,301 pairs; official test partition has 3,300 pairs |

The fragment dataset contains one million constructed examples: 333,334 linker,
333,333 motif-extension and 333,333 scaffold-decoration examples. These are
training examples rather than one million verified distinct molecules.

The released MolT5 molecular corpus contains 101,862,943 encoded records in
43 source shards, including its validation-source shard. The training recipe
caps molecular training at 100M records. C4's 1,032 English gzip files are
referenced by a pinned upstream revision in
[c4_upstream_files.tsv](https://huggingface.co/datasets/bi0s/STAMP_V1-data/blob/main/main/data/manifests/c4_upstream_files.tsv);
the raw C4 files are not duplicated in the STAMP data repository.

Property evaluation uses MoleculeNet scaffold splits and supplied MoleculeACE
train/test assignments. Canonicalization, invalid-input removal and
vocabulary construction are workflow-specific; held-out benchmark partitions
do not establish exclusion from every large or third-party pretraining corpus.
The data release's [coverage report](https://huggingface.co/datasets/bi0s/STAMP_V1-data/blob/main/main/data/manifests/coverage.json)
maps the exact S2 task snapshots (six tasks, 500 examples each), shared-label
transfer snapshot (120 examples) and change-identification snapshot (100 pairs)
to their original experiment manifests. These downstream evaluation inputs are
separate from the generative and encoder pretraining corpora.

## Objectives and optimization

### AO-GPT generation and infilling

AO-GPT minimizes token cross-entropy in a sampled prediction order. Random
orders are used for 90% of examples and identity order for the remainder,
keeping the first and last non-ignored positions fixed. Prompt and padding
positions are excluded from the infill loss. Group-aligned completions are
spliced into retained context before molecular decoding.

| Setting | `ao_generation` | `fragment_infill` | `scaffold_decoration` |
| --- | --- | --- | --- |
| Initialization | Separate large-corpus rerun | Original large-corpus AO-GPT, step 300,000 | Released fragment-infill model |
| Budget | Released checkpoint at step 300,000 | 3 epochs; 11,487 optimizer steps | 10 epochs; 11,500 optimizer steps |
| Global batch | 1,024 | 256 | 256 |
| GPUs / batch per GPU | 4 / 256 | 1 / 256 | 1 / 256 |
| Learning rate | 3e-4 | 1e-4 | 5e-5 |
| Optimizer | AdamW | AdamW | AdamW |
| Weight decay | 0.1 | 0.1 | 0.1 |
| Gradient clipping | 1.0 | 1.0 | 1.0 |
| Precision / seed | BF16 / 42 | BF16 / 42 | BF16 / 42 |

The infill parent is the original pretraining run, not the separately released
`ao_generation` rerun. That original parent checkpoint is not separately
packaged here. This distinction affects retracing adaptation, not inference
with the released infill weights.

### Encoder pretraining

MLM masks 15% of selected tokens using 80% mask replacement, 10% random
replacement and 10% unchanged tokens. MTR regresses 208 standardized RDKit
descriptors from pooled molecular representations. The MTR checkpoint contains
descriptor names, means and standard deviations, as well as the regression head.

| Setting | `encoder_mlm` | `encoder_mtr` |
| --- | --- | --- |
| Epochs | 10 | 5 |
| Batch size | 512 | 512 |
| Learning rate | 5e-4 | 3e-4 |
| Optimizer / weight decay | AdamW / 0.01 | AdamW / 0.01 |
| LR schedule in trainer | 10% linear warm-up, then cosine decay | 10% linear warm-up, then cosine decay |
| Precision / seed | BF16 / 42 | BF16 / 42 |
| Saved-model selection | Final epoch | Lowest validation MSE |
| Recorded validation result | Final loss 0.894945; perplexity 2.4472 | Standardized-descriptor MSE 0.0343506 |

The standard downstream fine-tuning configuration uses 20 epochs, batch 32
and learning rate 2e-5. Classification uses ROC-AUC and regression uses RMSE;
a downstream task head must be trained before using these encoders as property
predictors. MLM perplexity and MTR descriptor MSE measure different objectives
and should not be compared numerically.

### STAMP-MolT5 pretraining and adaptation

MolT5 is initialized from `google/t5-v1_1-base`. Each mixed pretraining batch
combines C4 text and molecular STAMP examples under span denoising, with noise
density 0.15 and mean noise-span length 3.

| Setting | Pretraining | ChEBI-20 adaptation |
| --- | --- | --- |
| Parent checkpoint | Step 350,000 | Up to 30 epochs; best-validation checkpoint |
| Global batch | 1,024, across four H200 GPUs | 32, on one GPU |
| Learning rate | 1.4e-4 | 5e-4 |
| Optimizer / weight decay | AdamW / 0.01 | AdamW / 0.01 |
| Warm-up | 5,000 steps | 5% of planned steps |
| Gradient clipping | 1.0 | 1.0 |
| Precision / seed | BF16 / 42 | BF16 / 42 |
| Validation / early stopping | Every 10,000 steps | Every 500 steps; patience 6 validation evaluations |

The pretraining trainer uses cosine decay after warm-up; the paper's Methods
describe linear decay for this stage. The downstream fine-tuning schedule
defaults to linear decay after warm-up.

| Downstream checkpoint | Saved epoch | Saved step | Validation loss |
| --- | ---: | ---: | ---: |
| `molt5_mol2text_valid_best` | 13 | 11,500 | 0.550620 |
| `molt5_text2mol_valid_best` | 13 | 11,500 | 0.554479 |

## Evaluation results

### Molecular generation

These are archived five-seed experiment results, not a new evaluation of the
uploaded files. Quality counts unique valid molecules passing QED ≥ 0.6 and
SA ≤ 4 per generation request. Constrained-task quality also requires retention
of the supplied fragment constraints. Diversity is based on Morgan fingerprint
Tanimoto distances. Values below are mean ± sample standard deviation.

| Task | Validity, % | Quality, % | Diversity |
| --- | ---: | ---: | ---: |
| De novo | 99.71 ± 0.15 | 87.99 ± 1.08 | 0.861 ± 0.001 |
| Linker design | 98.20 ± 0.35 | 36.56 ± 0.22 | 0.565 ± 0.003 |
| Motif extension | 98.38 ± 0.38 | 55.12 ± 0.80 | 0.609 ± 0.002 |
| Scaffold decoration | 98.84 ± 0.24 | 30.14 ± 0.89 | 0.521 ± 0.003 |
| Superstructure generation | 98.97 ± 0.21 | 58.00 ± 1.53 | 0.564 ± 0.002 |

[De novo results](https://github.com/YuX-Ren/STAMP_V1/blob/main/assets/results/denovo.json),
[fragment summary](https://github.com/YuX-Ren/STAMP_V1/blob/main/assets/results/fragment_generation.tsv),
and [per-seed results](https://github.com/YuX-Ren/STAMP_V1/blob/main/assets/results/fragment_generation_per_seed.tsv)
are available with the code. De novo sampling requests 1,024 molecules per seed
at temperature 0.7 and top-p 0.95. Fragment settings use temperature 1.3;
the task runner selects each task's nucleus threshold, diversity-prefix penalty
and hole-completion mode.

### Molecule–text evaluation and version correspondence

The following are archived ChEBI-20 test metrics. Captioning uses SciBERT
WordPiece BLEU and METEOR with ROUGE on text. Molecular evaluation decodes
STAMP before scoring; validity and exact match below are percentages.

| Captioning checkpoint | BLEU-2 | BLEU-4 | METEOR | ROUGE-L |
| --- | ---: | ---: | ---: | ---: |
| `molt5_mol2text_valid_best` | 0.511805 | 0.422907 | 0.550187 | 0.566299 |

| Text-to-molecule checkpoint | Validity, % | Exact match, % | MACCS similarity | Morgan similarity |
| --- | ---: | ---: | ---: | ---: |
| `molt5_text2mol_valid_best` | 97.5455 | 10.5455 | 0.774475 | 0.538839 |

Each downstream model directory includes `evaluation_metrics.json` with its
full archived metric record from `eval_test_v2`. Both released MolT5 models
were adapted from the step-350,000 pretraining checkpoint; standalone
pretraining weights are not included.

### PMO use of the pretrained baseline

The PMO runner uses the same `ao_generation` weights (112,057,459 parameters)
as a frozen motif-span mutation model. No additional PMO checkpoint is needed.
The released `run_pmo_baseline.py` uses the shared `stamp_pmo`
configuration with a 10,000-call budget per task. See the
[PMO command](https://github.com/YuX-Ren/STAMP_V1#practical-molecular-optimization-pmo)
for input downloads and execution.

## Download and use

Run examples from the source checkout after installing the environment in the
[README](https://github.com/YuX-Ren/STAMP_V1#requirements-and-installation).

```bash
python -m pip install huggingface_hub
hf download bi0s/STAMP_V1 --include 'ao_generation/*' --local-dir checkpoints
python scripts/benchmark/run_paper_generation.py denovo \
  --model-root checkpoints --output-dir results/denovo
```

For fragment tasks, download `fragment_infill/*` or `scaffold_decoration/*`
and choose `linker`, `motif`, `scaffold` or `superstructure` in the same runner.
Use `--device cpu --smoke` for a two-sample installation check, not benchmark
reproduction. The repository includes the corresponding fragment queries.

### Portable encoder feature extraction

This example explicitly uses the downloaded motif vocabulary instead of the
old training-system path stored in the checkpoint. It returns a 768-dimensional
pooled representation; it does not predict a biological property.

```python
import json
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from stamp.benchmark.lm import TokenVocab
from stamp.benchmark.representations import build_representation
from stamp.benchmark.roberta_property import MolRoBERTa, RoBERTaConfig

snapshot = Path(snapshot_download(
    "bi0s/STAMP_V1",
    revision="1830ad743012ff605f361a824b3ffcfd49d36cf2",
    allow_patterns=["encoder_mtr/*"],
))
d = snapshot / "encoder_mtr"
ckpt = torch.load(d / "pretrained_roberta.pt", map_location="cpu", weights_only=True)
cfg = RoBERTaConfig(**ckpt["cfg"])
vocab = TokenVocab.from_dict(json.loads((d / "vocab.json").read_text()))
encoder = MolRoBERTa(len(vocab.itos), cfg, pad_id=vocab.pad_id)
state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["encoder_state"].items()}
encoder.load_state_dict(state)
encoder.eval()
rep = build_representation(
    "stamp", stamp_vocab=str(d / "vocab_top4666.txt"),
    stamp_force_ring_aware_encode=ckpt.get("stamp_force_ring_aware_encode", False),
)
tokens = rep.tokenize(rep.encode("CCO"))
ids = [vocab.bos_id] + vocab.encode_tokens(tokens) + [vocab.eos_id]
with torch.inference_mode():
    features = encoder(torch.tensor([ids[:cfg.max_seq_len]]), return_cls=True)
print(features.shape)  # torch.Size([1, 768])
```

Use `encoder_mlm` in place of `encoder_mtr` to extract MLM encoder features.
For MolT5, load the chosen subdirectory's tokenizer and T5 model, and use
`scripts/benchmark/eval_molt5_chebi20.py` for evaluation. Its output-decoding
routine merges tokenizer pieces into STAMP motifs before molecular
reconstruction. Direct text detokenization is not a substitute for that step.
The [task instructions](https://github.com/YuX-Ren/STAMP_V1#run-paper-tasks)
include the prepared ChEBI-20 test-data download and full evaluation command.

## Hardware, compatibility and validation scope

Large AO-GPT training used four H200 GPUs; infill and scaffold adaptation used
one GPU each. Encoder training used one H200, and MolT5 pretraining used four.
The study reports approximately 2 h 45 min for MLM and 3 h for MTR encoder
training. These are selected run costs, not a total compute or carbon account.

The training environment recipe specifies Python 3.12, PyTorch 2.6.0,
CUDA 12.4 and RDKit 2025.09.4. CPU installation checks also ran with RDKit
2023.09.6. Tokenization depends on RDKit canonical atom ordering; preserve the
released vocabularies and check compatibility when changing RDKit versions.
A complete historical dependency lock for every run is not supplied.
The translation evaluator's additional dependencies are in
[requirements-translation.txt](https://github.com/YuX-Ren/STAMP_V1/blob/main/envs/requirements-translation.txt).

Release checks verified weight hashes, companion files and representative
CPU inference/evaluation paths. They did not rerun all paper benchmarks or
establish equality of results across hardware, dependencies or decoder versions.

## Intended use and limitations

The models support research on molecular representation learning, generation,
fragment-constrained construction and molecule–text translation. Encoder
features are suitable inputs to downstream research models after task-specific
validation. Generated structures and captions are model outputs, not measured
chemical or biological properties.

Training and benchmark collections cover selected chemical domains. Fixed
inputs and matched protocols help compare representations but do not establish
generalization to arbitrary chemical space. Molecules can exceed sequence
length or vocabulary limits. Decoder validity does not establish practical
synthesizability, activity, safety or clinical suitability. The release does
not establish complete removal of downstream benchmarks from all pretraining
sources or provide prospective experimental validation.

## Licensing and citation

The [STAMP source code](https://github.com/YuX-Ren/STAMP_V1/blob/main/LICENSE)
is MIT licensed. Upstream T5 weights, datasets and third-party assets retain
their applicable terms; the source-code license does not replace those terms.
The Hub collection is labelled `other` to avoid assigning one code license to
all combined model and data origins.

```bibtex
@misc{ren2026stamp,
  title={STAMP as a topology-aware molecular language for fragment-level learning and editing},
  author={Ren, Yuxuan and Li, Jiatong and Meng, Ziqiao and Bian, Yatao},
  year={2026},
  eprint={2610.18438},
  archivePrefix={arXiv}
}
```
