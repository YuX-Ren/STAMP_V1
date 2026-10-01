# Publishable Results Summary, 2026-06-12

This note consolidates the stable results for STAMP generative modeling and
FC-STAMP fragment-constrained generation. It separates the main publishable
claims from diagnostics that should remain in the appendix.

## Main Claims

1. **STAMP supports high-quality de novo generation with a compact hybrid
   motif/token vocabulary.** A 112M-parameter AO-GPT trained on full-scale
   STAMP data reaches near-saturated validity/uniqueness and high QED/SA
   quality under standard autoregressive sampling.

2. **FC-STAMP turns fragment-constrained molecular design into structured
   in-place STAMP infilling.** The model generates missing STAMP spans and
   splices them back into the prompt before decoding with the standard STAMP
   decoder. No global anchor token or modified molecular decoder is required.

3. **FC-STAMP is competitive with, and often stronger than, GenMol-style
   fragment generation baselines.** Across five random seeds, validity is
   consistently about 98-99%, uniqueness is above 93% for three of four tasks,
   and constraint-aware quality is high on linker, motif extension, and
   superstructure generation.

## De Novo STAMP Generation

Checkpoint:

```text
/scratch/yuxuan.ren/data/ckpt_stamp/zinc_l3_full_aogpt_uncond_top12k_4g_rerun_20260604/legacy_tree
```

Shared archive:

```text
/Project_Storage/CFP-03/CFP03-CF-130/data/zinc_pretrain/ckpt_stamp/zinc_l3_full_aogpt_uncond_top12k_4g_rerun_20260604
```

Configuration:

```text
model_type: ao_gpt
parameters: 112,057,459
d_model: 768
n_layers: 12
n_heads: 12
d_ff: 3072
max_seq_len: 256
vocab_type: hybrid
vocab_size: 12,659
atomic_motifs: 12,000
motif_frequency_cutoff: 5,000
motif_fallback: enabled
training_fallback_p: 0.02 to 0.15
training_rows: 425,457,500
global_batch_size: 1,024
precision: bf16
```

Evaluation:

```text
N = 1024 samples/seed
seeds = 5
temperature = 0.7
top_p = 0.95
top_k = 0
```

Metrics are percentages except diversity.

| Model | Validity | Uniqueness | Quality | Diversity |
| --- | ---: | ---: | ---: | ---: |
| old_top12k | 99.34 +/- 0.16 | 99.02 +/- 0.21 | 87.46 +/- 0.56 | 0.861 +/- 0.001 |
| rerun_501411 | 99.71 +/- 0.15 | 99.39 +/- 0.25 | 87.99 +/- 1.08 | 0.861 +/- 0.001 |

Quality is the fraction of unique valid molecules passing QED >= 0.6 and
SA <= 4.0, normalized by the number of generated samples.

Source files:

```text
results/ao_top12k_rerun_compare_20260605.md
results/ao_top12k_rerun_compare_20260605.json
```

## FC-STAMP Task Definition

The final formulation is structured in-place infilling over valid STAMP spans.
Each example is represented as:

```text
[TASK] left [INFILL] right [GEN] target_span [EOS]
```

At inference time:

```text
final_stamp = splice(left [INFILL] right, generated_span)
smiles = STAMP_decode(final_stamp)
```

Task-specific construction:

| Task | Prompt construction | Generated content |
| --- | --- | --- |
| linker_design | two fixed motifs/fragments with an in-place gap | linker STAMP span |
| motif_extension | fixed motif with an open extension site | extension subtree |
| scaffold_decoration | fixed scaffold with one or more masked subtrees | decorations, evaluated with sequential-hole filling |
| superstructure_generation | query molecule expanded over valid heavy-atom attach sites | one generated subtree per attach-site prompt |

Important implementation choices:

```text
STAMP decoder is unchanged.
[INFILL], [GEN], and task tokens are control/edit tokens only.
B tokens are not generated as target anchors for linker/scaffold holes.
Scaffold decoration and superstructure share the same training objective.
Superstructure test prompts enumerate valid attach sites and aggregate results by source molecule.
Stereo is retained in the final archived evaluation.
```

## FC-STAMP Final 5-Seed Results

Archive:

```text
results/fcstamp_genmol_final_20260604
```

Seeds:

```text
20260525, 20260526, 20260527, 20260528, 20260529
```

Metrics are reported as mean +/- standard deviation across five seeds.
The archive also stores sample variance (`ddof=1`) in `summary.csv` and
`summary.json`.

| Task | Validity | Uniqueness | Constraint Quality | Diversity | Distance |
| --- | ---: | ---: | ---: | ---: | ---: |
| linker_design | 98.20 +/- 0.35 | 94.78 +/- 0.73 | 36.56 +/- 0.22 | 0.565 +/- 0.003 | 0.566 +/- 0.001 |
| motif_extension | 98.38 +/- 0.38 | 96.11 +/- 0.73 | 55.12 +/- 0.80 | 0.609 +/- 0.002 | 0.658 +/- 0.002 |
| scaffold_decoration | 98.84 +/- 0.24 | 83.99 +/- 1.59 | 30.14 +/- 0.89 | 0.521 +/- 0.003 | 0.563 +/- 0.002 |
| superstructure_generation | 98.97 +/- 0.21 | 93.38 +/- 0.26 | 58.00 +/- 1.53 | 0.564 +/- 0.002 | 0.726 +/- 0.001 |

Constraint Quality is the task-preserving quality metric used by the final
evaluator: the molecule must satisfy the fragment/substructure constraint and
pass QED >= 0.6 and SA <= 4.0.

## Comparison To GenMol References

GenMol and GenMol V2 numbers below are the reference constants preserved in
`scripts/benchmark/eval_fcstamp_genmol.py`. FC-STAMP numbers are the 5-seed
means above.

| Task | Method | Validity | Uniqueness | Quality | Diversity | Distance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| linker_design | GenMol | 100.0 | 83.7 | 21.9 | 0.547 | 0.563 |
| linker_design | GenMol V2 | 100.0 | 76.6 | 18.4 | 0.512 | 0.539 |
| linker_design | FC-STAMP | 98.2 | 94.8 | 36.6 | 0.565 | 0.566 |
| motif_extension | GenMol | 82.9 | 77.5 | 30.1 | 0.617 | 0.682 |
| motif_extension | GenMol V2 | 99.4 | 84.5 | 49.0 | 0.626 | 0.659 |
| motif_extension | FC-STAMP | 98.4 | 96.1 | 55.1 | 0.609 | 0.658 |
| scaffold_decoration | GenMol | 96.6 | 82.7 | 31.8 | 0.591 | 0.651 |
| scaffold_decoration | GenMol V2 | 99.2 | 90.5 | 39.7 | 0.571 | 0.604 |
| scaffold_decoration | FC-STAMP | 98.8 | 84.0 | 30.1 | 0.521 | 0.563 |
| superstructure_generation | GenMol | 97.5 | 83.6 | 34.8 | 0.599 | 0.762 |
| superstructure_generation | GenMol V2 | 99.7 | 89.8 | 39.0 | 0.551 | 0.769 |
| superstructure_generation | FC-STAMP | 99.0 | 93.4 | 58.0 | 0.564 | 0.726 |

Interpretation:

- FC-STAMP strongly improves uniqueness and quality on linker design.
- FC-STAMP improves motif-extension uniqueness and quality while preserving
  high validity.
- FC-STAMP is strongest on superstructure generation, with much higher quality
  than both GenMol references.
- Scaffold decoration is the main remaining weak point: validity is high, but
  uniqueness, quality, diversity, and distance are below GenMol V2. This should
  be discussed as a limitation and a target for future data/task construction.

## Final Checkpoints And Data

Task checkpoint mapping:

| Task | Checkpoint |
| --- | --- |
| linker_design | `/scratch/yuxuan.ren/data/ckpt_stamp/fcstamp_inplace_balanced_1m_aogpt_top12k_1g/legacy_tree/model.pt` |
| motif_extension | `/scratch/yuxuan.ren/data/ckpt_stamp/fcstamp_inplace_balanced_1m_aogpt_top12k_1g/legacy_tree/model.pt` |
| scaffold_decoration | `/scratch/yuxuan.ren/data/ckpt_stamp/fcstamp_task_scaffold_super_redesign_shared300k_stereo_from_mixed_len256_ep10/legacy_tree/model.pt` |
| superstructure_generation | `/scratch/yuxuan.ren/data/ckpt_stamp/fcstamp_inplace_balanced_1m_aogpt_top12k_1g/legacy_tree/model.pt` |

Final task files:

| Task | File | Rows |
| --- | --- | ---: |
| linker_design | `data/genmol_fcstamp_linker_only_stereo.parquet` | 10 |
| motif_extension | `data/genmol_fcstamp_motif_only_stereo.parquet` | 10 |
| scaffold_decoration | `data/genmol_fcstamp_scaffold_decoration_only_stereo.parquet` | 10 |
| superstructure_generation | `data/genmol_fcstamp_super_valid4_stereo.parquet` | 39 |

Training sources:

```text
fcstamp_inplace_balanced_1m_aogpt_top12k_1g:
  train rows: 980,127
  valid rows: 8,000 encoded
  init: zinc_l3_full_aogpt_uncond_top12k_4g step_00300000
  epochs: 3

fcstamp_task_scaffold_super_redesign_shared300k_stereo_from_mixed_len256_ep10:
  train rows: 294,160
  valid rows: 5,840
  init: fcstamp_inplace_balanced_1m_aogpt_top12k_1g/model.pt
  epochs: 10
```

## Sampling Settings

Common seeds:

```text
20260525, 20260526, 20260527, 20260528, 20260529
```

Final settings from the archive:

| Task | Samples | Batch | T | top_p | top_k | Prefix penalty | max_new | Special mode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| linker_design | 100 | 100 | 1.3 | 1.0 | 0 | 4 | 64 | none |
| motif_extension | 100 | 100 | 1.3 | 1.0 | 0 | 4 | 64 | none |
| scaffold_decoration | 100 | 100 | 1.3 | 0.95 | 0 | 2 | 24 | sequential holes |
| superstructure_generation | 100 | 25 | 1.3 | 1.0 | 0 | 3 | 64 | aggregate attach sites, 25 samples/site |

## Reproduction

Run the archived 5-seed evaluation:

```bash
bash results/fcstamp_genmol_final_20260604/run_5seed_eval.sh
```

Key source artifacts:

```text
results/fcstamp_genmol_final_20260604/README.md
results/fcstamp_genmol_final_20260604/manifest.json
results/fcstamp_genmol_final_20260604/summary.csv
results/fcstamp_genmol_final_20260604/summary.json
results/fcstamp_genmol_final_20260604/per_seed.csv
results/fcstamp_genmol_final_20260604/script_snapshots/
```

## Recommended Paper Framing

Use the following as the central method statement:

```text
FC-STAMP formulates fragment-constrained molecular generation as task-conditioned
structured span infilling over STAMP sequences. The model generates only the
missing STAMP span, which is spliced back into the fixed prompt and decoded by
the original STAMP decoder.
```

Primary result claims:

- De novo STAMP AO-GPT reaches 99.7% validity, 99.4% uniqueness, and 88.0%
  QED/SA quality under direct sampling.
- FC-STAMP reaches 98-99% validity on all four GenMol-style constrained tasks.
- FC-STAMP improves quality over GenMol V2 on linker design, motif extension,
  and superstructure generation.
- FC-STAMP improves uniqueness over GenMol V2 on linker design, motif extension,
  and superstructure generation.

Limitations to state explicitly:

- Scaffold decoration is not yet stronger than GenMol V2 on quality/diversity.
- Current publishable FC-STAMP results use task-specific sampling settings.
- AO random-order inference is not a main result yet; current random-order
  samplers are mismatched to the trained shifted-token AO objective.
- The final archive was created from a dirty working tree, so script snapshots
  should be treated as the exact reproducibility reference.

## Appendix Diagnostics

AO random-order inference should be kept as a diagnostic appendix rather than a
main claim. The rerun checkpoint is AO-trained, but naive random-order AR on the
current hybrid top12k checkpoint is weak:

```text
random-order AR, N=1024:
  V about 46-47
  Q about 4-5
```

This indicates an inference/objective mismatch, not a failure of the main
autoregressive or FC-STAMP generation results.
