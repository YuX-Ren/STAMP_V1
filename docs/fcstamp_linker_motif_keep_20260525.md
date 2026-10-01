# FC-STAMP Linker/Motif Kept Result

Date: 2026-05-25

This is the kept mixed FC-STAMP checkpoint/result for linker design and motif extension.

## Checkpoint

Model directory:

```text
/scratch/yuxuan.ren/data/ckpt_stamp/fcstamp_inplace_balanced_1m_aogpt_top12k_1g/legacy_tree
```

Model file:

```text
/scratch/yuxuan.ren/data/ckpt_stamp/fcstamp_inplace_balanced_1m_aogpt_top12k_1g/legacy_tree/model.pt
```

## Evaluation Inputs

Task parquet:

```text
data/genmol_fcstamp_infill_stereo.parquet
```

Per-task sampling config:

```text
data/fcstamp_per_task_sampling_stereo_best.json
```

Eval output:

```text
/scratch/yuxuan.ren/data/ckpt_stamp/fcstamp_inplace_balanced_1m_aogpt_top12k_1g/legacy_tree/eval_per_task_stereo/full_best/fcstamp_genmol_eval_seed20260525.json
```

Eval script:

```text
scripts/hpc/eval_fcstamp_genmol.pbs
scripts/benchmark/eval_fcstamp_genmol.py
```

## Kept Metrics

Samples per drug/case: 100

Sampling parameters:

```json
{
  "linker_design": {
    "temperature": 1.3,
    "top_p": 1.0,
    "diverse_prefix_penalty": 4.0
  },
  "motif_extension": {
    "temperature": 1.3,
    "top_p": 1.0,
    "diverse_prefix_penalty": 4.0
  }
}
```

Results:

```text
linker_design:
  validity 98.1
  uniqueness 94.6
  quality 49.1
  preservation 78.4

motif_extension:
  validity 98.2
  uniqueness 95.6
  quality 54.7
  preservation 100.0
```

## Reproduce

```bash
qsub -N fcst_pt_best \
  -v MODEL_DIR=/scratch/yuxuan.ren/data/ckpt_stamp/fcstamp_inplace_balanced_1m_aogpt_top12k_1g/legacy_tree,TASK_PARQUET=data/genmol_fcstamp_infill_stereo.parquet,OUTPUT_DIR=/scratch/yuxuan.ren/data/ckpt_stamp/fcstamp_inplace_balanced_1m_aogpt_top12k_1g/legacy_tree/eval_per_task_stereo/full_best,NUM_SAMPLES=100,BATCH_SIZE=100,TEMPERATURE=1.0,TOP_K=0,TOP_P=1.0,DIVERSE_PREFIX_PENALTY=0.0,TASK_SAMPLING_JSON=data/fcstamp_per_task_sampling_stereo_best.json,MAX_NEW=64,SEED=20260525,STRIP_STEREO=0 \
  scripts/hpc/eval_fcstamp_genmol.pbs
```

## Next Training Target

The kept checkpoint is good enough for `linker_design` and `motif_extension`.

The next training target is:

```text
scaffold_decoration
superstructure_generation
```
