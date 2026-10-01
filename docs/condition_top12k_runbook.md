# Condition Top12k Runbook

Shared code path:

```bash
cd /Project_Storage/CFP-03/CFP03-CF-130/data/zinc_pretrain/code/STAMP_V1.5_condition_top12k_20260521
```

## 1. Precompute Condition Columns

This does not re-encode STAMP. It copies existing parquet shards and adds:

- `qed`
- `sa`
- `condition_qed`
- `condition_sa`
- `condition_tokens`

```bash
qsub scripts/hpc/precompute_condition_tokens.pbs
```

Default output:

```text
/scratch/yuxuan.ren/data/zinc_pretrain/output_condition_qedsa
```

For a small smoke test:

```bash
MAX_FILES=1 MAX_ROWS_PER_FILE=100 OUT_DIR=/scratch/yuxuan.ren/data/zinc_pretrain/output_condition_qedsa_smoke qsub scripts/hpc/precompute_condition_tokens.pbs
```

## 2. Train Conditional AO-GPT

Use the precomputed parquet and the condition column:

```bash
STAMP_PARQUET='/scratch/yuxuan.ren/data/zinc_pretrain/output_condition_qedsa/*_stamp.parquet' \
CONDITION_MODE=qed_sa \
CONDITION_COLUMN=condition_tokens \
MODEL_TYPE=ao_gpt \
TRAIN_MAX_ROWS=10000000 \
EPOCHS=10 \
NPROC_PER_NODE=2 \
BATCH_SIZE=256 \
GLOBAL_BATCH_SIZE=1024 \
RUN_NAME=zinc_stamp_nodecomp_aogpt_10m_qedsa_top12k_ddp2_small \
qsub -l select=1:ngpus=2:ncpus=24:mem=450gb scripts/hpc/pretrain_zinc_stamp_nodecomp_ddp4.pbs
```

The training script uses:

- motif frequency file: `/Project_Storage/CFP-03/CFP03-CF-130/data/zinc_pretrain/vocab_merged/zinc_plus_augment_full_freq.txt`
- cutoff: `FREQ_CUTOFF=5000`
- top motif cap: `MAX_ATOMIC_MOTIFS=12000`
- random motif fallback enabled by default

## 3. Condition Grid Evaluation

After checkpoints exist:

```bash
MODEL_DIR=/scratch/yuxuan.ren/data/ckpt_stamp/zinc_stamp_nodecomp_aogpt_10m_qedsa_top12k_ddp2_small \
NUM_SAMPLES=1024 \
qsub scripts/hpc/eval_condition_grid.pbs
```

Outputs:

- `condition_grid_eval.json`
- `condition_grid_summary.tsv`
