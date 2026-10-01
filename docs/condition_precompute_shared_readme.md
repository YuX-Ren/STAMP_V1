# Condition Precompute

This folder is standalone and does not require the STAMP repository.

Shared path:

```bash
cd /Project_Storage/CFP-03/CFP03-CF-130/data/zinc_pretrain/code/condition_precompute
```

Smoke test:

```bash
MAX_FILES=1 MAX_ROWS_PER_FILE=100 OUT_DIR=/scratch/yuxuan.ren/data/zinc_pretrain/output_condition_qedsa_smoke qsub precompute_condition_tokens.pbs
```

Full run:

```bash
qsub precompute_condition_tokens.pbs
```

Default input:

```text
/Project_Storage/CFP-03/CFP03-CF-130/data/zinc_pretrain/output/*_stamp.parquet
```

Default output:

```text
/scratch/yuxuan.ren/data/zinc_pretrain/output_condition_qedsa
```

Added parquet columns:

- `qed`
- `sa`
- `condition_qed`
- `condition_sa`
- `condition_tokens`

The output parquet can be used for conditional training with:

```bash
STAMP_PARQUET='/scratch/yuxuan.ren/data/zinc_pretrain/output_condition_qedsa/*_stamp.parquet'
CONDITION_COLUMN=condition_tokens
CONDITION_MODE=qed_sa
```
