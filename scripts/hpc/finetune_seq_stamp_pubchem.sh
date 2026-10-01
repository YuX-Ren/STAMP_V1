#!/bin/bash
# Sequential 9-task MoleculeNet fine-tune for the PubChem-pretrained STAMP RoBERTa.
# Run via:  nohup bash scripts/hpc/finetune_seq_stamp_pubchem.sh > run.log 2>&1 &
set -u

cd /home/svu/yuxuan.ren/STAMP_V1.5
source /scratch/yuxuan.ren/miniconda3/bin/activate stamp-train-rdkit-2025_09_4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRETRAIN="${PRETRAIN:-/scratch/yuxuan.ren/data/pubchem10m/pretrain_stamp_top4666}"
OUT="${OUT:-/scratch/yuxuan.ren/data/pubchem10m/finetune_stamp_top4666}"
DATA_DIR="${DATA_DIR:-/scratch/yuxuan.ren/data/moleculenet}"
TASKS="${TASKS:-freesolv esol clintox sider bace bbbp lipophilicity tox21 hiv}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-2e-5}"
BS="${BS:-32}"

mkdir -p "$OUT"

echo "PRETRAIN=$PRETRAIN"
echo "OUT=$OUT"
echo "DATA_DIR=$DATA_DIR"
echo "TASKS=$TASKS"

for task in $TASKS; do
  OUT_DIR="$OUT/$task"
  mkdir -p "$OUT_DIR"
  if [ -f "$OUT_DIR/finetune_results.json" ]; then
    echo "[SKIP] $task already done"
    continue
  fi
  echo
  echo "========== [$task] start: $(date) =========="

  time python scripts/experiments_cli.py finetune-roberta \
    --pretrained-dir "$PRETRAIN" \
    --task "$task" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUT_DIR" \
    --finetune-epochs "$EPOCHS" \
    --finetune-lr "$LR" \
    --finetune-batch-size "$BS" \
    --dropout 0.1 \
    --encode-workers 32 --encode-chunksize 64 \
    --num-data-workers 4 --prefetch-factor 4 \
    --amp bf16 \
    --device cuda \
    2>&1 | tee "$OUT_DIR/finetune.log"
  echo "[DONE] $task: $(date)"
done

echo
echo "==== ALL DONE: $(date) ===="
