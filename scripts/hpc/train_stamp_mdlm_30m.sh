#!/bin/bash
# MDLM (masked diffusion LM) pretraining on 30M STAMP molecules.
# Small quick-test config: d=512 / l=8 / 3 epochs.
# Run directly on a GPU node (no PBS headers).
set -euo pipefail

: "${REPO_ROOT:=/home/svu/yuxuan.ren/STAMP_V1.5}"
: "${CONDA_ENV:=stamp-train-rdkit-2025_09_4}"
: "${PARQUET:=/scratch/yuxuan.ren/data/dataset_stamp.parquet}"
: "${STAMP_VOCAB:=/scratch/yuxuan.ren/data/v3_cm_union.txt}"
: "${CKPT_ROOT:=/scratch/yuxuan.ren/data/ckpt_stamp}"
: "${RUN_NAME:=stamp_mdlm_30m_d512_l8_e3}"
: "${OUTPUT_DIR:=${CKPT_ROOT}/${RUN_NAME}}"

: "${D_MODEL:=512}"
: "${N_HEADS:=8}"
: "${N_LAYERS:=8}"
: "${D_FF:=2048}"
: "${DROPOUT:=0.1}"
: "${MAX_SEQ_LEN:=64}"
: "${BATCH_SIZE:=4096}"
: "${GLOBAL_BATCH_SIZE:=4096}"
: "${LR:=5e-4}"
: "${WEIGHT_DECAY:=0.01}"
: "${EPOCHS:=3}"
: "${AMP:=bf16}"
: "${NUM_DATA_WORKERS:=8}"
: "${PREFETCH_FACTOR:=6}"
: "${TORCH_COMPILE:=0}"
: "${REUSE_SCAN_CACHE:=0}"
: "${NO_TQDM:=1}"

: "${DIFF_EPS:=1e-3}"
: "${DIFF_SCHEDULER:=linear}"
: "${DIFF_LOSS_WEIGHT:=scheduler}"

: "${RESUME_FROM:=}"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

mkdir -p "${OUTPUT_DIR}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] Host=$(hostname) Start=$(date)"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] Model: d=${D_MODEL} L=${N_LAYERS} H=${N_HEADS} ff=${D_FF} seq=${MAX_SEQ_LEN}"
echo "[INFO] Batch: micro=${BATCH_SIZE} global=${GLOBAL_BATCH_SIZE} epochs=${EPOCHS} amp=${AMP}"
echo "[INFO] LOG_FILE=${LOG_FILE}"
nvidia-smi || true

ARGS=(
  scripts/benchmark/train_stamp_diffusion_parquet.py
  --parquet           "${PARQUET}"
  --stamp-vocab       "${STAMP_VOCAB}"
  --output-dir        "${OUTPUT_DIR}"
  --train-max-rows    0
  --valid-max-rows    20000
  --epochs            "${EPOCHS}"
  --d-model           "${D_MODEL}"
  --n-heads           "${N_HEADS}"
  --n-layers          "${N_LAYERS}"
  --d-ff              "${D_FF}"
  --dropout           "${DROPOUT}"
  --max-seq-len       "${MAX_SEQ_LEN}"
  --batch-size        "${BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --lr                "${LR}"
  --weight-decay      "${WEIGHT_DECAY}"
  --amp               "${AMP}"
  --num-data-workers  "${NUM_DATA_WORKERS}"
  --prefetch-factor   "${PREFETCH_FACTOR}"
  --diffusion-time-epsilon "${DIFF_EPS}"
  --diffusion-scheduler    "${DIFF_SCHEDULER}"
  --diffusion-loss-weight  "${DIFF_LOSS_WEIGHT}"
  --device cuda:0
)
[[ "${NO_TQDM}" == "1" ]]          && ARGS+=(--no-tqdm)
[[ "${REUSE_SCAN_CACHE}" == "1" ]] && ARGS+=(--reuse-scan-cache)
[[ -n "${RESUME_FROM}" ]]          && ARGS+=(--resume-from "${RESUME_FROM}")
if [[ "${TORCH_COMPILE}" == "1" ]]; then
  ARGS+=(--torch-compile)
else
  ARGS+=(--no-torch-compile)
fi

python "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

echo "[INFO] End=$(date)"
echo "[INFO] ckpt=${OUTPUT_DIR}/model.pt  epochs at ${OUTPUT_DIR}/checkpoints/"
