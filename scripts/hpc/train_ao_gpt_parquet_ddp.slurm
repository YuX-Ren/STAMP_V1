#!/bin/bash
#SBATCH --job-name=ao_gpt_ddp
#SBATCH --partition=blackwell
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${REPO_ROOT:=/playpen-shared/mufan/github/ethanR/STAMP}"
: "${PYTHON_BIN:=/playpen-shared/mufan/.miniforge3/bin/python}"
: "${PARQUET:=/playpen-shared/mufan/github/ethanR/data/final/v_C_30M_v3_cm/dataset_stamp.parquet}"
: "${STAMP_VOCAB:=/playpen-shared/mufan/github/ethanR/data/data_for_stamp/vocab/v3_cm_union.txt}"
: "${OUTPUT_DIR:=/playpen-shared/mufan/github/ethanR/STAMP/results/slurm_ddp/v_C_30M_v3_cm/ao_bert_base_ddp}"
: "${TEXT_COLUMN:=stamp_smiles}"
: "${SPLIT_COLUMN:=split_random}"
: "${TRAIN_SPLIT:=train}"
: "${VALID_SPLIT:=val}"
: "${TRAIN_MAX_ROWS:=0}"
: "${VALID_MAX_ROWS:=20000}"
: "${EPOCHS:=10}"
: "${D_MODEL:=768}"
: "${N_HEADS:=12}"
: "${N_LAYERS:=12}"
: "${D_FF:=3072}"
: "${MAX_SEQ_LEN:=64}"
: "${BATCH_SIZE:=1024}"
: "${GLOBAL_BATCH_SIZE:=8192}"
: "${LR:=3e-4}"
: "${WEIGHT_DECAY:=0.01}"
: "${DROPOUT:=0.1}"
: "${AMP:=bf16}"
: "${NUM_DATA_WORKERS:=4}"
: "${PREFETCH_FACTOR:=4}"
: "${NO_TQDM:=1}"
: "${REUSE_SCAN_CACHE:=1}"
: "${SCAN_CACHE_SRC:=}"
: "${TORCH_COMPILE:=0}"
: "${OMP_NUM_THREADS:=4}"
: "${NCCL_DEBUG:=WARN}"
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"

mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS
export NCCL_DEBUG
export PYTORCH_CUDA_ALLOC_CONF
export TOKENIZERS_PARALLELISM=false

if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "[ERROR] This script must run under Slurm." >&2
  exit 1
fi

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
MASTER_PORT="$((12000 + (${SLURM_JOB_ID:-0} % 20000)))"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}"

mkdir -p "${OUTPUT_DIR}"
if [[ -n "${SCAN_CACHE_SRC}" && -f "${SCAN_CACHE_SRC}" && ! -f "${OUTPUT_DIR}/_scan_cache.pt" ]]; then
  cp "${SCAN_CACHE_SRC}" "${OUTPUT_DIR}/_scan_cache.pt"
fi

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] PYTHON_BIN=${PYTHON_BIN}"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
echo "[INFO] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unknown}"
echo "[INFO] MASTER_ADDR=${MASTER_ADDR}"
echo "[INFO] MASTER_PORT=${MASTER_PORT}"
echo "[INFO] NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] BATCH_SIZE=${BATCH_SIZE} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} EPOCHS=${EPOCHS}"

nvidia-smi || true
"${PYTHON_BIN}" - <<'PY'
import torch
from rdkit import rdBase
print(f"[INFO] torch={torch.__version__}")
print(f"[INFO] cuda_available={torch.cuda.is_available()}")
print(f"[INFO] cuda_device_count={torch.cuda.device_count()}")
print(f"[INFO] rdkit={rdBase.rdkitVersion}")
PY

ARGS=(
  scripts/benchmark/train_ao_gpt_parquet.py
  --parquet "${PARQUET}"
  --stamp-vocab "${STAMP_VOCAB}"
  --output-dir "${OUTPUT_DIR}"
  --text-column "${TEXT_COLUMN}"
  --split-column "${SPLIT_COLUMN}"
  --train-split "${TRAIN_SPLIT}"
  --valid-split "${VALID_SPLIT}"
  --train-max-rows "${TRAIN_MAX_ROWS}"
  --valid-max-rows "${VALID_MAX_ROWS}"
  --epochs "${EPOCHS}"
  --d-model "${D_MODEL}"
  --n-heads "${N_HEADS}"
  --n-layers "${N_LAYERS}"
  --d-ff "${D_FF}"
  --dropout "${DROPOUT}"
  --max-seq-len "${MAX_SEQ_LEN}"
  --batch-size "${BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --amp "${AMP}"
  --num-data-workers "${NUM_DATA_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
  --device cuda
)

if [[ "${NO_TQDM}" == "1" ]]; then
  ARGS+=(--no-tqdm)
fi
if [[ "${REUSE_SCAN_CACHE}" == "1" ]]; then
  ARGS+=(--reuse-scan-cache)
fi
if [[ "${TORCH_COMPILE}" == "0" ]]; then
  ARGS+=(--no-torch-compile)
else
  ARGS+=(--torch-compile)
fi

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  "${ARGS[@]}"
