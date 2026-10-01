#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${CONDA_ENV:=stamp-train-rdkit-2025_09_4}"
: "${CONDA_SH:=/scratch/yuxuan.ren/miniconda3/etc/profile.d/conda.sh}"

: "${DATA_DIR:=/Project_Storage/CFP-03/CFP03-CF-130/data/zinc_pretrain/output_multiscale}"
: "${VOCAB_DIR:=/Project_Storage/CFP-03/CFP03-CF-130/data/zinc_pretrain/vocab_merged}"
: "${OUT_DIR:=/Project_Storage/CFP-03/CFP03-CF-130/ckpt/cams_stamp_decoder_multiscale_3s_zinc}"

: "${TRAIN_GLOB:=${DATA_DIR}/train_*_ms.parquet}"
: "${VALID_GLOB:=${DATA_DIR}/val_*_ms.parquet}"
: "${TEXT_COLUMN:=stamp}"

: "${MULTISCALE_VOCABS:=${VOCAB_DIR}/motif_vocab_top1k.txt,${VOCAB_DIR}/motif_vocab_top4k.txt,${VOCAB_DIR}/motif_vocab_top12k.txt}"
: "${MULTISCALE_SCALE_NAMES:=top1k,top4k,top12k}"
: "${EXPECTED_SINGLE_SCALE_VIEWS:=3}"

: "${D_MODEL:=720}"
: "${N_HEADS:=8}"
: "${N_LAYERS:=16}"
: "${D_FF:=2880}"
: "${DROPOUT:=0.0}"
: "${MAX_SEQ_LEN:=4096}"
: "${BATCH_SIZE:=128}"
: "${GLOBAL_BATCH_SIZE:=4096}"
: "${LR:=3e-4}"
: "${MIN_LR:=0.0}"
: "${WEIGHT_DECAY:=0.01}"
: "${LR_SCHEDULER:=cosine}"
: "${WARMUP_STEPS:=5000}"
: "${EPOCHS:=1}"
: "${TRAIN_MAX_ROWS:=0}"
: "${VALID_MAX_ROWS:=20000}"
: "${SCAN_TRAIN_MAX_ROWS:=200000}"
: "${SCAN_VALID_MAX_ROWS:=20000}"
: "${NUM_DATA_WORKERS:=4}"
: "${PREFETCH_FACTOR:=4}"
: "${AMP:=auto}"
: "${DEVICE:=cuda}"
: "${SEED:=42}"
: "${TRAIN_VIEW_MODE:=cams_multiview}"
: "${VALID_VIEW_MODE:=concat_only}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

if [ -f "${CONDA_SH}" ]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi
if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "${OUT_DIR}"
cd "${REPO_ROOT}"

if [ ! -d "${DATA_DIR}" ]; then
  echo "ERROR: DATA_DIR does not exist on this machine: ${DATA_DIR}" >&2
  echo "For hopper qsub, /Project_Storage is not mounted; stage data to /scratch or run on a node that can see DATA_DIR." >&2
  exit 2
fi

python -u scripts/benchmark/train_cams_decoder_shards.py \
  --train-glob "${TRAIN_GLOB}" \
  --valid-glob "${VALID_GLOB}" \
  --stamp-vocab "${VOCAB_DIR}/motif_vocab.txt" \
  --output-dir "${OUT_DIR}" \
  --text-column "${TEXT_COLUMN}" \
  --multiscale-vocabs "${MULTISCALE_VOCABS}" \
  --multiscale-scale-names "${MULTISCALE_SCALE_NAMES}" \
  --train-view-mode "${TRAIN_VIEW_MODE}" \
  --valid-view-mode "${VALID_VIEW_MODE}" \
  --expected-single-scale-views "${EXPECTED_SINGLE_SCALE_VIEWS}" \
  --d-model "${D_MODEL}" \
  --n-heads "${N_HEADS}" \
  --n-layers "${N_LAYERS}" \
  --d-ff "${D_FF}" \
  --dropout "${DROPOUT}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --lr "${LR}" \
  --min-lr "${MIN_LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --lr-scheduler "${LR_SCHEDULER}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --epochs "${EPOCHS}" \
  --train-max-rows "${TRAIN_MAX_ROWS}" \
  --valid-max-rows "${VALID_MAX_ROWS}" \
  --scan-train-max-rows "${SCAN_TRAIN_MAX_ROWS}" \
  --scan-valid-max-rows "${SCAN_VALID_MAX_ROWS}" \
  --seed "${SEED}" \
  --num-data-workers "${NUM_DATA_WORKERS}" \
  --prefetch-factor "${PREFETCH_FACTOR}" \
  --amp "${AMP}" \
  --device "${DEVICE}" \
  --reuse-scan-cache
