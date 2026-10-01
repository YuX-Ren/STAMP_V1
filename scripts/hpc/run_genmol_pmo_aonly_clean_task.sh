#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-qed}"
SEED="${SEED:-1}"
BUDGET="${BUDGET:-10000}"
FREQ_LOG="${FREQ_LOG:-100}"
ENV_NAME="${ENV_NAME:-genmol-align}"
ROOT="/nfs/home/svu/yuxuan.ren/STAMP_V1.5"
REPO_DIR="${ROOT}/third_party/genmol-pmo-aonly"
OUT_DIR="${OUT_DIR:-${ROOT}/results/baselines/genmol_pmo_aonly_clean/${TASK}_seed${SEED}_budget${BUDGET}}"

mkdir -p "${OUT_DIR}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "[genmol-aonly-clean] host=$(hostname) task=${TASK} seed=${SEED} budget=${BUDGET}"

conda run -n "${ENV_NAME}" python -u scripts/exps/pmo/run.py \
  -o "${TASK}" \
  -s "${SEED}" \
  --disable-remask \
  --max_oracle_calls "${BUDGET}" \
  --freq_log "${FREQ_LOG}" \
  --output_dir "${OUT_DIR}"
