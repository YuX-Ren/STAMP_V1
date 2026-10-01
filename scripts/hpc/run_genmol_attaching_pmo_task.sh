#!/bin/bash
# Run one GenMol attaching-only PMO baseline task using a prebuilt fragment vocab.

set -euo pipefail

REPO_ROOT=/nfs/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4

: "${TASK:=qed}"
: "${BUDGET:=10000}"
: "${SEED:=1}"
: "${POPULATION_SIZE:=100}"
: "${SAMPLE_POOL_SIZE:=1000}"
: "${SAMPLE_POOL_EXPAND_STEP:=100}"
: "${SAMPLE_POOL_EXPAND_EVERY:=50}"
: "${MAX_PAIR_REUSES:=1}"
: "${LOG_EVERY:=100}"
: "${SAMPLE_ATTEMPTS:=1000}"
: "${STALL_LOG_EVERY:=50}"
: "${MAX_NO_PROGRESS_ITERS:=2000}"
: "${VOCAB_CSV:=results/pmo_fragment_vocab/${TASK}.csv}"
: "${OUTPUT:=results/baselines/genmol_attaching_exact_${TASK}_budget${BUDGET}_seed${SEED}_cpu.json}"
: "${LOG_FILE:=results/baselines/logs/genmol_attaching_exact_${TASK}_budget${BUDGET}_seed${SEED}_cpu.log}"
: "${SKIP_IF_OUTPUT_EXISTS:=1}"

mkdir -p "${REPO_ROOT}/results/baselines/logs"
source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

echo "[genmol-A-task] host=$(hostname) date=$(date) task=${TASK} budget=${BUDGET} seed=${SEED}"
echo "[genmol-A-task] vocab=${VOCAB_CSV} output=${OUTPUT}"

if [[ "${SKIP_IF_OUTPUT_EXISTS}" == "1" && -f "${OUTPUT}" ]]; then
  echo "[genmol-A-task] skip_existing_output=1 output=${OUTPUT}"
  exit 0
fi

python -u scripts/benchmark/run_genmol_attaching_pmo.py \
  --task "${TASK}" \
  --vocab-csv "${VOCAB_CSV}" \
  --budget "${BUDGET}" \
  --seed "${SEED}" \
  --population-size "${POPULATION_SIZE}" \
  --sample-pool-size "${SAMPLE_POOL_SIZE}" \
  --sample-pool-expand-step "${SAMPLE_POOL_EXPAND_STEP}" \
  --sample-pool-expand-every "${SAMPLE_POOL_EXPAND_EVERY}" \
  --max-pair-reuses "${MAX_PAIR_REUSES}" \
  --log-every "${LOG_EVERY}" \
  --sample-attempts "${SAMPLE_ATTEMPTS}" \
  --stall-log-every "${STALL_LOG_EVERY}" \
  --max-no-progress-iters "${MAX_NO_PROGRESS_ITERS}" \
  --output "${OUTPUT}" \
  2>&1 | tee "${LOG_FILE}"

echo "[genmol-A-task] end date=$(date) task=${TASK}"
