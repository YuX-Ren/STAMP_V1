#!/bin/bash
# Run GenMol attaching-only PMO baselines in parallel across CPU workers.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=/nfs/home/svu/yuxuan.ren/STAMP_V1.5

: "${PARALLEL_TASKS:=8}"
: "${SEED:=1}"
: "${BUDGET:=10000}"
: "${POPULATION_SIZE:=100}"
: "${SAMPLE_POOL_SIZE:=1000}"
: "${SAMPLE_POOL_EXPAND_STEP:=100}"
: "${SAMPLE_POOL_EXPAND_EVERY:=50}"
: "${MAX_PAIR_REUSES:=1}"
: "${LOG_EVERY:=100}"
: "${SAMPLE_ATTEMPTS:=1000}"
: "${SKIP_IF_OUTPUT_EXISTS:=1}"
: "${TASKS:=drd2 gsk3b jnk3 celecoxib_rediscovery troglitazone_rediscovery thiothixene_rediscovery albuterol_similarity mestranol_similarity median1 median2 osimertinib_mpo fexofenadine_mpo ranolazine_mpo perindopril_mpo amlodipine_mpo sitagliptin_mpo zaleplon_mpo deco_hop scaffold_hop isomers_c7h8n2o2 isomers_c9h10n2o2pf2cl}"
: "${MASTER_LOG:=results/baselines/logs/genmol_attaching_exact_all_parallel.log}"

mkdir -p "${REPO_ROOT}/results/baselines/logs"
cd "${REPO_ROOT}"

echo "[genmol-A-all] host=$(hostname) date=$(date) parallel_tasks=${PARALLEL_TASKS}" | tee -a "${MASTER_LOG}"

active_jobs=0
for TASK in ${TASKS}; do
  echo "[genmol-A-all] launch task=${TASK} date=$(date)" | tee -a "${MASTER_LOG}"
  (
    TASK="${TASK}" \
    BUDGET="${BUDGET}" \
    SEED="${SEED}" \
    POPULATION_SIZE="${POPULATION_SIZE}" \
    SAMPLE_POOL_SIZE="${SAMPLE_POOL_SIZE}" \
    SAMPLE_POOL_EXPAND_STEP="${SAMPLE_POOL_EXPAND_STEP}" \
    SAMPLE_POOL_EXPAND_EVERY="${SAMPLE_POOL_EXPAND_EVERY}" \
    MAX_PAIR_REUSES="${MAX_PAIR_REUSES}" \
    LOG_EVERY="${LOG_EVERY}" \
    SAMPLE_ATTEMPTS="${SAMPLE_ATTEMPTS}" \
    SKIP_IF_OUTPUT_EXISTS="${SKIP_IF_OUTPUT_EXISTS}" \
    VOCAB_CSV="results/pmo_fragment_vocab/${TASK}.csv" \
    OUTPUT="results/baselines/genmol_attaching_exact_${TASK}_budget${BUDGET}_seed${SEED}_cpu.json" \
    LOG_FILE="results/baselines/logs/genmol_attaching_exact_${TASK}_budget${BUDGET}_seed${SEED}_cpu.log" \
    bash "${SCRIPT_DIR}/run_genmol_attaching_pmo_task.sh"
  ) >> "${MASTER_LOG}" 2>&1 &
  active_jobs=$((active_jobs + 1))
  if [[ "${active_jobs}" -ge "${PARALLEL_TASKS}" ]]; then
    wait -n
    active_jobs=$((active_jobs - 1))
  fi
done

wait
echo "[genmol-A-all] done date=$(date)" | tee -a "${MASTER_LOG}"
