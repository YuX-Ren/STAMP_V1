#!/bin/bash
# Build all PMO offline STAMP banks in parallel.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=/nfs/home/svu/yuxuan.ren/STAMP_V1.5

: "${PARALLEL_TASKS:=8}"
: "${BANK_WORKERS:=12}"
: "${TASKS:=qed drd2 gsk3b jnk3 celecoxib_rediscovery troglitazone_rediscovery thiothixene_rediscovery albuterol_similarity mestranol_similarity median1 median2 osimertinib_mpo fexofenadine_mpo ranolazine_mpo perindopril_mpo amlodipine_mpo sitagliptin_mpo zaleplon_mpo valsartan_smarts deco_hop scaffold_hop isomers_c7h8n2o2 isomers_c9h10n2o2pf2cl}"
: "${MASTER_LOG:=results/pmo_stamp_bank/logs/build_all_parallel.log}"

mkdir -p "${REPO_ROOT}/results/pmo_stamp_bank/logs"
cd "${REPO_ROOT}"

echo "[bank-all] host=$(hostname) date=$(date) parallel_tasks=${PARALLEL_TASKS} bank_workers=${BANK_WORKERS}" | tee -a "${MASTER_LOG}"

active_jobs=0
for TASK in ${TASKS}; do
  echo "[bank-all] launch task=${TASK} date=$(date)" | tee -a "${MASTER_LOG}"
  (
    TASK="${TASK}" \
    BANK_WORKERS="${BANK_WORKERS}" \
    bash "${SCRIPT_DIR}/build_stamp_pmo_bank_task.sh"
  ) >> "${MASTER_LOG}" 2>&1 &
  active_jobs=$((active_jobs + 1))
  if [[ "${active_jobs}" -ge "${PARALLEL_TASKS}" ]]; then
    wait -n
    active_jobs=$((active_jobs - 1))
  fi
done

wait
echo "[bank-all] done date=$(date)" | tee -a "${MASTER_LOG}"
