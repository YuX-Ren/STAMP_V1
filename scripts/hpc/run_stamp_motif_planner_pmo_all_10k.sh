#!/bin/bash
# Sequentially run all 23 PMO benchmark tasks with the STAMP motif planner.
# Banks are built lazily per task if missing.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

: "${CUDA_DEVICE:=0}"
: "${SEED:=1}"
: "${BUDGET:=10000}"
: "${INIT_SAMPLES:=200}"
: "${TASKS:=qed drd2 gsk3b jnk3 celecoxib_rediscovery troglitazone_rediscovery thiothixene_rediscovery albuterol_similarity mestranol_similarity median1 median2 osimertinib_mpo fexofenadine_mpo ranolazine_mpo perindopril_mpo amlodipine_mpo sitagliptin_mpo zaleplon_mpo valsartan_smarts deco_hop scaffold_hop isomers_c7h8n2o2 isomers_c9h10n2o2pf2cl}"

for TASK in ${TASKS}; do
  echo "[all] starting task=${TASK} date=$(date)"
  CUDA_DEVICE="${CUDA_DEVICE}" \
  SEED="${SEED}" \
  BUDGET="${BUDGET}" \
  INIT_SAMPLES="${INIT_SAMPLES}" \
  TASK="${TASK}" \
  bash "${SCRIPT_DIR}/run_stamp_motif_planner_pmo_task_fast.sh"
  echo "[all] finished task=${TASK} date=$(date)"
done
