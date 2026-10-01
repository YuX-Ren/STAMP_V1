#!/bin/bash
# Launch planner-only PMO benchmark shards across multiple GPUs via tmux.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=/nfs/home/svu/yuxuan.ren/STAMP_V1.5

: "${GPU_LIST:=0,2,3,4,5,6,7}"
: "${SEED:=1}"
: "${BUDGET:=10000}"
: "${INIT_SAMPLES:=200}"
: "${SKIP_IF_OUTPUT_EXISTS:=1}"
: "${KILL_EXISTING_SHARDS:=1}"
: "${TASKS:=qed drd2 gsk3b jnk3 celecoxib_rediscovery troglitazone_rediscovery thiothixene_rediscovery albuterol_similarity mestranol_similarity median1 median2 osimertinib_mpo fexofenadine_mpo ranolazine_mpo perindopril_mpo amlodipine_mpo sitagliptin_mpo zaleplon_mpo valsartan_smarts deco_hop scaffold_hop isomers_c7h8n2o2 isomers_c9h10n2o2pf2cl}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
read -r -a TASK_ARR <<< "${TASKS}"

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_LIST is empty" >&2
  exit 1
fi

declare -a SHARDS
for ((i=0; i<${#GPUS[@]}; i++)); do
  SHARDS[i]=""
done

for ((i=0; i<${#TASK_ARR[@]}; i++)); do
  shard_idx=$((i % ${#GPUS[@]}))
  SHARDS[shard_idx]="${SHARDS[shard_idx]} ${TASK_ARR[i]}"
done

cd "${REPO_ROOT}"
for ((i=0; i<${#GPUS[@]}; i++)); do
  gpu="${GPUS[i]}"
  tasks="${SHARDS[i]# }"
  session="stamp_pmo_planner_g${gpu}"
  if [[ -z "${tasks}" ]]; then
    continue
  fi
  if [[ "${KILL_EXISTING_SHARDS}" == "1" ]]; then
    tmux kill-session -t "${session}" >/dev/null 2>&1 || true
  fi
  echo "[launch] gpu=${gpu} session=${session} tasks=${tasks}"
  tmux new-session -d -s "${session}" \
    "cd ${REPO_ROOT} && env CUDA_DEVICE=${gpu} SEED=${SEED} BUDGET=${BUDGET} INIT_SAMPLES=${INIT_SAMPLES} SKIP_IF_OUTPUT_EXISTS=${SKIP_IF_OUTPUT_EXISTS} TASKS='${tasks}' bash ${SCRIPT_DIR}/run_stamp_motif_planner_pmo_all_10k_planner_only.sh"
done

tmux ls
