#!/bin/bash
# Build one offline STAMP PMO bank from ZINC250k.

set -euo pipefail

REPO_ROOT=/nfs/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4

: "${TASK:=qed}"
: "${ZINC_PATH:=${REPO_ROOT}/data/zinc250k.csv}"
: "${BANK_DIR:=results/pmo_stamp_bank/${TASK}_full_mt}"
: "${SKIP_IF_EXISTS:=1}"
: "${BANK_WORKERS:=12}"
: "${BANK_CHUNK_SIZE:=2000}"
: "${BANK_UNIT_TYPES:=motif,subtree}"
: "${BANK_MAX_SUBTREE_GROUPS:=2}"
: "${BANK_MIN_SUPPORT:=3}"
: "${BANK_TOP_MOTIFS:=1000}"
: "${BANK_TOP_SUBTREES:=5000}"
: "${BANK_TOP_CONTEXTS:=0}"
: "${BANK_PROGRESS_EVERY:=25000}"
: "${LOG_FILE:=results/pmo_stamp_bank/logs/${TASK}_bank_build.log}"

mkdir -p "${REPO_ROOT}/results/pmo_stamp_bank/logs"
source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
export TOKENIZERS_PARALLELISM=false

if [[ "${SKIP_IF_EXISTS}" == "1" && -f "${BANK_DIR}/summary.json" ]]; then
  echo "[bank-task] task=${TASK} skip_existing=1 bank_dir=${BANK_DIR}"
  exit 0
fi

echo "[bank-task] host=$(hostname) date=$(date) task=${TASK} workers=${BANK_WORKERS} bank_dir=${BANK_DIR}"

SCORE_ARGS=()
if [[ "${TASK}" == "qed" ]]; then
  SCORE_ARGS=(--score-column qed)
fi

python scripts/benchmark/build_stamp_pmo_bank.py \
  --input "${ZINC_PATH}" \
  --smiles-column smiles \
  --task "${TASK}" \
  "${SCORE_ARGS[@]}" \
  --output-dir "${BANK_DIR}" \
  --unit-types "${BANK_UNIT_TYPES}" \
  --max-subtree-groups "${BANK_MAX_SUBTREE_GROUPS}" \
  --num-workers "${BANK_WORKERS}" \
  --chunk-size "${BANK_CHUNK_SIZE}" \
  --min-support "${BANK_MIN_SUPPORT}" \
  --top-motifs "${BANK_TOP_MOTIFS}" \
  --top-subtrees "${BANK_TOP_SUBTREES}" \
  --top-contexts "${BANK_TOP_CONTEXTS}" \
  --progress-every "${BANK_PROGRESS_EVERY}" \
  2>&1 | tee "${LOG_FILE}"

echo "[bank-task] end date=$(date) task=${TASK}"
