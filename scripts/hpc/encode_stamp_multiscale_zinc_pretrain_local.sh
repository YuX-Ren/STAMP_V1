#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${SHARED:=/Project_Storage/CFP-03/CFP03-CF-130/data/zinc_pretrain}"
: "${CONDA_ENV:=stamp-train-rdkit-2025_09_4}"
: "${CONDA_SH:=/scratch/yuxuan.ren/miniconda3/etc/profile.d/conda.sh}"

: "${MACHINE_ID:=0}"
: "${NUM_MACHINES:=1}"
: "${N_PARALLEL:=4}"
: "${WORKERS_PER:=8}"
: "${BATCH_SIZE:=20000}"

: "${IN_DIR:=${SHARED}/input}"
: "${OUT_DIR:=${SHARED}/output_multiscale}"
: "${SMILES_COLUMN:=smiles}"
: "${LOG_DIR:=${SHARED}/logs}"

: "${SCALE_VOCABS:=${SHARED}/vocab_merged/motif_vocab_top1k.txt,${SHARED}/vocab_merged/motif_vocab_top4k.txt,${SHARED}/vocab_merged/motif_vocab_top12k.txt}"
: "${SCALE_NAMES:=top1k,top4k,top12k}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

if [ -f "${CONDA_SH}" ]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/encode_multiscale_local_m${MACHINE_ID}.log"
TASK_FILE="/tmp/encode_multiscale_local_m${MACHINE_ID}_$$.txt"
trap 'rm -f "${TASK_FILE}"' EXIT

echo "[INFO] Host: $(hostname)" | tee -a "${LOG_FILE}"
echo "[INFO] Start: $(date)" | tee -a "${LOG_FILE}"
echo "[INFO] REPO_ROOT=${REPO_ROOT}" | tee -a "${LOG_FILE}"
echo "[INFO] MACHINE_ID=${MACHINE_ID}/${NUM_MACHINES} N_PARALLEL=${N_PARALLEL} WORKERS_PER=${WORKERS_PER}" | tee -a "${LOG_FILE}"
echo "[INFO] SCALE_NAMES=${SCALE_NAMES}" | tee -a "${LOG_FILE}"
echo "[INFO] IN_DIR=${IN_DIR}" | tee -a "${LOG_FILE}"
echo "[INFO] OUT_DIR=${OUT_DIR}" | tee -a "${LOG_FILE}"

find "${IN_DIR}" -maxdepth 1 -name '*.parquet' | sort | awk -v mid="${MACHINE_ID}" -v n="${NUM_MACHINES}" '
  ((NR-1) % n) == mid { print }
' | while read -r shard; do
  name="$(basename "${shard}" .parquet)"
  out="${OUT_DIR}/${name}_stamp_ms.parquet"
  [ ! -f "${out}" ] && echo "${shard} ${out}"
done > "${TASK_FILE}"

N_TASKS="$(wc -l < "${TASK_FILE}")"
echo "[INFO] Pending shards: ${N_TASKS}" | tee -a "${LOG_FILE}"

if [ "${N_TASKS}" -eq 0 ]; then
  echo "[INFO] Nothing to do; exiting." | tee -a "${LOG_FILE}"
  exit 0
fi

xargs -a "${TASK_FILE}" -L 1 -P "${N_PARALLEL}" bash -c '
  shard="$1"; out="$2"
  python -u scripts/benchmark/encode_one_smiles_multiscale_shard.py \
    --in-parquet "$shard" \
    --out-parquet "$out" \
    --smiles-column "'"${SMILES_COLUMN}"'" \
    --scale-vocabs "'"${SCALE_VOCABS}"'" \
    --scale-names "'"${SCALE_NAMES}"'" \
    --n-workers "'"${WORKERS_PER}"'" \
    --batch-size "'"${BATCH_SIZE}"'" 2>&1
' _ | tee -a "${LOG_FILE}"

DONE="$(find "${OUT_DIR}" -maxdepth 1 -name '*_stamp_ms.parquet' | wc -l)"
TOTAL="$(find "${IN_DIR}" -maxdepth 1 -name '*.parquet' | wc -l)"
echo "[INFO] End: $(date)" | tee -a "${LOG_FILE}"
echo "[INFO] Completed shards in ${OUT_DIR}: ${DONE}/${TOTAL}" | tee -a "${LOG_FILE}"
