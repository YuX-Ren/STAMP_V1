#!/bin/bash
# Build one GenMol-style PMO fragment vocabulary from ZINC250k.

set -euo pipefail

REPO_ROOT=/nfs/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4

: "${TASK:=qed}"
: "${ZINC_PATH:=${REPO_ROOT}/data/zinc250k.csv}"
: "${VOCAB_CSV:=results/pmo_fragment_vocab/${TASK}.csv}"
: "${SKIP_IF_EXISTS:=1}"
: "${TOP_FRAGMENTS:=10000}"
: "${CUT_REPEATS:=3}"
: "${SEED:=1}"
: "${PROGRESS_EVERY:=1000}"
: "${NUM_WORKERS:=8}"
: "${CHUNKSIZE:=32}"
: "${LOG_FILE:=results/pmo_fragment_vocab/${TASK}.build.log}"

mkdir -p "${REPO_ROOT}/results/pmo_fragment_vocab"
source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

if [[ "${SKIP_IF_EXISTS}" == "1" && -f "${VOCAB_CSV}" ]]; then
  echo "[frag-task] task=${TASK} skip_existing=1 vocab=${VOCAB_CSV}"
  exit 0
fi

echo "[frag-task] host=$(hostname) date=$(date) task=${TASK} vocab=${VOCAB_CSV}"

SCORE_ARGS=()
if [[ "${TASK}" == "qed" ]]; then
  SCORE_ARGS=(--score-column qed)
fi

python -u scripts/benchmark/build_genmol_fragment_vocab.py \
  --task "${TASK}" \
  --input "${ZINC_PATH}" \
  --smiles-column smiles \
  "${SCORE_ARGS[@]}" \
  --output "${VOCAB_CSV}" \
  --top-fragments "${TOP_FRAGMENTS}" \
  --cut-repeats "${CUT_REPEATS}" \
  --seed "${SEED}" \
  --progress-every "${PROGRESS_EVERY}" \
  --num-workers "${NUM_WORKERS}" \
  --chunksize "${CHUNKSIZE}" \
  2>&1 | tee "${LOG_FILE}"

echo "[frag-task] end date=$(date) task=${TASK}"
