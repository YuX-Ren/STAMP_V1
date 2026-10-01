#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/nfs/home/svu/yuxuan.ren/STAMP_V1.5_pp}"
PROJECT_DOWNLOADS="${PROJECT_DOWNLOADS:-/Project_Storage/CFP-03/CFP03-CF-130/data/molformer_original/downloads}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/yuxuan.ren/data/molformer_original}"
SCRATCH_DOWNLOADS="${SCRATCH_DOWNLOADS:-${SCRATCH_ROOT}/downloads}"
SCRATCH_RAW="${SCRATCH_RAW:-${SCRATCH_ROOT}/raw}"

mkdir -p "${SCRATCH_DOWNLOADS}" "${SCRATCH_RAW}"
rsync -a "${PROJECT_DOWNLOADS}/" "${SCRATCH_DOWNLOADS}/"

cd "${REPO_ROOT}"
python scripts/benchmark/extract_molformer_original_archives.py \
  --downloads-dir "${SCRATCH_DOWNLOADS}" \
  --raw-dir "${SCRATCH_RAW}"

echo "[done] staged downloads to ${SCRATCH_DOWNLOADS}"
echo "[done] extracted PubChem/finetune data to ${SCRATCH_RAW}"
