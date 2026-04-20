#!/bin/bash
#SBATCH --job-name=stamp_diff
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --requeue=expedite

set -euo pipefail

: "${REPO_ROOT:=/playpen-shared/mufan/github/ethanR/STAMP}"
: "${ENV_NAME:=stamp-train-rdkit-2025_09_4}"
: "${CONDA_BASE:=/playpen-shared/mufan/.miniforge3}"

source "${CONDA_BASE}/bin/activate"
conda activate "${ENV_NAME}"

cd "${REPO_ROOT}"
mkdir -p logs

echo "[INFO] ENV_NAME=${ENV_NAME}"
echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
echo "[INFO] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unknown}"

nvidia-smi || true
python - <<'PY'
from rdkit import rdBase
import torch
print(f"[INFO] rdkit={rdBase.rdkitVersion}")
print(f"[INFO] torch={torch.__version__}")
print(f"[INFO] cuda_available={torch.cuda.is_available()}")
PY

bash scripts/hpc/train_stamp_diffusion_datasets.sh
