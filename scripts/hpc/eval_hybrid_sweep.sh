#!/bin/bash
# Sampling-hyperparameter sweep on hybrid ep9.
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
RUN_DIR=/scratch/yuxuan.ren/data/ckpt_stamp/stamp_hybrid_30m_d512_l8_e10_cut5k
export RUN_DIR
export EPOCH=${EPOCH:-0009}
export N=${N:-1024}

SCRIPT=/home/svu/yuxuan.ren/STAMP_V1.5/scripts/hpc/eval_hybrid_ep10.sh

for T in 0.7 0.8 0.9 1.0; do
  for K in 20 50 100; do
    export TEMPERATURE=$T TOP_K=$K
    export TAG="ep${EPOCH}_t${T}_k${K}"
    bash "$SCRIPT" 2>&1 | grep -E "validity|uniqueness|quality|GenMol|QED|seq_len" | head -6
    echo "  -> T=$T K=$K"
    echo ""
  done
done
