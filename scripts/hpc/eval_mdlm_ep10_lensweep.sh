#!/bin/bash
# Sweep max_new_tokens at eval time to see if shorter canvas recovers quality.
set -u
export RUN_DIR=/scratch/yuxuan.ren/data/ckpt_stamp/stamp_mdlm_30m_d512_l8_e3
export CKPT_PATH=${RUN_DIR}/checkpoints/epoch_0010/model.pt
export N=512
export STEPS=128
export TEMPERATURE=1.0
export TOP_K=50

SCRIPT=/home/svu/yuxuan.ren/STAMP_V1.5/scripts/hpc/eval_mdlm_ep1.sh

for MN in 16 20 24 32 48; do
  export MAX_NEW=$MN
  export TAG="ep10_mn${MN}"
  echo "==== max_new_tokens=${MN} ===="
  bash "$SCRIPT" 2>&1 | grep -E "len stats|validity|uniqueness|quality|GenMol|QED mean|first 5" | head -20
  echo ""
done
