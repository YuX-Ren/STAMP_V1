#!/bin/bash
# Sweep STEPS/T/TOP_K on ep10 checkpoint to diagnose length/EOS issue.
set -u
export RUN_DIR=/scratch/yuxuan.ren/data/ckpt_stamp/stamp_mdlm_30m_d512_l8_e3
export CKPT_PATH=${RUN_DIR}/checkpoints/epoch_0010/model.pt
export N=512

SCRIPT=/home/svu/yuxuan.ren/STAMP_V1.5/scripts/hpc/eval_mdlm_ep1.sh

for STEPS in 64 128; do
  for T in 0.5 0.7 1.0; do
    for TK in 10 20 50; do
      export STEPS T TEMPERATURE=$T TOP_K=$TK
      export TAG="ep10_s${STEPS}_t${T}_k${TK}"
      echo "==== ${TAG} ===="
      bash "$SCRIPT" 2>&1 | grep -E "len stats|validity|uniqueness|quality|GenMol|QED mean|first 5" | head -20
      echo ""
    done
  done
done
