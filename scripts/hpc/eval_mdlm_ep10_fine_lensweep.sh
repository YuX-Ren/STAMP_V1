#!/bin/bash
# Fine sweep of max_new_tokens and (T, top_k) on ep10 ckpt.
# At each L, the model operates on an L-sized all-mask canvas (in-distribution
# for MDLM — training used variable-length sequences with no trailing pad).
set -u
export RUN_DIR=/scratch/yuxuan.ren/data/ckpt_stamp/stamp_mdlm_30m_d512_l8_e3
export CKPT_PATH=${RUN_DIR}/checkpoints/epoch_0010/model.pt
export N=512
export STEPS=128

SCRIPT=/home/svu/yuxuan.ren/STAMP_V1.5/scripts/hpc/eval_mdlm_ep1.sh

for MN in 12 14 16 18; do
  for T in 0.7 1.0; do
    for TK in 20 50; do
      export MAX_NEW=$MN TEMPERATURE=$T TOP_K=$TK
      export TAG="ep10_mn${MN}_t${T}_k${TK}"
      echo "==== L=${MN} T=${T} K=${TK} ===="
      bash "$SCRIPT" 2>&1 | grep -E "len stats|validity|uniqueness|quality|GenMol|QED mean" | head -10
      echo ""
    done
  done
done
