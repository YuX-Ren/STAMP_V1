#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/nfs/home/svu/yuxuan.ren/STAMP_V1.5_pp}"
NUM_SHARDS="${NUM_SHARDS:-128}"
SOURCE="${SOURCE:-all}"
PBS_SCRIPT="${PBS_SCRIPT:-${REPO_ROOT}/scripts/hpc/encode_molformer_original_stamp.pbs}"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  qsub -v "SHARD_INDEX=${i},NUM_SHARDS=${NUM_SHARDS},SOURCE=${SOURCE}" "${PBS_SCRIPT}"
done
