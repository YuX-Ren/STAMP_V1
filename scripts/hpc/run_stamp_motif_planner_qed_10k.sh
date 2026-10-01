#!/bin/bash
# Run the 10k-call STAMP motif-planner QED baseline on the offline STAMP bank.
# Intended to be launched directly on hopper-32.

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

: "${CUDA_DEVICE:=0}"
: "${BUDGET:=10000}"
: "${INIT_SAMPLES:=200}"
: "${SEED:=1}"
: "${BANK_TOP_UNITS:=1500}"
: "${BEAM_SIZE:=4}"
: "${CANDIDATES_PER_STEP:=8}"
: "${PROPOSAL_SAMPLES:=4}"
: "${ROLLOUT_STEPS:=8}"
: "${POPULATION_SIZE:=128}"
: "${ACTION_TYPES:=grow,replace,shrink,refine}"
: "${EXPLORE_SAMPLE_RATE:=0.0}"
: "${REFINE_PROBABILITY:=0.08}"
: "${REFINE_PATCH_MAX_GROUPS:=2}"
: "${REFINE_MIN_CANDIDATES:=1}"
: "${REFINE_STEP_CANDIDATES:=2}"
: "${MAX_PLAN_ATTEMPTS:=1}"
: "${LOG_EVERY:=100}"
: "${REPLACE_MIN_FRACTION:=0.625}"
: "${MAX_CONSECUTIVE_GROWS:=2}"
: "${LOW_CONTRIB_FRACTION:=0.5}"
: "${REPLACE_PRIOR_BONUS:=0.3}"

export CUDA_DEVICE
export BUDGET
export INIT_SAMPLES
export SEED
export BANK_TOP_UNITS
export BEAM_SIZE
export CANDIDATES_PER_STEP
export PROPOSAL_SAMPLES
export ROLLOUT_STEPS
export POPULATION_SIZE
export ACTION_TYPES
export EXPLORE_SAMPLE_RATE
export REFINE_PROBABILITY
export REFINE_PATCH_MAX_GROUPS
export REFINE_MIN_CANDIDATES
export REFINE_STEP_CANDIDATES
export MAX_PLAN_ATTEMPTS
export LOG_EVERY
export REPLACE_MIN_FRACTION
export MAX_CONSECUTIVE_GROWS
export LOW_CONTRIB_FRACTION
export REPLACE_PRIOR_BONUS

bash "${SCRIPT_DIR}/run_stamp_motif_planner_qed_fast.sh"
