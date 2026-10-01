#!/bin/bash
# Run a fast STAMP motif-planner PMO baseline on QED using the offline STAMP bank.
# Intended to be launched on hopper-32 (or similar) where CUDA is available.

set -u

REPO_ROOT=/nfs/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4

: "${CUDA_DEVICE:=0}"
: "${MODEL_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d512_l8_e10}"
: "${BANK_DIR:=results/pmo_stamp_bank/qed_full_mt}"
: "${BANK_TOP_UNITS:=1500}"
: "${BUDGET:=200}"
: "${INIT_SAMPLES:=40}"
: "${SEED:=1}"
: "${ACTION_TYPES:=grow,replace,shrink,refine}"
: "${BEAM_SIZE:=4}"
: "${CANDIDATES_PER_STEP:=8}"
: "${PROPOSAL_SAMPLES:=4}"
: "${ROLLOUT_STEPS:=8}"
: "${POPULATION_SIZE:=128}"
: "${EXPLORE_SAMPLE_RATE:=0.0}"
: "${REFINE_PROBABILITY:=0.08}"
: "${REFINE_PATCH_MAX_GROUPS:=2}"
: "${REFINE_MIN_CANDIDATES:=1}"
: "${REFINE_STEP_CANDIDATES:=2}"
: "${MAX_PLAN_ATTEMPTS:=1}"
: "${LOG_EVERY:=20}"
: "${REPLACE_MIN_FRACTION:=0.5}"
: "${MAX_CONSECUTIVE_GROWS:=4}"
: "${LOW_CONTRIB_FRACTION:=0.35}"
: "${REPLACE_PRIOR_BONUS:=0.2}"
: "${OUTPUT:=results/baselines/stamp_motif_planner_pmo_qed_bank_budget${BUDGET}_seed${SEED}_gpu_fast.json}"
: "${LOG_FILE:=results/baselines/logs/stamp_motif_planner_pmo_qed_bank_budget${BUDGET}_seed${SEED}_gpu_fast.log}"

echo "[remote] host=$(hostname) date=$(date)"
echo "[remote] cuda_device=${CUDA_DEVICE} budget=${BUDGET} seed=${SEED}"
echo "[remote] action_types=${ACTION_TYPES} refine_probability=${REFINE_PROBABILITY} refine_patch_max_groups=${REFINE_PATCH_MAX_GROUPS} refine_min_candidates=${REFINE_MIN_CANDIDATES} refine_step_candidates=${REFINE_STEP_CANDIDATES}"
echo "[remote] replace_min_fraction=${REPLACE_MIN_FRACTION} max_consecutive_grows=${MAX_CONSECUTIVE_GROWS} low_contrib_fraction=${LOW_CONTRIB_FRACTION} replace_prior_bonus=${REPLACE_PRIOR_BONUS}"

mkdir -p "${REPO_ROOT}/results/baselines/logs"
source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python scripts/benchmark/run_stamp_motif_planner_pmo.py \
  --model-dir "${MODEL_DIR}" \
  --task qed \
  --bank-dir "${BANK_DIR}" \
  --bank-top-units "${BANK_TOP_UNITS}" \
  --budget "${BUDGET}" \
  --init-samples "${INIT_SAMPLES}" \
  --seed "${SEED}" \
  --device cuda \
  --output "${OUTPUT}" \
  --beam-size "${BEAM_SIZE}" \
  --candidates-per-step "${CANDIDATES_PER_STEP}" \
  --action-types "${ACTION_TYPES}" \
  --refine-probability "${REFINE_PROBABILITY}" \
  --refine-patch-max-groups "${REFINE_PATCH_MAX_GROUPS}" \
  --refine-min-candidates "${REFINE_MIN_CANDIDATES}" \
  --refine-step-candidates "${REFINE_STEP_CANDIDATES}" \
  --proposal-samples "${PROPOSAL_SAMPLES}" \
  --rollout-steps "${ROLLOUT_STEPS}" \
  --population-size "${POPULATION_SIZE}" \
  --explore-sample-rate "${EXPLORE_SAMPLE_RATE}" \
  --replace-min-fraction "${REPLACE_MIN_FRACTION}" \
  --max-consecutive-grows "${MAX_CONSECUTIVE_GROWS}" \
  --low-contrib-fraction "${LOW_CONTRIB_FRACTION}" \
  --replace-prior-bonus "${REPLACE_PRIOR_BONUS}" \
  --max-plan-attempts-per-step "${MAX_PLAN_ATTEMPTS}" \
  --log-every "${LOG_EVERY}" \
  2>&1 | tee "${LOG_FILE}"

echo "[remote] end=$(date)"
