#!/bin/bash
# Run one PMO task with the STAMP motif planner.
# If the offline STAMP bank is missing, build it first from ZINC250k.

set -euo pipefail

REPO_ROOT=/nfs/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4

: "${CUDA_DEVICE:=0}"
: "${TASK:=qed}"
: "${MODEL_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d512_l8_e10}"
: "${ZINC_PATH:=${REPO_ROOT}/data/zinc250k.csv}"
: "${BANK_DIR:=results/pmo_stamp_bank/${TASK}_full_mt}"
: "${BUILD_BANK_IF_MISSING:=1}"
: "${SKIP_IF_OUTPUT_EXISTS:=0}"
: "${BANK_WORKERS:=32}"
: "${BANK_CHUNK_SIZE:=2000}"
: "${BANK_UNIT_TYPES:=motif,subtree}"
: "${BANK_MAX_SUBTREE_GROUPS:=2}"
: "${BANK_MIN_SUPPORT:=3}"
: "${BANK_TOP_MOTIFS:=1000}"
: "${BANK_TOP_SUBTREES:=5000}"
: "${BANK_TOP_CONTEXTS:=0}"
: "${BANK_PROGRESS_EVERY:=25000}"
: "${BANK_TOP_UNITS:=1500}"
: "${BUDGET:=10000}"
: "${INIT_SAMPLES:=200}"
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
: "${LOG_EVERY:=100}"
: "${REPLACE_MIN_FRACTION:=0.625}"
: "${MAX_CONSECUTIVE_GROWS:=2}"
: "${LOW_CONTRIB_FRACTION:=0.5}"
: "${REPLACE_PRIOR_BONUS:=0.3}"
: "${OUTPUT:=results/baselines/stamp_motif_planner_pmo_${TASK}_bank_budget${BUDGET}_seed${SEED}_gpu_fast.json}"
: "${LOG_FILE:=results/baselines/logs/stamp_motif_planner_pmo_${TASK}_bank_budget${BUDGET}_seed${SEED}_gpu_fast.log}"

echo "[remote] host=$(hostname) date=$(date)"
echo "[remote] task=${TASK} cuda_device=${CUDA_DEVICE} budget=${BUDGET} seed=${SEED}"
echo "[remote] action_types=${ACTION_TYPES} explore_sample_rate=${EXPLORE_SAMPLE_RATE} refine_probability=${REFINE_PROBABILITY}"
echo "[remote] replace_min_fraction=${REPLACE_MIN_FRACTION} max_consecutive_grows=${MAX_CONSECUTIVE_GROWS} low_contrib_fraction=${LOW_CONTRIB_FRACTION} replace_prior_bonus=${REPLACE_PRIOR_BONUS}"

if [[ "${SKIP_IF_OUTPUT_EXISTS}" == "1" && -f "${OUTPUT}" ]]; then
  echo "[remote] skip_existing_output=1 output=${OUTPUT}"
  exit 0
fi

mkdir -p "${REPO_ROOT}/results/baselines/logs"
source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

if [[ "${BUILD_BANK_IF_MISSING}" == "1" && ! -f "${BANK_DIR}/summary.json" ]]; then
  echo "[remote] building bank for task=${TASK} at ${BANK_DIR}"
  SCORE_ARGS=()
  if [[ "${TASK}" == "qed" ]]; then
    SCORE_ARGS=(--score-column qed)
  fi
  python scripts/benchmark/build_stamp_pmo_bank.py \
    --input "${ZINC_PATH}" \
    --smiles-column smiles \
    --task "${TASK}" \
    "${SCORE_ARGS[@]}" \
    --output-dir "${BANK_DIR}" \
    --unit-types "${BANK_UNIT_TYPES}" \
    --max-subtree-groups "${BANK_MAX_SUBTREE_GROUPS}" \
    --num-workers "${BANK_WORKERS}" \
    --chunk-size "${BANK_CHUNK_SIZE}" \
    --min-support "${BANK_MIN_SUPPORT}" \
    --top-motifs "${BANK_TOP_MOTIFS}" \
    --top-subtrees "${BANK_TOP_SUBTREES}" \
    --top-contexts "${BANK_TOP_CONTEXTS}" \
    --progress-every "${BANK_PROGRESS_EVERY}"
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python scripts/benchmark/run_stamp_motif_planner_pmo.py \
  --model-dir "${MODEL_DIR}" \
  --task "${TASK}" \
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
