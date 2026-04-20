#!/usr/bin/env bash
#DATASETS=zinc250k bash scripts/hpc/train_stamp_diffusion_datasets.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

: "${RESULTS_ROOT:=results/stamp_diffusion_$(date +%Y%m%d_%H%M%S)}"
: "${DATASETS:=moses guacamol zinc250k qm9}"

: "${PSVAE_VOCAB_SIZE:=500}"
: "${PSVAE_VOCAB_WORKERS:=8}"
: "${PSVAE_VOCAB_MAX_MOLS:=0}"
: "${ENCODE_WORKERS:=16}"
: "${ENCODE_CHUNKSIZE:=128}"
: "${NUM_DATA_WORKERS:=4}"
: "${PREFETCH_FACTOR:=4}"

: "${D_MODEL:=256}"
: "${N_HEADS:=8}"
: "${N_LAYERS:=4}"
: "${D_FF:=1024}"
: "${DROPOUT:=0.1}"
: "${MAX_SEQ_LEN:=256}"
: "${BATCH_SIZE:=128}"
: "${LR:=3e-4}"
: "${WEIGHT_DECAY:=0.01}"
: "${EPOCHS:=10}"
: "${SEED:=42}"

: "${DIFFUSION_STEPS:=128}"
: "${DIFFUSION_TOKENS_PER_STEP:=1}"
: "${DIFFUSION_TIME_EPSILON:=1e-3}"
: "${DIFFUSION_SCHEDULER:=linear}"
: "${DIFFUSION_LOSS_WEIGHT:=scheduler}"
: "${DIFFUSION_REMASKING:=low_confidence}"
: "${DIFFUSION_BLOCK_SIZE:=0}"
: "${EVAL_N_SAMPLES:=10000}"
: "${EVAL_DECODE_BATCH_SIZE:=512}"
: "${EVAL_WORKERS:=8}"
: "${EVAL_CHUNKSIZE:=256}"
: "${REFERENCE_MAX_SAMPLES:=50000}"

: "${ZINC_TRAIN_SIZE:=200000}"
: "${ZINC_VALID_SIZE:=20000}"
: "${ZINC_TEST_SIZE:=20000}"

: "${QM9_TRAIN_SIZE:=100000}"
: "${QM9_VALID_SIZE:=10000}"
: "${QM9_TEST_SIZE:=10000}"

mkdir -p "${RESULTS_ROOT}"

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[INFO] DATASETS=${DATASETS}"
python - <<'PY'
from rdkit import rdBase
import torch
print(f"[INFO] rdkit={rdBase.rdkitVersion}")
print(f"[INFO] torch={torch.__version__}")
print(f"[INFO] cuda_available={torch.cuda.is_available()}")
PY

prepare_split_dataset() {
  local dataset_name="$1"
  local input_path="$2"
  local smiles_column="$3"
  local train_size="$4"
  local valid_size="$5"
  local test_size="$6"
  local prep_dir="${RESULTS_ROOT}/prep/${dataset_name}"

  if [[ -f "${prep_dir}/splits/train.smiles" && -f "${prep_dir}/splits/valid.smiles" ]]; then
    echo "[INFO] Reusing prepared splits for ${dataset_name}: ${prep_dir}/splits"
    return
  fi

  mkdir -p "${prep_dir}"
  python scripts/benchmark/prepare_smiles_splits.py \
    --input "${input_path}" \
    --smiles-column "${smiles_column}" \
    --output-dir "${prep_dir}" \
    --train-size "${train_size}" \
    --valid-size "${valid_size}" \
    --test-size "${test_size}" \
    --low-resource-sizes 10000,50000,100000
}

build_vocab_for_dataset() {
  local dataset_name="$1"
  local smiles_column="$2"
  local train_path="$3"
  local vocab_file="${RESULTS_ROOT}/vocab/${dataset_name}/psvae_vocab.txt"

  if [[ -f "${vocab_file}" ]]; then
    echo "[INFO] Reusing PS-VAE vocab for ${dataset_name}: ${vocab_file}"
    return
  fi

  mkdir -p "$(dirname "${vocab_file}")"
  local cmd=(
    python scripts/experiments_cli.py mine-vocab
    --input "${train_path}"
    --smiles-column "${smiles_column}"
    --output "${vocab_file}"
    --vocab-size "${PSVAE_VOCAB_SIZE}"
    --workers "${PSVAE_VOCAB_WORKERS}"
  )
  if [[ "${PSVAE_VOCAB_MAX_MOLS}" -gt 0 ]]; then
    cmd+=(--max-mols "${PSVAE_VOCAB_MAX_MOLS}")
  fi
  "${cmd[@]}"
}

train_diffusion_for_dataset() {
  local dataset_name="$1"
  local train_path="$2"
  local valid_path="$3"
  shift 3
  local vocab_fit_paths=("$@")
  local vocab_file="${RESULTS_ROOT}/vocab/${dataset_name}/psvae_vocab.txt"
  local model_dir="${RESULTS_ROOT}/models/${dataset_name}_stamp_diffusion"

  if [[ -f "${model_dir}/train_meta.json" ]]; then
    echo "[INFO] Reusing trained model for ${dataset_name}: ${model_dir}"
    return
  fi

  mkdir -p "${model_dir}"
  local cmd=(
    python scripts/benchmark/train_stamp_diffusion.py
    --train "${train_path}"
    --valid "${valid_path}"
    --output-dir "${model_dir}"
    --representation stamp
    --stamp-vocab "${vocab_file}"
    --encode-workers "${ENCODE_WORKERS}"
    --encode-chunksize "${ENCODE_CHUNKSIZE}"
    --d-model "${D_MODEL}"
    --n-heads "${N_HEADS}"
    --n-layers "${N_LAYERS}"
    --d-ff "${D_FF}"
    --dropout "${DROPOUT}"
    --max-seq-len "${MAX_SEQ_LEN}"
    --batch-size "${BATCH_SIZE}"
    --lr "${LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --epochs "${EPOCHS}"
    --seed "${SEED}"
    --num-data-workers "${NUM_DATA_WORKERS}"
    --prefetch-factor "${PREFETCH_FACTOR}"
    --amp auto
    --torch-compile
    --diffusion-steps "${DIFFUSION_STEPS}"
    --diffusion-tokens-per-step "${DIFFUSION_TOKENS_PER_STEP}"
    --diffusion-time-epsilon "${DIFFUSION_TIME_EPSILON}"
    --diffusion-scheduler "${DIFFUSION_SCHEDULER}"
    --diffusion-loss-weight "${DIFFUSION_LOSS_WEIGHT}"
    --diffusion-remasking "${DIFFUSION_REMASKING}"
    --diffusion-block-size "${DIFFUSION_BLOCK_SIZE}"
    --device cuda
  )
  for p in "${vocab_fit_paths[@]}"; do
    cmd+=(--vocab-fit "$p")
  done
  "${cmd[@]}"
}

evaluate_diffusion_for_dataset() {
  local dataset_name="$1"
  local train_path="$2"
  local reference_path="$3"
  local model_dir="${RESULTS_ROOT}/models/${dataset_name}_stamp_diffusion"
  local eval_dir="${RESULTS_ROOT}/eval/${dataset_name}_stamp_diffusion"

  if [[ -f "${eval_dir}/denovo_metrics.json" ]]; then
    echo "[INFO] Reusing evaluation for ${dataset_name}: ${eval_dir}"
    return
  fi

  mkdir -p "${eval_dir}"
  python scripts/benchmark/eval_stamp_diffusion.py \
    --model-dir "${model_dir}" \
    --train "${train_path}" \
    --reference "${reference_path}" \
    --output-dir "${eval_dir}" \
    --n-samples "${EVAL_N_SAMPLES}" \
    --decode-batch-size "${EVAL_DECODE_BATCH_SIZE}" \
    --eval-workers "${EVAL_WORKERS}" \
    --eval-chunksize "${EVAL_CHUNKSIZE}" \
    --reference-max-samples "${REFERENCE_MAX_SAMPLES}" \
    --assume-train-canonical \
    --diffusion-steps "${DIFFUSION_STEPS}" \
    --diffusion-tokens-per-step "${DIFFUSION_TOKENS_PER_STEP}" \
    --diffusion-block-size "${DIFFUSION_BLOCK_SIZE}" \
    --diffusion-remasking "${DIFFUSION_REMASKING}" \
    --device cuda
}

run_moses() {
  local train_path="data/moses/train.smiles"
  local valid_path="data/moses/test.smiles"
  build_vocab_for_dataset "moses" "smiles" "${train_path}"
  train_diffusion_for_dataset "moses" "${train_path}" "${valid_path}" \
    "data/moses/train.smiles" "data/moses/test.smiles" "data/moses/test_scaffolds.smiles"
  evaluate_diffusion_for_dataset "moses" "${train_path}" "data/moses/test.smiles"
}

run_guacamol() {
  local train_path="data/guacamol/train.smiles"
  local valid_path="data/guacamol/valid.smiles"
  build_vocab_for_dataset "guacamol" "smiles" "${train_path}"
  train_diffusion_for_dataset "guacamol" "${train_path}" "${valid_path}" \
    "data/guacamol/train.smiles" "data/guacamol/valid.smiles" "data/guacamol/test.smiles"
  evaluate_diffusion_for_dataset "guacamol" "${train_path}" "data/guacamol/test.smiles"
}

run_zinc250k() {
  prepare_split_dataset "zinc250k" "data/zinc250k.csv" "smiles" "${ZINC_TRAIN_SIZE}" "${ZINC_VALID_SIZE}" "${ZINC_TEST_SIZE}"
  local split_dir="${RESULTS_ROOT}/prep/zinc250k/splits"
  build_vocab_for_dataset "zinc250k" "smiles" "${split_dir}/train.smiles"
  train_diffusion_for_dataset "zinc250k" "${split_dir}/train.smiles" "${split_dir}/valid.smiles" \
    "${split_dir}/train.smiles" "${split_dir}/valid.smiles" "${split_dir}/test.smiles"
  evaluate_diffusion_for_dataset "zinc250k" "${split_dir}/train.smiles" "${split_dir}/test.smiles"
}

run_qm9() {
  prepare_split_dataset "qm9" "data/qm9/qm9_dataset.csv" "smiles" "${QM9_TRAIN_SIZE}" "${QM9_VALID_SIZE}" "${QM9_TEST_SIZE}"
  local split_dir="${RESULTS_ROOT}/prep/qm9/splits"
  build_vocab_for_dataset "qm9" "smiles" "${split_dir}/train.smiles"
  train_diffusion_for_dataset "qm9" "${split_dir}/train.smiles" "${split_dir}/valid.smiles" \
    "${split_dir}/train.smiles" "${split_dir}/valid.smiles" "${split_dir}/test.smiles"
  evaluate_diffusion_for_dataset "qm9" "${split_dir}/train.smiles" "${split_dir}/test.smiles"
}

for dataset in ${DATASETS}; do
  echo "[INFO] ===== dataset=${dataset} ====="
  case "${dataset}" in
    moses)
      run_moses
      ;;
    guacamol)
      run_guacamol
      ;;
    zinc250k)
      run_zinc250k
      ;;
    qm9)
      run_qm9
      ;;
    *)
      echo "[ERROR] Unsupported dataset: ${dataset}" >&2
      exit 1
      ;;
  esac
done

echo "[DONE] RESULTS_ROOT=${RESULTS_ROOT}"
