#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <dataset> [results_root]" >&2
  echo "Valid datasets: moses | guacamol | zinc250k | qm9" >&2
  exit 1
fi

DATASET="$1"
RESULTS_ROOT_ARG="${2:-}"

case "${DATASET}" in
  moses|guacamol|zinc250k|qm9)
    ;;
  *)
    echo "[ERROR] Unsupported dataset: ${DATASET}" >&2
    echo "Valid datasets: moses | guacamol | zinc250k | qm9" >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -n "${RESULTS_ROOT_ARG}" ]]; then
  export RESULTS_ROOT="${RESULTS_ROOT_ARG}"
else
  export RESULTS_ROOT="results/stamp_diffusion_${DATASET}_$(date +%Y%m%d_%H%M%S)"
fi

export DATASETS="${DATASET}"

cd "${REPO_ROOT}"
bash scripts/hpc/train_stamp_diffusion_datasets.sh
