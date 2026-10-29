#!/usr/bin/env bash
# Run scripts/run_experiments.py with one process per seed, JOBS at a time.
# Usage: bash scripts/run_parallel.sh TAG [extra run_experiments.py args]
# Environment: PYTHON (interpreter), JOBS (default 5), SEEDS (default 0..9).
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="$1"; shift
PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
JOBS="${JOBS:-5}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
mkdir -p "results/raw/${TAG}"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
printf '%s\n' ${SEEDS} | xargs -P "${JOBS}" -I{} sh -c \
  "\"${PYTHON}\" scripts/run_experiments.py --seeds {} --tag ${TAG} $* > results/raw/${TAG}/seed{}.log 2>&1 && echo seed {} ok || echo seed {} FAILED"
