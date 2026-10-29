#!/usr/bin/env bash
# Reproduce every result from scratch. Run from the repository root:
#   bash scripts/reproduce_all.sh
# PYTHON can point at a specific interpreter (default: .venv, else python).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x .venv/Scripts/python.exe ]]; then PYTHON=.venv/Scripts/python.exe
  elif [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python
  else PYTHON=python; fi
fi

bash data/download.sh
# data/uniprot_map.tsv is committed; refresh it with: $PYTHON scripts/fetch_uniprot_map.py
$PYTHON scripts/build_dataset.py
$PYTHON scripts/build_families.py
$PYTHON -m pytest -q
$PYTHON scripts/run_experiments.py --scenarios S1 S2 S3 --tag main
$PYTHON scripts/summarize.py --tag main
