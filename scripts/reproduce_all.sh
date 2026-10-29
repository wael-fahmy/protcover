#!/usr/bin/env bash
# Reproduce every result, figure, table and the paper PDF from scratch.
# Run from the repository root:  bash scripts/reproduce_all.sh
# PYTHON can point at a specific interpreter (default: .venv, else python).
# About an hour on a 16-core CPU; see README.md.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x .venv/Scripts/python.exe ]]; then PYTHON=.venv/Scripts/python.exe
  elif [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python
  else PYTHON=python; fi
fi
export PYTHON

# Data and families.
bash data/download.sh
# data/uniprot_map.tsv and data/esm2_t12_35M_meanpool.npz are committed; refresh with
#   $PYTHON scripts/fetch_uniprot_map.py
#   $PYTHON scripts/embed_esm2.py   (needs requirements-embed.txt; about 25 min on CPU)
$PYTHON scripts/build_dataset.py
$PYTHON scripts/build_families.py
$PYTHON -m pytest -q

# Experiments: main run, then the two family-definition sensitivity runs.
bash scripts/run_parallel.sh main --scenarios S1 S2 S3 S4
bash scripts/run_parallel.sh sens_mm050 --scenarios S1 S2 S3 S4 --family-def fam_mm050 --no-ablations
bash scripts/run_parallel.sh sens_hc --scenarios S1 S2 S3 S4 --family-def fam_hc --no-ablations
# Deployment-realistic run: raw score only, no variant-count descriptor.
bash scripts/run_parallel.sh realistic --scenarios S1 S2 S3 S4 --no-ablations --realistic
$PYTHON scripts/summarize.py --tag main
$PYTHON scripts/summarize.py --tag sens_mm050 --skip-support
$PYTHON scripts/summarize.py --tag sens_hc --skip-support
$PYTHON scripts/summarize.py --tag realistic --skip-support
$PYTHON scripts/analyze_size_effect.py --tag main
$PYTHON scripts/base_model_diagnostics.py
$PYTHON scripts/run_synthetic.py

# Figures, tables, citation check, paper.
$PYTHON scripts/make_figures.py --tag main
$PYTHON scripts/make_tables.py --tag main
$PYTHON scripts/verify_citations.py
( cd paper && latexmk -pdf -interaction=nonstopmode main.tex )
