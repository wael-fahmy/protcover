#!/usr/bin/env bash
# Reproduce every result, figure, table and the paper PDF from scratch.
# Run from the repository root:  bash scripts/reproduce_all.sh
# PYTHON can point at a specific interpreter (default: .venv, else python).
# Several hours on a 16-core CPU; see README.md for measured runtimes.
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

# Experiments: main run, then the three family-definition sensitivity runs.
bash scripts/run_parallel.sh main --scenarios S1 S2 S3 S4
bash scripts/run_parallel.sh sens_mm050 --scenarios S1 S2 S3 S4 --family-def fam_mm050 --no-ablations
bash scripts/run_parallel.sh sens_hc --scenarios S1 S2 S3 S4 --family-def fam_hc --no-ablations
bash scripts/run_parallel.sh sens_remote --scenarios S1 S2 S3 S4 --family-def fam_remote --no-ablations
# Deployment-realistic run: raw score only, no variant-count descriptor.
bash scripts/run_parallel.sh realistic --scenarios S1 S2 S3 S4 --no-ablations --realistic
# Alignment-free descriptors (raw score, no alignment depth), as for proteins without an MSA.
bash scripts/run_parallel.sh nomsa --scenarios S1 S2 S3 S4 --no-msa --realistic --no-ablations
# Strict S4: whole embedding regions only.
bash scripts/run_parallel.sh s4strict --scenarios S4s --no-ablations
# Comparisons added after the second review: one variant per family, richer feature classes, ratio-fit shares.
bash scripts/run_parallel.sh review --review-only --scenarios S1 S2 S3 S4 S4s
# Stress test S5: a label-based shift that violates A2 on purpose.
bash scripts/run_parallel.sh stress_s5 --scenarios S5 --no-ablations
$PYTHON scripts/summarize.py --tag main
for tag in sens_mm050 sens_hc sens_remote realistic nomsa s4strict review stress_s5; do
  $PYTHON scripts/summarize.py --tag "$tag" --skip-support
done
# Audit of the conditional-calibration threshold against an independent primal computation.
for s in S1 S2 S3 S4; do $PYTHON scripts/audit_thresholds.py --scenarios "$s" --n-fam 3; done
$PYTHON scripts/audit_thresholds.py --summarize

# External temporal evaluation (pre-registered in docs/preregistration_external.md).
bash data/external/download.sh
$PYTHON -I scripts/build_external.py
$PYTHON scripts/run_external.py --base realistic --tag external
$PYTHON scripts/run_external.py --base main --tag external_main
$PYTHON scripts/summarize.py --tag external --skip-support
$PYTHON scripts/summarize.py --tag external_main --skip-support
$PYTHON scripts/score_external.py --tags external external_main

# Evaluation on proteins outside the benchmark (pre-registered in docs/preregistration_unseen.md).
# data/esm2_t12_35M_meanpool_unseen.npz is committed; refresh with
#   $PYTHON -I scripts/embed_esm2_unseen.py   (needs requirements-embed.txt)
bash data/unseen/download.sh
$PYTHON -I scripts/build_unseen.py
$PYTHON scripts/run_unseen.py --tag unseen
$PYTHON scripts/summarize.py --tag unseen --skip-support
$PYTHON scripts/score_unseen.py --tag unseen
$PYTHON scripts/analyze_size_effect.py --tag main
$PYTHON scripts/base_model_diagnostics.py
$PYTHON scripts/run_synthetic.py

# Figures, tables, citation check, paper.
$PYTHON scripts/make_figures.py --tag main
$PYTHON scripts/make_tables.py --tag main
$PYTHON scripts/verify_citations.py
( cd paper && latexmk -pdf -interaction=nonstopmode main.tex )
