# ProtCOVER

Family-aware conformal prediction for variant effect predictors on unseen
protein families. Code, experiments and paper for the RECOMB 2027 submission
"ProtCOVER: Family-Aware Conformal Prediction for Variant Effect Predictors on
Unseen Protein Families".

The method turns a zero-shot variant effect predictor (ESM-1b, EVE, GEMME,
TranceptEVE, PoET) into prediction sets over {benign, pathogenic} for ClinVar
missense variants, treating the protein family as the unit of exchangeability.
Everything runs on a CPU from ProteinGym's precomputed scores; no protein
language model is trained or run.

> Status: work in progress. Sections marked TODO are filled in as the
> corresponding experiments land.

## Install

Python 3.10 or later.

```bash
python -m venv .venv            # or: uv venv --python 3.11 .venv
.venv/bin/pip install -e ".[test]"   # Windows: .venv\Scripts\pip
```

Exact versions used for the paper are pinned in `requirements.txt`.
MMseqs2 (release `18-8cc5c`) is downloaded into `tools/` by the data script.

## Download data

```bash
bash data/download.sh
```

This fetches ProteinGym v1.3 clinical substitutions, the zero-shot clinical
scores, the clinical MSA weights and the reference file (about 50 MB of
downloads). See [data/README.md](data/README.md) for sources, checksums,
predictor choice and the score sign convention.

## Reproduce everything

```bash
bash scripts/reproduce_all.sh
```

Steps, in order:

| Script | Output |
|---|---|
| `scripts/build_dataset.py` | `data/processed/{variants,proteins}.parquet`, `results/data_summary.csv`, `results/sign_check.csv` |
| `scripts/build_families.py` | `data/processed/{families,homology}.parquet`, `results/families_summary.csv` |
| `pytest` | synthetic coverage tests (`tests/`) |
| `scripts/run_experiments.py` | `results/raw/<tag>/{metrics.csv,perfamily.parquet,splits.csv}` |
| `scripts/summarize.py` | `results/<tag>/{summary,dispersion,by_support}.csv` |

Expected runtime on a 16-core laptop CPU: data and families about 2 minutes;
experiments about 90 seconds per seed for S1, S2 and S3 together. TODO: final
figure for the full pipeline.

All settings live in `configs/default.yaml`. Seeds are fixed there, and every
random draw is seeded from (seed, scenario, method), so reruns are identical.

## Repository layout

```
protcover/   package: data, families, descriptors, splits, base model, conformal methods, metrics, synthetic data
scripts/     pipeline entry points
configs/     experiment configuration
data/        download script and data documentation (raw data is gitignored)
results/     summary CSVs (raw per-run output is gitignored)
figures/     figures for the paper
paper/       LNCS LaTeX source
tests/       pytest coverage tests on synthetic data
```

## License

MIT, see [LICENSE](LICENSE). ProteinGym data are distributed under their own
terms; see the ProteinGym repository.
