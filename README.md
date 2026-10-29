# ProtCOVER

Family-aware conformal prediction for variant effect predictors on unseen
protein families. Code, experiments and paper for the RECOMB 2027 submission
"ProtCOVER: Family-Aware Conformal Prediction for Variant Effect Predictors on
Unseen Protein Families" (W. Fahmy, E. Krikun, A. Alkhateeb; Lakehead University).

ProtCOVER turns a zero-shot variant effect predictor (ESM-1b, EVE, GEMME,
TranceptEVE, PoET) into prediction sets over {benign, pathogenic} for ClinVar
missense variants, treating the protein family as the unit of exchangeability:

1. **Family-level calibration** (hierarchical conformal pooling): each calibration
   family has total weight one, so large families cannot set the error rate.
2. **Conditional calibration on an estimated density ratio**: a family-weighted
   quantile regression (test family at +inf) whose features are a constant and a
   cross-fitted density ratio of label-free family descriptors (sequence
   similarity to the training proteins, alignment depth, ESM2 embedding
   distance). Family-weighted coverage holds exactly under family
   exchangeability, under the estimated shift, and under every nonnegative
   function in their span (`protcover/conditional.py`). Two variants are also
   reported: reweighting families by the ratio (ProtCOVER-W) and conditioning on
   the descriptors themselves, which needs no batch of target families (ProtCOVER-D).
3. **Support gate**: queries with too little effective calibration support get
   the full label set. This preserves every coverage guarantee; on this
   benchmark and on the external test set it did not improve selective risk.

An external test on ClinVar variants added after the benchmark's snapshot was
pre-registered before it was run: see
[docs/preregistration_external.md](docs/preregistration_external.md) and the
outcome in [docs/external_results.md](docs/external_results.md).

Everything runs on a CPU from ProteinGym's precomputed scores; no protein
language model is trained. The paper is in `paper/main.pdf`.

## Install

Python 3.10 or later.

```bash
python -m venv .venv                      # or: uv venv --python 3.11 .venv
.venv/bin/pip install -r requirements.txt # Windows: .venv\Scripts\pip
.venv/bin/pip install -e ".[test]"
```

`requirements.txt` pins the exact versions used for the paper. Recomputing the
ESM2 embeddings (optional, they are committed) also needs
`pip install -r requirements-embed.txt`. Building the PDF needs a LaTeX
distribution with `latexmk` and the Springer `llncs` class.

## Download data

```bash
bash data/download.sh
```

Fetches ProteinGym v1.3 clinical substitutions, zero-shot clinical scores,
clinical MSA weights, the reference file and MMseqs2 release 18-8cc5c (about
50 MB). See [data/README.md](data/README.md) for sources, checksums, predictor
choice and the score sign convention.

## Reproduce everything

```bash
bash scripts/reproduce_all.sh
```

| Step | Script | Output |
|---|---|---|
| Tidy data | `build_dataset.py` | `data/processed/`, `results/data_summary.csv`, `results/sign_check.csv` |
| Families | `build_families.py` | `data/processed/families.parquet`, `results/families_summary.csv` |
| Tests | `pytest` | synthetic coverage checks in `tests/` |
| Experiments | `run_parallel.sh` (calls `run_experiments.py` once per seed) | `results/raw/<tag>/seed<k>/` |
| Summaries | `summarize.py`, `analyze_size_effect.py`, `base_model_diagnostics.py` | `results/<tag>/*.csv`, `results/base_model.csv` |
| External test set | `build_external.py`, `run_external.py`, `score_external.py` | `data/processed/external_temporal.parquet`, `results/external*/` |
| Synthetic validation | `run_synthetic.py` | `results/synthetic.csv` |
| Figures and tables | `make_figures.py`, `make_tables.py` | `figures/`, `paper/tables/` |
| Citations | `verify_citations.py` | `results/citation_check.csv` |
| Paper | `latexmk` in `paper/` | `paper/main.pdf` |

Every number in the paper is a LaTeX macro written by `make_tables.py` from the
CSVs under `results/`; nothing is typed by hand.

Splits: S1 exchangeable families; S2 test families sampled toward low
sequence identity and shallow alignments (strength beta); S3 orphan families;
S4 held-out regions of ESM2 embedding space; S5 a stress test that shifts
predictor accuracy without changing the descriptors (it violates the
assumptions on purpose). Besides the main run, three sensitivity runs change the
family definition (down to components of any significant MMseqs2 hit), one
deployment-realistic run drops the features that need a protein's whole variant
panel, and `run_external.py` evaluates on the external temporal test set
(`build_external.py` builds it; see [data/external/README.md](data/external/README.md)).

**Runtime** on a 16-core laptop CPU (Windows 11, Python 3.11): data and
families about 2 minutes; main experiment (10 seeds, 5 in parallel) about
8 minutes; each sensitivity run about 5–15 minutes; summaries about 2 minutes
each; synthetic validation about 8 minutes; about an hour in total.
Recomputing the ESM2 embeddings adds about 25 minutes.

All settings live in `configs/default.yaml`. Every random draw is seeded from
(seed, scenario, method), so reruns give identical results.

## Repository layout

```
protcover/   package: data, families, descriptors, splits, base model, conformal methods,
             density ratios, support gate, metrics, synthetic data, per-split pipeline
scripts/     pipeline entry points (see the table above)
configs/     experiment configuration
data/        download script, data documentation, UniProt map, ESM2 embeddings (raw data is gitignored)
results/     summary CSVs (main, sensitivity runs, synthetic); results/checkpoint1 holds the
             first checkpoint run of an earlier pipeline version, kept for the record
figures/     paper figures (PDF and PNG)
paper/       LNCS LaTeX source, generated tables, bibliography, PDF
tests/       pytest coverage tests on synthetic data
```

## License

MIT, see [LICENSE](LICENSE). ProteinGym data are distributed under their own
terms; see the ProteinGym repository.
