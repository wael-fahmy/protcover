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

Two external evaluations were pre-registered before they were run. The first
tests ClinVar variants added after the benchmark's snapshot in the benchmark's
own proteins: see
[docs/preregistration_external.md](docs/preregistration_external.md) and the
outcome in [docs/external_results.md](docs/external_results.md). The second
calibrates on the benchmark and tests ClinVar variants in 12,225 human proteins
outside it, with alignment-free descriptors:
[docs/preregistration_unseen.md](docs/preregistration_unseen.md).

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
| Threshold audit | `audit_thresholds.py` | `results/audit/` |
| External test set | `build_external.py`, `run_external.py`, `score_external.py` | `data/processed/external_temporal.parquet`, `results/external*/` |
| New-protein test set | `build_unseen.py`, `embed_esm2_unseen.py`, `run_unseen.py`, `score_unseen.py` | `data/processed/unseen_*.parquet`, `results/unseen/` |
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
assumptions on purpose); S4s holds out whole embedding regions only. Besides
the main run, three sensitivity runs change the family definition (down to
components of any significant MMseqs2 hit), one deployment-realistic run drops
the features that need a protein's whole variant panel, one alignment-free run
also drops alignment depth (`--no-msa`), and `--review-only` runs the comparisons
added after the second review (one variant per family, richer feature classes,
ratio-fit shares). `audit_thresholds.py` checks the computed thresholds against
an independent primal computation. `run_external.py` evaluates on the external
temporal test set (`build_external.py` builds it; see
[data/external/README.md](data/external/README.md)), and `run_unseen.py` on the
new proteins (`build_unseen.py` builds them; see
[data/unseen/README.md](data/unseen/README.md)).

**Runtime** on a 16-core laptop CPU (Windows 11, Python 3.11), measured for
the last full rerun, with runs overlapped: the main experiment (10 seeds, 6 in
parallel), both external analyses and the synthetic validation together took
76 minutes; the three family-definition runs and the deployment-realistic run
together, then the stress run, 2 h 17 min; summaries, scoring, tables and
figures 8 minutes; 3 h 41 min in total (one seed of the main experiment
takes about 55 minutes). Data and families take about 2 minutes, and recomputing the
ESM2 embeddings adds about 25 minutes. `reproduce_all.sh` runs the steps one
after another and takes longer.

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
