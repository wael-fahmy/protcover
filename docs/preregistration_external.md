# Pre-registration: external temporal evaluation

Written on 2026-10-29, before any method was evaluated on the external data. The
external table was built by `scripts/build_external.py` (see `data/external/README.md`);
only a sanity check of the predictor itself (AUC) has been computed on it. Nothing in
this document may be changed after the evaluation is run; if a bug is found afterwards,
it is fixed, the evaluation is rerun, and both the bug and the change are reported.

The ProteinGym clinical benchmark (all splits S1 to S4, the sensitivity runs, the
ablations) has been used for method development and is reported as development data.
This evaluation is the confirmation.

## Data

- Source: `data/processed/external_temporal.parquet`: ClinVar missense variants in the
  benchmark's proteins whose VariationIDs are absent from the 2023-06-01 ClinVar release
  and whose amino-acid change is not in the benchmark; P/LP versus B/LB, at least one star;
  ESM-1b scores from Brandes et al. (oriented: higher = more pathogenic).
- Inclusion: proteins of at most 1,022 residues (`over_1022 == False`), where the
  external ESM-1b scores equal the benchmark's to rounding, so calibration and test use
  the same score function. Excluded as a precaution: rows with `in_2023_07_release` (could
  predate the benchmark snapshot) and rows with `aa_change_in_2023_06_release` (the same
  change was in ClinVar before the snapshot under another VariationID).
- Predictor: ESM-1b only (no genome-wide scores exist for the others on the same MSAs).

## Splits and fitting

For each seed 0 to 9 and each benchmark setting (S1; S2 with beta 0.5, 1, 2; S3; S4),
the benchmark split defines training, calibration and test families exactly as in the
benchmark runs (`scripts/run_experiments.py`, family definition `fam_mm030`). The base
model is fitted on the benchmark variants of the training families; calibration uses the
benchmark variants of the calibration families; the test set is the external variants of
the proteins in the test families (families without external variants drop out).
Family descriptors are the benchmark's (label-free). Gate parameters are tuned on the
inner split of the training families, as in the benchmark.

- Primary base model: raw score only (the `--realistic` configuration), which needs no
  variant panel and no variant-count descriptor.
- Secondary base model: the main configuration, with each external variant standardized
  by the mean and standard deviation of the same protein's benchmark ESM-1b scores.

## Methods (all evaluated and all reported)

Variant-level split and Mondrian split (B1, B2); HCP, single and repeated subsampling
(B3); group-weighted conformal prediction (B4); density-ratio weighting (ProtCOVER (ii),
`P_w`) and with the gate (`P_full`); family-level conditional calibration with the
support-descriptor class (`C_supp`, primary new method) and with the density-ratio
class (`C_ratio`). Marginal and class-conditional versions. Settings as in
`configs/default.yaml` at the commit that adds this file.

## Endpoints

Alpha = 0.1 is primary; 0.05 and 0.2 are secondary.

1. Family-weighted pathogenic and benign coverage (Cov_1, Cov_0) of class-conditional
   methods, per setting, mean over seeds, with family-bootstrap intervals (1,000
   resamples within seed) and the seed-to-seed standard deviation.
2. Worse-class shortfall max(0, 0.90 - min(Cov_0, Cov_1)), averaged over the shifted
   settings (S2 with beta > 0, S3, S4) and its maximum.
3. Mean set size and singleton error; set size at matched pathogenic coverage.

## Hypotheses (stated before the evaluation)

- H1. In S1, variant-level Mondrian split (B2) covers less than 0.90 of pathogenic
  variants per family.
- H2. In S1, family-level class-conditional HCP has Cov_1 and Cov_0 of at least 0.89.
- H3. Over the shifted settings, C_supp's mean worse-class shortfall is at most that of
  HCP and of GWCP. A difference smaller than 0.01 is treated as no difference.
- H4. Support-based abstention (`P_full` versus `B5_random`) does not reduce the error of
  retained singleton sets (the benchmark's finding).

Each hypothesis is reported as supported or not, whatever the result.

## Known caveats (stated in advance)

- Recent ClinVar classifications can use computational evidence (PP3/BP4), which may
  inflate the predictor's apparent accuracy on new variants.
- The test proteins are benchmark proteins: this evaluates new variants and later labels
  within held-out families, not new proteins.
- Proteins over 1,022 residues are excluded by design, which removes 43% of the external
  variants and shifts the length distribution.
