# Results: evaluation on proteins outside the benchmark (ESM-1b)

Pre-registration: `docs/preregistration_unseen.md` (commits 7b92c3e and 11b8122, public before
any method was run on the new proteins' labels). Every number below is copied from
`results/unseen/` (`hypotheses.csv`, `endpoints.csv`, `shortfall.csv`, `matched.csv`,
`certification.csv`) and `results/unseen_data.csv`, written by `scripts/summarize.py` and
`scripts/score_unseen.py`. Ten seeds; alpha = 0.1 unless stated; class-conditional versions.

## Verdicts

The primary population (all 8,579 new families) decides; the secondary population is the 4,448
families none of whose proteins has an MMseqs2 hit at E <= 1e-3 to a benchmark protein.

| Hypothesis | Quantity | Primary | Verdict | Secondary | Verdict |
|---|---|---|---|---|---|
| U1: Mondrian split under-covers pathogenic variants per family | B2 Cov_1 [95% CI] | 0.778 [0.771, 0.786] | supported | 0.729 [0.716, 0.743] | supported |
| U2: ProtCOVER's worse class at least 0.89 | min(Cov_0, Cov_1) | 0.873 | not supported | 0.838 | not supported |
| U3: ProtCOVER's shortfall not larger than HCP's or GWCP's by 0.01 or more | shortfall minus HCP's / GWCP's | -0.008 / -0.007 | supported | -0.021 / -0.017 | supported |
| U4: ProtCOVER's shortfall smaller than GWCP's by more than 0.01 | GWCP's minus ProtCOVER's | 0.007 | not supported | 0.017 | supported |

## A bug found while the evaluation ran, and the rerun

While the first run was in progress, an adversarial review of the proofs found that the linear
programming solver behind ProtCOVER's thresholds could, near a tie, accept within its tolerances
a basis that is optimal for the unperturbed program but not for the perturbed one, so that a
computed threshold fell below the supremum of program (3) (constructed examples: up to 4.6e-4).
The fix certifies every threshold in exact rational arithmetic and makes the weighted quantile of
HCP, GWCP and ProtCOVER-W exact at ties (commit 23bf9b0). It was committed and pushed before the
first run's outputs were opened; the first run's outputs are kept in `results/unseen_firstrun/`.
As the pre-registration requires, the evaluation was rerun with the fixed code (commit c240c13).
Every reported quantity of the rerun equals the first run's: the largest absolute difference in
any shortfall or mean set size, at any alpha and in any population, is 0. Of the 1,544,220
thresholds of the rerun, 1,543,938 were certified at the first solve and 282 after a cold
re-solve with tight tolerances; none had to be set to +inf; 1,906 certificates were decided in
exact arithmetic.

## Endpoints at alpha = 0.1

Family-weighted pathogenic coverage Cov_1 (95% family-bootstrap interval), benign coverage
Cov_0, worse-class shortfall below 0.90, mean set size and the error rate of singleton sets.

All new families (primary):

| Method | Cov_1 | Cov_0 | Shortfall | Set size | Singleton error |
|---|---|---|---|---|---|
| Mondrian split (B2) | 0.778 [0.771, 0.786] | 0.972 | 0.122 | 1.088 | 0.055 |
| HCP | 0.864 [0.858, 0.871] | 0.966 | 0.036 | 1.232 | 0.069 |
| Subsampling, once / repeated | 0.866 / 0.865 | 0.966 / 0.966 | 0.034 / 0.035 | 1.233 / 1.233 | 0.070 / 0.069 |
| GWCP | 0.866 [0.860, 0.872] | 0.965 | 0.034 | 1.236 | 0.071 |
| ProtCOVER-W | 0.868 [0.862, 0.874] | 0.965 | 0.032 | 1.237 | 0.072 |
| ProtCOVER-D | 0.875 [0.869, 0.881] | 0.967 | 0.025 | 1.231 | 0.066 |
| ProtCOVER | 0.873 [0.867, 0.879] | 0.965 | 0.027 | 1.251 | 0.059 |

No detectable benchmark homolog (secondary): HCP 0.817 [0.805, 0.830], GWCP 0.821 [0.810,
0.833], ProtCOVER-W 0.823, ProtCOVER-D 0.852 [0.842, 0.863], ProtCOVER 0.838 [0.827, 0.849];
Mondrian split 0.729. Benign coverage is 0.969 to 0.977 for every method.

Within the benchmark's length range (exploratory, no hypothesis; 8,356 families): shortfall 0.021
for HCP, 0.019 for GWCP, 0.017 for ProtCOVER-W, 0.011 for ProtCOVER-D and 0.012 for ProtCOVER
(worse-class coverage 0.888).

Other levels (worse-class shortfall, primary population): at alpha = 0.05, HCP 0.019, GWCP
0.020, ProtCOVER-D 0.013, ProtCOVER 0.012; at alpha = 0.2, HCP 0.043, GWCP 0.041,
ProtCOVER-D 0.049, ProtCOVER 0.039.

Mean set size at pathogenic coverage 0.90, interpolated along alpha (primary population): HCP
1.373, GWCP 1.372, ProtCOVER-W 1.366, ProtCOVER 1.366, ProtCOVER-D 1.343.

## Descriptive statistics (not endpoints)

- ESM-1b's per-protein mean AUC is 0.873 on the 818 new proteins with both classes, against
  0.892 on the benchmark proteins (Brandes et al. scores on both).
- The density ratio barely departs from one: the effective sample size of the calibration
  weights is 0.96 of the number of calibration families (range over seeds 0.94 to 0.98), and the
  mean maximum identity to training proteins is 0.142 for test families against 0.163 for
  calibration families.
- 9.4% of the test variants are pathogenic (9,555 of 101,197), in 911 of the 8,579 families.

## Reading

Variant-level calibration fails on new proteins as it does on the benchmark (U1). Under this real
shift every family-level method falls short in the pathogenic class, by 2.5 to 3.6 points in the
primary population and 4.8 to 8.3 points without detectable homologs, while benign variants are
over-covered. The label-free descriptors see little of the shift (effective sample size 0.96), but
the predictor is less accurate on the new proteins: as in the benchmark's stress test S5, the
shift is mostly one that the descriptors do not encode, which is outside assumption A2, so U2 was
not met. Conditioning still helps: ProtCOVER and ProtCOVER-D have the smallest shortfalls, by
amounts within the pre-registered margin of GWCP in the primary population and beyond it without
detectable homologs, at no larger sets at matched coverage. Leaving out the 223 families with a
protein longer than any benchmark protein, which no calibration family resembles in length,
lowers every family-level method's shortfall by 0.014 to 0.016 (exploratory).
