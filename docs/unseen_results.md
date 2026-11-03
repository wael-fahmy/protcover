# Results: evaluation on proteins outside the benchmark (ESM-1b and ESM2)

Two pre-registrations, one per predictor:

- ESM-1b (Part 1): `docs/preregistration_unseen.md` (commits 7b92c3e and 11b8122, public before
  any method was run on the new proteins' labels).
- ESM2 with 150M parameters (Part 2): `docs/preregistration_unseen_esm2.md` (commit d259b3a,
  public before any ESM-1b result was seen). Same data, populations, hypotheses and decision
  rules; the verdicts are reported next to ESM-1b's, not pooled.

Every number below is copied from `results/unseen/` and `results/unseen_esm2/`
(`hypotheses.csv`, `endpoints.csv`, `shortfall.csv`, `matched.csv`, `certification.csv`, and for
ESM2 `validity.json`) and from `results/unseen_data.csv` and `results/unseen_esm2/data.csv`,
written by `scripts/esm2_validity.py`, `scripts/summarize.py` and `scripts/score_unseen.py`. Ten
seeds; alpha = 0.1 unless stated; class-conditional versions.

# Part 1: ESM-1b

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

## A second rerun, on the exact-threshold code

The benchmark reruns then showed that the certificate can fail even after the cold re-solve
(334 of 3.37 million benchmark thresholds, which the code above set to +inf): a score whose
residual is below the solver's tolerances can belong at the other bound in exact arithmetic.
Since commit 7a68869, a rational dual simplex warm-started from the solver's basis then finds an
exactly optimal basis or proves the program unbounded, and every unboundedness verdict of the
solver is checked the same way. In this evaluation no threshold had been set to +inf and no
program is unbounded, so the change could not alter it. It was rerun on commit 7a68869 anyway,
and every output file (`results/unseen/*.csv`, `results/unseen_data.csv`), the certification
counts included, is byte-identical to that of the rerun above.

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

# Part 2: ESM2

ESM2 (`esm2_t30_150M_UR50D`) is scored by the wild-type marginal log-likelihood ratio, negated so
that higher means more pathogenic, on the same variants on both sides (`scripts/score_esm2.py`;
`data/unseen/README.md`). The evaluation ran on commit db0dcd8, after the validity check below.

## Validity check (pre-registered, run before the evaluation)

On the benchmark variants that train and calibrate the evaluation (development data), the
per-protein mean AUC of the oriented ESM2 scores is 0.808 over the 2,423 benchmark proteins with
both classes (59,972 variants), against 0.892 for ESM-1b on the same variants. A value at or below
0.5 would have meant an orientation or position bug; the check passed
(`results/unseen_esm2/validity.json`).

## Verdicts

| Hypothesis | Quantity | Primary | Verdict | Secondary | Verdict |
|---|---|---|---|---|---|
| U1: Mondrian split under-covers pathogenic variants per family | B2 Cov_1 [95% CI] | 0.850 [0.844, 0.857] | supported | 0.835 [0.824, 0.844] | supported |
| U2: ProtCOVER's worse class at least 0.89 | min(Cov_0, Cov_1) | 0.886 | not supported | 0.874 | not supported |
| U3: ProtCOVER's shortfall not larger than HCP's or GWCP's by 0.01 or more | shortfall minus HCP's / GWCP's | -0.008 / -0.007 | supported | -0.009 / -0.007 | supported |
| U4: ProtCOVER's shortfall smaller than GWCP's by more than 0.01 | GWCP's minus ProtCOVER's | 0.007 | not supported | 0.007 | not supported |

In the primary population, which decides, the four verdicts are those of ESM-1b. In the
secondary population U4 is not supported with ESM2, whereas it was with ESM-1b.

Every conditional-calibration threshold (1,544,220 over the ten seeds) was certified optimal:
1,543,686 on the first solve and 534 after a cold re-solve. None needed the exact simplex, and
none is infinite (`results/unseen_esm2/certification.csv`).

## Endpoints at alpha = 0.1

All new families (primary):

| Method | Cov_1 | Cov_0 | Shortfall | Set size | Singleton error |
|---|---|---|---|---|---|
| Mondrian split (B2) | 0.850 [0.844, 0.857] | 0.955 | 0.050 | 1.482 | 0.151 |
| HCP | 0.878 [0.872, 0.884] | 0.951 | 0.022 | 1.542 | 0.177 |
| Subsampling, once / repeated | 0.881 / 0.878 | 0.947 / 0.950 | 0.019 / 0.022 | 1.542 / 1.542 | 0.187 / 0.177 |
| GWCP | 0.879 [0.874, 0.885] | 0.949 | 0.021 | 1.542 | 0.181 |
| ProtCOVER-W | 0.881 [0.875, 0.887] | 0.949 | 0.019 | 1.544 | 0.182 |
| ProtCOVER-D | 0.885 [0.880, 0.891] | 0.953 | 0.015 | 1.559 | 0.169 |
| ProtCOVER | 0.886 [0.881, 0.892] | 0.951 | 0.014 | 1.559 | 0.182 |

No detectable benchmark homolog (secondary): HCP 0.865 [0.855, 0.874], GWCP 0.867 [0.857,
0.876], ProtCOVER-W 0.870 [0.860, 0.878], ProtCOVER-D 0.865 [0.856, 0.875], ProtCOVER 0.874
[0.865, 0.883]; Mondrian split 0.835. Benign coverage is 0.953 to 0.961 for every family-level
method. Shortfalls: HCP 0.035, GWCP 0.033, ProtCOVER-W 0.030, ProtCOVER-D 0.035, ProtCOVER
0.026 (subsampling once 0.031, repeated 0.034).

Within the benchmark's length range (exploratory, no hypothesis; 8,356 families): shortfall 0.014
for HCP, 0.013 for GWCP, 0.011 for ProtCOVER-W, 0.010 for ProtCOVER-D and 0.009 for ProtCOVER
(worse-class coverage 0.891).

Other levels (worse-class shortfall, primary population): at alpha = 0.05, HCP 0.012, GWCP
0.012, ProtCOVER-W 0.013, ProtCOVER-D 0.012, ProtCOVER 0.010; at alpha = 0.2, HCP 0.034, GWCP
0.032, ProtCOVER-W 0.033, ProtCOVER-D 0.023, ProtCOVER 0.030.

Mean set size at pathogenic coverage 0.90, interpolated along alpha (primary population): HCP
1.611, GWCP 1.607, ProtCOVER-W 1.605, ProtCOVER 1.606, ProtCOVER-D 1.608.

## Descriptive statistics (not endpoints)

- ESM2's per-protein mean AUC is 0.795 on the 818 new proteins with both classes, against 0.808
  on the 2,423 benchmark proteins (ESM-1b: 0.873 and 0.892).
- The descriptors and the density ratio do not depend on the predictor, so the calibration
  weights are those of Part 1 (effective sample size 0.96 of the calibration families).
- Sets are larger and singleton sets less often right than with ESM-1b (mean set size 1.48 to
  1.56 against 1.09 to 1.25; singleton error 0.15 to 0.19 against 0.06 to 0.07), as expected of
  a less accurate predictor.

## Reading

With ESM2 the pre-registered verdicts are those of ESM-1b. Variant-level calibration fails on the
new proteins (U1); every family-level method falls short in the pathogenic class, by 1.4 to 2.2
points in the primary population and 2.6 to 3.5 points without detectable homologs, while benign
variants are over-covered, so U2 is not met; ProtCOVER's shortfall is within the margin of HCP's
and GWCP's (U3) but not smaller than GWCP's by more than it (U4). ESM2 is less accurate than
ESM-1b on both sides but loses less accuracy on the new proteins (per-protein AUC 0.808 to 0.795,
against 0.892 to 0.873), and every method falls short less with it. Two ESM-1b results outside
the primary verdicts do not replicate. Without detectable homologs, ProtCOVER's advantage over
GWCP is within the margin (0.007, against 0.017 with ESM-1b). ProtCOVER-D, which fell short least
with ESM-1b, falls short slightly more than ProtCOVER in the primary population (0.015 against
0.014) and as much as HCP without detectable homologs (0.035). Leaving out the families with a
protein longer than any benchmark protein lowers every family-level method's shortfall by 0.004
to 0.008 (exploratory), about half as much as with ESM-1b.
