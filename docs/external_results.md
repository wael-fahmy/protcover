# External temporal evaluation: outcome

Run on 2026-10-29 after the pre-registration (`docs/preregistration_external.md`, with
Amendment 1) was committed and pushed (commit 39fb88e on `origin/main`). Commands:

    python scripts/run_external.py --base realistic --tag external        # primary
    python scripts/run_external.py --base main --tag external_main       # secondary
    python scripts/summarize.py --tag external --skip-support
    python scripts/summarize.py --tag external_main --skip-support
    python scripts/score_external.py --tags external external_main

After the first run, an implementation error in the conditional methods was found and
fixed, and the evaluation was rerun (see "Rerun after a fix" below). No verdict changed.
Every value below is from the rerun and is in `results/<tag>/hypotheses.csv`,
`shortfall.csv`, `endpoints.csv` and `matched.csv`.

## Verdicts (alpha = 0.1)

| | Primary (raw score) | Secondary (main base model) |
|---|---|---|
| H1: S1 B2 pathogenic coverage below 0.90 | supported (0.873, CI 0.864 to 0.883) | supported (0.838, CI 0.827 to 0.849) |
| H2: S1 HCP Cov_1 and Cov_0 at least 0.89 | supported (0.939 and 0.918) | supported (0.947 and 0.919) |
| H3a: C_supp shortfall at most HCP's and GWCP's | supported | supported |
| H3b: C_ratio shortfall at most HCP's and GWCP's | supported | supported |
| H4: abstention does not reduce retained singleton error | supported | supported |

## What the verdicts do and do not show

- H3a and H3b hold because no shift-aware method falls short on this data: over the five
  shifted settings, GWCP, density-ratio weighting, C_supp and C_ratio all have zero
  worse-class shortfall at every alpha. HCP falls short in one setting (S2, beta 2).
  The external data therefore confirm that shift-aware calibration removes HCP's
  shortfall, but they do not separate the new methods from GWCP or from weighting.
- Coverage is above target in most cells for every family-level method (S1 HCP Cov_1 is
  0.939 against 0.90). The predictor separates the new variants better than the
  benchmark variants it was calibrated on, which is consistent with the pre-stated
  PP3/BP4 caveat. Over-coverage hides shortfall, so this test is less demanding than the
  benchmark.
- Set size at matched pathogenic coverage (shifted settings, primary): weighting and
  GWCP are the smallest, the conditional methods are in between, HCP is the largest.

## Rule fixed after the run

The pre-registration states H4 without a numeric decision rule. The rule used was chosen
after the run, when the per-setting differences had already been seen: abstention reduces the error if,
in some setting, the seed-paired 95% t-interval of P_full minus B5_random (retained
singleton error) lies entirely below zero. No setting meets it; the most negative
setting is S2 with beta 2 (primary: -0.0012, interval -0.0038 to 0.0015). The gate
abstains on at most 2% of families in any setting, so the comparison has little power,
and the verdict is the same under any rule that requires the difference to exceed noise.

## Rerun after a fix

As the pre-registration requires, the error and the change are reported here.

**Error.** The conditional methods (C_supp, C_ratio) compute their threshold from a
quantile-regression linear program. The coverage proof needs the largest optimal
fitted value when the program has several optimal solutions; the first run returned
whichever optimal solution the solver found, which can be smaller. The gap was found
while the proof was being checked, after the first run.

**Change** (commit 2347088). The test family's weight in the program is raised from tau
to tau + 1e-4, which yields a fitted value at least the largest optimal one (or an
infinite threshold when the raised program is unbounded), and a test score is in the
set if it is at most the threshold plus 1e-9. Both changes can only enlarge the sets. A
unit test (`tests/test_conditional.py::test_tie_returns_largest_optimal_threshold`)
fails on the old code and passes on the new.

**Rerun.** The five commands above were rerun on commit 2347088, with the same
configuration, data and seeds. Only C_supp and C_ratio changed. The largest change in
any reported value was 1.2e-4 (a coverage mean in `summary.csv`); in `shortfall.csv`
it was 1.4e-5, in `matched.csv` 1.1e-5 and in `endpoints.csv` 9.8e-5. All other
methods, every H1, H2 and H4 value, and every verdict in `hypotheses.csv` are identical
to the first run. The first run's files are in git history at commit 3f93e52.

## Second rerun after a further fix

**Error.** An adversarial review of the proofs (2026-11-02) found that the solver could, near a
tie, accept within its tolerances a basis that is optimal for the program with test weight tau
but not for tau + 1e-4, so that a computed threshold could still fall below the largest optimal
fitted value; and that the weighted quantile of HCP, GWCP and ProtCOVER-W accepted a cumulative
weight up to a relative 1e-12 below the level.

**Change** (commit 23bf9b0). Every threshold of the conditional methods is certified in exact
rational arithmetic to come from an optimal basis of the perturbed program, and rounded up; an
uncertifiable threshold is re-solved cold with tight tolerances and, failing that, set to +inf.
The weighted quantile decides comparisons that rounding could get wrong in exact arithmetic.

**Rerun.** The five commands above were rerun on commit c240c13. Every row of `seed_summary.csv`
in both analyses (pathogenic and benign coverage, set size; 432 rows each) is identical to the
previous run, and so is every value and verdict in `hypotheses.csv`. Of the thresholds
computed, 674,049 (primary) and 674,235 (secondary) were certified at the first solve, 66 and 78
after a cold re-solve, and none was set to +inf by the check; 885 and 687 programs were
unbounded (infinite thresholds by construction). The counts are in `results/<tag>/certification.csv`.

## Third rerun, on the exact-threshold code

**Change** (commit 7a68869). On the benchmark, the certificate above failed for 334 of 3.37
million thresholds even after the cold re-solve, and those thresholds were set to +inf: a score
whose residual is below the solver's tolerances can belong at the other bound in exact
arithmetic. A rational dual simplex warm-started from the solver's basis now finds an exactly
optimal basis in that case or proves the program unbounded, and every unboundedness verdict of
the solver is checked by the same exact simplex.

**Rerun.** The five commands above were rerun on commit 7a68869. Every output file of both analyses (`seed_summary.csv`,
`hypotheses.csv`, `shortfall.csv`, `matched.csv` and the rest) and `results/external_data.csv`
are byte-identical to the previous run, except `certification.csv`: its counts are unchanged and
it gains the number of exact pivots. Each of the 885 (primary) and 687 (secondary) unbounded
verdicts was confirmed exactly (141 and 91 pivots in all), none was corrected, and no threshold
needed the exact simplex otherwise.
