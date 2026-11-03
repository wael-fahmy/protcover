# Pre-registration: a second predictor (ESM2) for the evaluation on proteins outside the benchmark

Written on 2026-11-02, while the pre-registered ESM-1b evaluation
(`docs/preregistration_unseen.md`) was running and before any of its results were seen. It
becomes binding when it is committed and pushed. After that, nothing in it may be changed; if
a bug is found afterwards, it is fixed, the evaluation is rerun, and both the bug and the
change are reported.

## Why

A reviewer asked for the evaluation on unseen proteins to use more than one predictor. Of the
benchmark's five predictors, only ESM-1b has precomputed scores for every human protein (Brandes
et al.); EVE, GEMME, TranceptEVE and PoET need multiple sequence alignments or retrieval that the
new proteins do not have in this pipeline. We therefore score a second protein language model
ourselves, with one score function for the benchmark proteins and the new proteins alike.

## What was seen before this document was frozen

- Everything listed in `docs/preregistration_unseen.md`.
- No result of the ESM-1b evaluation on the new proteins: no metric, coverage, set size or
  hypothesis outcome.
- The ESM2 scoring had started (`scripts/score_esm2.py`, below). Before it started, a smoke
  test scored two proteins (NP_000007.1 from the benchmark; A0AVI2, a new protein of 2,057
  residues) to check the windows; it looked at five score values and the window counts, and
  at no label.
- No ESM2 score has been compared with any label, on the benchmark or on the new proteins.

## Data

- Variants, proteins, families, populations (U, Ustrict, Ulen) and splits: exactly those of
  `docs/preregistration_unseen.md`. The 2,755 benchmark variants without a Brandes et al.
  ESM-1b score stay excluded, although ESM2 can score them, so that both predictors are
  evaluated on the same variants.
- Scores: `scripts/score_esm2.py` at the commit that adds this file. Model
  `esm2_t30_150M_UR50D` (fair-esm 2.0.0; checkpoint SHA-256
  `881c7176cf198ef8dec26a3c375d40eb58d0c33df95c22562ca6cc6d3f812c62`). Score: the wild-type
  marginal log-likelihood ratio, log p(mutant) - log p(wild type) at the variant's position
  given the unmasked wild-type sequence, oriented as its negative (higher = more
  pathogenic). Sequences longer than 1,022 residues are tiled in windows of 1,022 with stride
  511 and the log-probabilities of a position are averaged over the windows that cover it.
  Benchmark proteins use the benchmark sequence; new proteins the sequence scored by Brandes
  et al. (the same sequences as for ESM-1b). The wild type is checked at every position.
- The script writes `data/processed/esm2_150M_scores.parquet` and records its SHA-256, with
  the model's, in `data/unseen/esm2_150M_summary.json`, which is committed with the results.
- Validity check before the evaluation is run: the per-protein mean AUC of the oriented ESM2
  scores on the benchmark variants used for training and calibration (development data,
  whose labels were used throughout development). A value at or below 0.5 would mean an
  orientation or position bug, which would be fixed before running. The value is reported.

## Splits, fitting, methods, endpoints and populations

As in `docs/preregistration_unseen.md`, with ESM-1b replaced by ESM2:
`scripts/run_unseen.py --predictor ESM2_150M` (raw score only, alignment-free descriptors,
the same seeds, splits, gate tuning, methods, alphas and populations). Settings as in
`configs/default.yaml` at the commit that adds this file.

## Hypotheses (stated before the evaluation)

The four hypotheses of `docs/preregistration_unseen.md`, applied to ESM2, with the same
decision rules and the same code (`hypotheses()` in `scripts/score_unseen.py`, unchanged
since commit 11b8122). A difference in shortfall smaller than 0.01 is treated as no
difference.

- U1 (ESM2). Variant-level Mondrian split (B2) covers less than 0.90 of pathogenic variants
  per test family (the upper end of the bootstrap interval is below 0.90).
- U2 (ESM2). ProtCOVER's worse-class coverage is at least 0.89.
- U3 (ESM2). ProtCOVER's worse-class shortfall exceeds neither HCP's nor GWCP's by 0.01 or
  more.
- U4 (ESM2). ProtCOVER's worse-class shortfall is smaller than GWCP's by more than 0.01.

Each is reported as supported or not, whatever the result, for the primary and secondary
populations; the primary population decides. The exploratory population has no hypothesis.
The paper reports the ESM2 verdicts next to the ESM-1b ones and does not pool them.

## Commands

```
python -I scripts/score_esm2.py --threads 8                     # label-free scores
python scripts/run_unseen.py --predictor ESM2_150M --tag unseen_esm2
python scripts/summarize.py --tag unseen_esm2 --skip-support
python scripts/score_unseen.py --tag unseen_esm2 --predictor ESM2_150M
```

`scripts/score_unseen.py` gains a `--predictor` option that changes only its descriptive data
summary (the per-protein AUC of the chosen predictor, written to `results/unseen_esm2/` instead
of `results/unseen_data.csv`); the endpoint and hypothesis code stays as it is.

## Known caveats (stated in advance)

- ESM2 and ESM-1b are protein language models from the same group, trained on UniRef with the
  same masked-language objective; they are related, not independent, predictors. This tests
  whether the conclusions depend on one score function, not whether they hold for
  alignment-based predictors, which cannot be scored on the new proteins here.
- The 150M-parameter model was chosen for the CPU budget (estimated at about 11 hours on 8
  threads for both sets; the 650M model would take several times longer); it is smaller than
  ESM-1b (650M) and usually less
  accurate. A weaker predictor gives larger sets but does not by itself change what the
  coverage guarantees claim.
- Wild-type marginal scores (one forward pass per window) rather than masked marginals (one
  pass per position), again for the CPU budget.
- The ESM-1b results will be known when the ESM2 results are scored; the hypotheses above
  were fixed before either was seen. Every caveat of `docs/preregistration_unseen.md` applies.
