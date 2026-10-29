# Registered abstract (RECOMB 2027, submission 434)

Registered 2026-10-28 23:32. The final paper must stay consistent with this text.

**Title:** ProtCOVER: Family-Aware Conformal Prediction for Variant Effect Predictors on Unseen Protein Families

**Authors (in order):**

1. Wael Fahmy, Lakehead University, Canada (wfahmy@lakeheadu.ca), corresponding, submitting author
2. Elena Krikun, Lakehead University, Canada (ekrikun@lakeheadu.ca), corresponding
3. Abedalrhman Alkhateeb, Lakehead University, Canada (aalkhate@lakeheadu.ca), corresponding

**Keywords:** conformal prediction; variant effect prediction; protein language models;
uncertainty quantification; distribution shift; selective prediction; ProteinGym; protein families

## Abstract

Protein language models and related variant effect predictors (VEPs) are increasingly used to
classify missense variants, yet their reliability is typically summarized by benchmark-wide
averages. In practice, predictors are applied to proteins that are evolutionarily distant from
those used for calibration, which constitutes a family-level distribution shift. Standard split
conformal prediction assumes exchangeable variants and offers no guarantee when entire protein
families are unseen, while existing VEP calibration schemes condition on residue, gene or domain
attributes rather than on homology support.

We present ProtCOVER, a conformal procedure that treats homology families, rather than individual
variants, as the exchangeable unit. ProtCOVER (i) constructs label-free homology clusters and
calibrates at the family level, yielding marginal coverage for a new family under family
exchangeability; (ii) reweights calibration families using estimated family-level density ratios
from label-free descriptors (sequence similarity, alignment depth and embedding distance) to adapt
to descriptor shift; and (iii) computes the effective calibration support for each query protein
and returns a conservative fallback set, or abstains, when support is insufficient. We state the
assumptions under which each guarantee holds and separately measure empirical robustness where they
do not.

We evaluate ProtCOVER on ProteinGym clinical substitution benchmarks using precomputed zero-shot
scores from multiple predictors under leave-family-out splits. We compare against split, Mondrian,
group-weighted and hierarchical conformal baselines, reporting per-family coverage, fraction of
undercovered families, prediction set size, pathogenic miss rate and risk versus abstention. Code
will be released publicly.

## Consistency checklist for the final paper

- Descriptors must include sequence similarity, alignment depth **and embedding distance**.
- Component (iii) must be a per-query effective calibration support with a fallback set or abstention.
- Baselines: split, Mondrian, group-weighted, hierarchical conformal.
- Metrics: per-family coverage, fraction of undercovered families, set size, pathogenic miss rate,
  risk versus abstention.
- Splits: leave-family-out on ProteinGym clinical substitutions, several zero-shot predictors.
