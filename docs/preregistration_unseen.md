# Pre-registration: evaluation on proteins outside the benchmark

Written on 2026-10-30, before any method was evaluated on the real labels of the data
below. It becomes binding when it is committed and pushed. After that, nothing in it may be
changed; if a bug is found afterwards, it is fixed, the evaluation is rerun, and both the
bug and the change are reported.

The ProteinGym clinical benchmark (all splits, sensitivity runs, ablations, the review
comparisons and the alignment-free development run) and the earlier temporal evaluation
(`docs/preregistration_external.md`) have been used for development. This evaluation is
the confirmation on proteins that played no part in development.

## Why

The earlier external evaluation tested new variants in the benchmark's own proteins. A
reviewer asked for proteins and families that the method has never seen. This evaluation
calibrates on the benchmark and tests on ClinVar variants in human proteins outside it,
under the shift that actually separates them, rather than a simulated one.

## What was seen before this document was frozen

- Counts and class balance of the new data (`data/unseen/build_summary.json`), and a
  label-free comparison of the two ESM-1b score sources on benchmark variants
  (`data/unseen/README.md`).
- The alignment-free development run on the benchmark (`results/nomsa/`), with all its
  results.
- A dry run of `scripts/run_unseen.py --dry-run` to test the code path: 300 random test
  families, labels randomly permuted, seed 0, alpha 0.1. Its label-free diagnostics were
  seen (descriptor means of calibration and test families, effective sample size of the
  calibration weights); no metric was looked at, and metrics on permuted labels carry no
  information about the real ones. Its outputs were deleted.
- Nothing else: no method, base model or performance statistic has been run or computed on
  the real labels of the new proteins.

## Data

- Built by `scripts/build_unseen.py` (see `data/unseen/README.md` and
  `data/unseen/build_summary.json` for every filter step, input file and count).
- Test variants: ClinVar missense variants (`variant_summary.txt.gz` of 2026-10-26,
  `hgvs4variation.txt.gz` of 2026-10-25) in human protein-coding genes (HGNC, 2026-10-06)
  not in the benchmark (excluded by gene symbol, GeneID and UniProt accession), GRCh38,
  single nucleotide variants, at least one review star, P/LP versus B/LB, placed on the
  Brandes et al. canonical UniProt entry through a RefSeq protein (release 237) with the
  same numbering; wild type checked at every position; start-loss changes excluded; no
  date filter; no length cap.
- 101,197 variants (9,555 pathogenic, 91,642 benign) in 12,225 proteins.
- Predictor: ESM-1b, with the precomputed scores of Brandes et al. (oriented: higher =
  more pathogenic) for both the benchmark variants (training and calibration) and the test
  variants, so that every protein, at any length, is scored by the same function. 2,755 of
  the 62,727 benchmark variants (102 proteins) have no Brandes entry with exactly the
  benchmark sequence; they are dropped from training and calibration. The near-match column
  `esm1b_brandes_oriented_near` is not used.
- Test families: MMseqs2 clusters of the new proteins among themselves (30% identity, 80%
  coverage, as for the benchmark), 8,579 families. Benchmark families are `fam_mm030`.
  Clustering the new proteins jointly with the benchmark would have put 1,581 of them
  (656 families) into a family with a benchmark protein; they stay in the primary
  population, and the secondary population excludes them.
- Files evaluated (SHA-256):

| File | SHA-256 |
|---|---|
| `data/processed/unseen_variants.parquet` | `7554cdadc4ebe5dd6ee7853682559d4b537e3b9735054e9ffaab7e285538c272` |
| `data/processed/unseen_proteins.parquet` | `a315afda6945f3f12ded50acad7a7e0c153697dcb72fad3eb0cb63dd9028f410` |
| `data/processed/unseen_families.parquet` | `cf4a83ac9ec237b0188036fc8d6460a72365662dc69f856b3974330d70cb949c` |
| `data/processed/unseen_benchmark_esm1b.parquet` | `c2973dd463a22cbe6751770365acbb926c735c5b8741eeafbec169f1bb2e8e07` |
| `data/esm2_t12_35M_meanpool_unseen.npz` | `261e79f6538052a34efa9568a2cf3bb9004b692e0b4b4e79ecc66bd7adb1e39c` |

## Splits and fitting

For each seed 0 to 9: the training families are the S1 training draw of the benchmark
(40% of benchmark families); every other benchmark family is a calibration family; the
test families are all families of new proteins (the same in every seed). The base model
is fitted on the training families' variants; calibration uses the calibration families'
variants; the density ratio is fitted between the calibration families' fitting third and
the test families (unlabeled), cross-fitted over five folds of test families, as in the
benchmark.

- Base model: raw score only (the deployment-realistic configuration).
- Descriptors: alignment-free (the `--no-msa` configuration of the development run):
  maximum identity to training proteins, embedding distance to the nearest training
  protein, log length. Alignment depth is not available for new proteins. GWCP bins are
  maximum identity (zero, low, high) crossed with embedding-distance terciles; the gate
  uses identity and embedding distance.
- The gate's parameters are tuned, as on the benchmark, on an inner split of the training
  families shifted by maximum identity and alignment depth; these are benchmark families,
  which have alignments. No new protein enters the tuning.

## Methods (all evaluated and all reported)

Variant-level split and Mondrian split (B1, B2); HCP, single and repeated subsampling
(B3); GWCP (B4); the gate with random and support-ranked abstention (B5); ProtCOVER-W
(`P_w`), with the gate (`P_gate`, `P_full`); ProtCOVER-D (`C_supp`); ProtCOVER
(`C_ratio`). Marginal and class-conditional versions, alpha 0.05, 0.1 and 0.2. Settings as
in `configs/default.yaml` at the commit that adds this file.

## Populations

The sets are those of the primary analysis; only the families evaluated differ.

| Population | Test families | With pathogenic variants | Variants | Pathogenic |
|---|---|---|---|---|
| Primary (`U`): all test families | 8,579 | 911 | 101,197 | 9,555 |
| Secondary (`Ustrict`): no protein has any MMseqs2 hit with E <= 1e-3 to a benchmark protein | 4,448 | 354 | 33,998 | 2,042 |
| Exploratory (`Ulen`): longest protein at most 2,302 residues, the longest benchmark protein | 8,356 | 785 | 68,266 | 3,229 |

The exploratory population has no hypothesis. It is reported because 287 new proteins are
longer than every benchmark protein and hold 6,313 of the 9,555 pathogenic variants;
family weighting counts each of their families once.

## Endpoints

Alpha = 0.1 is primary; 0.05 and 0.2 are secondary.

1. Family-weighted pathogenic and benign coverage (Cov_1, Cov_0) of class-conditional
   methods, mean over seeds, with family-bootstrap intervals (1,000 resamples of test
   families within seed) and the seed-to-seed standard deviation.
2. Worse-class shortfall max(0, 0.90 - min(Cov_0, Cov_1)).
3. Mean set size and singleton error; set size at matched pathogenic coverage.
4. Shift diagnostics: effective sample size of the calibration weights, as a fraction of
   the number of calibration families.

## Hypotheses (stated before the evaluation)

A difference in shortfall smaller than 0.01 is treated as no difference.

- U1. Variant-level Mondrian split (B2) covers less than 0.90 of pathogenic variants per
  test family (the upper end of the bootstrap interval is below 0.90).
- U2. ProtCOVER's worse-class coverage is at least 0.89.
- U3. ProtCOVER's worse-class shortfall exceeds neither HCP's nor GWCP's by 0.01 or more.
- U4. ProtCOVER's worse-class shortfall is smaller than GWCP's by more than 0.01.

Each hypothesis is reported as supported or not, whatever the result, for the primary and
secondary populations; the primary population decides.

## Commands

```
python scripts/run_unseen.py --tag unseen          # seeds 0 to 9
python scripts/summarize.py --tag unseen --skip-support
python scripts/score_unseen.py --tag unseen        # U1 to U4 as stated above
```

`scripts/score_unseen.py` also reports the predictor's per-protein AUC on the new proteins
and on the benchmark proteins, as a descriptive statistic that is not an endpoint.

## Known caveats (stated in advance)

- Recent ClinVar classifications can use computational evidence (PP3/BP4), which may
  inflate the predictor's apparent accuracy, and the new proteins are on average less
  studied than the benchmark's.
- Alignment depth, the descriptor that drove the correction under descriptor shift on the
  benchmark, is unavailable here. In the dry run's label-free diagnostics, the
  alignment-free descriptors of the test families were close to those of the calibration
  families (effective sample size about 0.99 of the calibration families), so the
  reweighted and conditional methods may differ little from HCP. The evaluation then tests
  validity under the real shift more than it tests the correction.
- Only 9.4% of the test variants are pathogenic (against about half on the benchmark);
  class-conditional calibration targets each class separately, and Cov_1 averages over
  the 911 families with pathogenic variants.
- Primary-population test families may be homologous to benchmark families; the
  secondary population removes every detectable homolog.
- 392 genes were lost because their canonical isoform has no RefSeq protein with the same
  numbering, and 2,059 labelled VariationIDs of the kept genes are described only on other
  isoforms; positions were kept exact rather than lifted over.
- There is one predictor and one test set, so this evaluation has less replication than
  the benchmark; seeds vary only the training and calibration draws.
