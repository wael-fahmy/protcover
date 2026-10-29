# External temporal test set

`scripts/build_external.py` builds `data/processed/external_temporal.parquet`
(gitignored): ClinVar missense variants in the 2,525 proteins of the ProteinGym
v1.3 clinical benchmark that **entered ClinVar after the 2023-06-01 release**,
labelled pathogenic (1) or benign (0), and scored with the precomputed ESM-1b
log-likelihood ratios of Brandes et al. (2023). No (protein, amino-acid change)
in this table occurs in the benchmark, whatever its benchmark label.

The build evaluates no conformal method and computes no coverage. It only checks
the predictor itself (AUC of the oriented ESM-1b score) and the score
orientation. Every number below is copied from `build_summary.json`, which the
build writes.

## Rebuild

```bash
bash data/external/download.sh                            # ClinVar files into data/raw/external/ (about 1.4 GB)
.venv/Scripts/python.exe -I scripts/build_external.py     # .venv/bin/python on Linux or macOS
```

The build needs `data/processed/variants.parquet` and `proteins.parquet`
(`scripts/build_dataset.py`). On first use it fetches the ESM-1b entries it needs
by HTTP range requests (`scripts/external_esm1b.py`) and caches them in
`data/raw/external/esm1b/` (about 330 MB); later builds read the cache. A build
takes about two minutes once the inputs are local.

Files written:

* `data/processed/external_temporal.parquet`: the test set (columns below).
* `data/external/build_summary.json`: input checksums, counts after every
  filter, and the checks reported here.
* `data/external/esm1b_matches.tsv`: for every protein that reached the scoring
  step, the Brandes et al. entry whose sequence equals the benchmark sequence
  (or `no_exact_match`), how many entries were candidates and how many were tried.

## Sources

| Input | Location | Date | SHA-256 |
|---|---|---|---|
| Snapshot boundary | `https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/2023/variant_summary_2023-06.txt.gz` | monthly release of 2023-06-01 | `fb8be67b506d52920f0f610f2695902756af2f2c9214135c991da8c15bd542d4` |
| Next release (flag only) | `https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/2023/variant_summary_2023-07.txt.gz` | monthly release of 2023-07-06 | `1d551964acdbe2099f2d740be33c39642ef2e1b946b694a0d4a98d24b032392e` |
| Current labels | `https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz` | Last-Modified 2026-10-26 13:50:58 GMT, downloaded 2026-10-29 | `657b47d0a00ba3ad4b3ddf03e160835c8a4a3c7a2e130c92c904b70021a35703` |
| Current protein HGVS | `https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/hgvs4variation.txt.gz` | Last-Modified 2026-10-25 03:05:32 GMT, downloaded 2026-10-29 | `3ba31904f004573af2bc2222f38c4ad3bfd0e995d636f8ff95bd650c3f042f17` |
| ESM-1b LLRs | Hugging Face Space `ntranoslab/esm_variants`, revision `90e7e89167f7ba5ccce395855e3bc8dc258fa1f4`, file `ALL_hum_isoforms_ESM1b_LLR.zip` (1,335,361,543 bytes, 42,286 entries) and `isoform_list.csv` | Space commit of 2023-10-24 | Git LFS `42b90ce74a007ce7147f8e4be4035fe7f11881215fecce0983441cc229207cd0` |

`download.sh` also checks the MD5 files NCBI publishes for the two current files
(`c79b46f3c02ef544608e0d9a83cac46e` for `variant_summary.txt.gz`,
`5e6f8b17a7cc76306821ccd5227f0320` for `hgvs4variation.txt.gz`) and writes
`data/raw/external/SHA256SUMS`. The ESM-1b archive is never downloaded whole:
the build reads its central directory (SHA-256
`0a81e1dd0b68408314e1809368de97b5e1ca96c11e9bf0f1580607e667116ddc`) and checks
every fetched entry against the CRC-32 stored there.

**Exact reproduction needs the files above.** NCBI overwrites
`variant_summary.txt.gz` and `hgvs4variation.txt.gz` every week and does not
archive `hgvs4variation`, so a later download gives slightly different counts.
`download.sh` never replaces a file that already exists; keep
`data/raw/external/` to rebuild the same table.

ESM-1b scores: N. Brandes, G. Goldman, C. H. Wang, C. J. Ye, V. Ntranos.
Genome-wide prediction of disease variant effects with a deep protein language
model. *Nature Genetics* 55, 1512-1522 (2023). The Space declares the licence
CC BY-NC 4.0.

## Snapshot boundary

ProteinGym does not state the date of the ClinVar download behind its clinical
substitutions. "New" therefore means a VariationID that does not occur anywhere
(any assembly, type or classification) in the archived monthly release of
2023-06-01, not a guessed date.

As a check, the build maps the 62,727 benchmark variants to current
VariationIDs with the same protein rules as below. 62,667 map; of these, 52 map
only to VariationIDs absent from the 2023-06-01 release, and 4 only to
VariationIDs absent from the 2023-07-06 release. The benchmark snapshot is thus
somewhat later than 2023-06-01. This cannot leak benchmark variants into the
test set, because every (protein, change) in the benchmark is removed (step
11). For a stricter boundary, drop the 138 rows with `in_2023_07_release`.

## Construction and counts

| Step | VariationIDs | Rows | Pathogenic | Benign | Proteins |
|---|---|---|---|---|---|
| 1. Current `variant_summary`, all rows | 4,690,666 | | | | |
| 2. Has a GRCh38 row | 4,620,295 | | | | |
| 3. `Type` is `single nucleotide variant` | 4,270,212 | | | | |
| 4. Review status of at least one star | 3,774,396 | | | | |
| 5. P/LP or B/LB | 1,490,093 | 1,490,093 | 179,877 | 1,310,216 | |
| 6. Absent from the 2023-06-01 release ("new") | 603,250 | 603,250 | 68,898 | 534,352 | |
| 7. Protein HGVS on a benchmark RefSeq accession | 198,730 | 198,730 | 33,293 | 165,437 | |
| 8. Missense on that protein | 28,593 | | | | |
| 9. Wild type matches, change unambiguous | 28,592 | 28,628 | 15,088 | 13,540 | 2,334 |
| 10. Not at position 1 | 28,089 | 28,125 | 14,593 | 13,532 | 2,315 |
| 11. (Protein, change) not in the benchmark | 27,702 | 27,738 | 14,248 | 13,490 | 2,312 |
| 12. One row per (protein, change) | | 27,621 | 14,148 | 13,473 | 2,312 |
| 13. Exact ESM-1b sequence match | | 25,981 | 13,312 | 12,669 | 2,210 |

Rows from step 9 on are (VariationID, protein) pairs, then (protein, change).

1. Current `variant_summary.txt.gz`, columns `VariationID`, `Type`, `Assembly`,
   `ClinicalSignificance` (the germline classification), `ReviewStatus`.
2. Rows with `Assembly == GRCh38`.
3. Single nucleotide variants only.
4. `ReviewStatus` is one of `criteria provided, single submitter`,
   `criteria provided, multiple submitters, no conflicts`,
   `reviewed by expert panel`, `practice guideline`. Conflicting and
   no-criteria records are excluded here.
5. `ClinicalSignificance` exactly `Pathogenic`, `Likely pathogenic` or
   `Pathogenic/Likely pathogenic` (label 1), or exactly `Benign`,
   `Likely benign` or `Benign/Likely benign` (label 0). Uncertain significance
   and compound values (for example `Pathogenic, low penetrance` or
   `Pathogenic; drug response`) are excluded. 458 VariationIDs had two GRCh38
   rows with the same classification and were collapsed.
6. VariationID not in `variant_summary_2023-06.txt.gz` (2,264,956 VariationIDs).
7. `hgvs4variation.txt.gz` has a `ProteinExpression` on the RefSeq protein
   accession of a benchmark protein, ignoring the version.
8. If ClinVar gives an expression on the benchmark's exact accession version,
   only those are used; otherwise expressions on other versions are used. A
   missense change is `p.Xxx123Yyy` (optionally in parentheses) with two
   different standard amino acids; synonymous, nonsense, frameshift and other
   changes are dropped (170,409 VariationID-protein pairs).
9. The wild-type residue must equal the benchmark sequence at that position (1
   pair dropped) and all retained expressions must agree on one change (0
   dropped). Every row of the final table uses the benchmark's exact accession
   version.
10. Changes of the initiator methionine are start-loss variants, not missense,
    and the benchmark has none (503 rows dropped).
11. Any (protein, change) present in the benchmark is removed, whatever its
    label (387 rows).
12. VariationIDs that give the same amino-acid change are merged into one row
    (109 final rows merge more than one). Rows whose VariationIDs disagree on
    the label would be dropped; there were none. The representative VariationID
    has the highest review status, then the smallest ID; all are kept in
    `variation_ids`.
13. Scores come from the Brandes et al. entry whose sequence (read from the
    column headers) equals the benchmark sequence exactly. Candidates are the
    mapped UniProt accession (`data/uniprot_map.tsv`) and its isoforms, then
    every entry listed under the same gene symbol. Matches: 1,952 UniProt
    canonical, 252 UniProt isoform, 6 by gene symbol. 102 proteins have no
    exact match (28 of them have no UniProt mapping, 26 have no candidate entry
    at all) and are dropped with their 1,640 variants.

## The final table

25,981 variants in 2,210 proteins: 13,312 pathogenic and 12,669 benign.
1,233 proteins have both classes, 314 only pathogenic, 663 only benign. The
median protein has 6 variants (maximum 344); 713 proteins have at least 10.

* Review status: 23,061 single submitter, 2,575 multiple submitters without
  conflicts, 345 expert panel, none practice guideline.
* Classification: 10,984 Likely benign, 10,356 Likely pathogenic, 2,282
  Pathogenic, 1,465 Benign, 674 Pathogenic/Likely pathogenic, 220
  Benign/Likely benign.
* **Proteins longer than 1,022 residues** (ESM-1b's context) are kept and
  flagged: 474 proteins, 11,154 variants (4,316 pathogenic, 6,838 benign).

| Column | Meaning |
|---|---|
| `protein_id` | benchmark RefSeq protein ID, e.g. `NP_000007.1` |
| `uniprot`, `gene` | UniProt accession from `data/uniprot_map.tsv`; ClinVar gene symbol |
| `mutant` | benchmark format: wild type, 1-based position in the benchmark sequence, mutant (e.g. `F55L`) |
| `position` | the position in `mutant` |
| `label` | 1 = P/LP, 0 = B/LB |
| `esm1b_oriented` | `-esm1b_llr`; higher = more pathogenic, as the benchmark's `ESM1b` column |
| `esm1b_llr` | Brandes et al. LLR, log p(mutant) - log p(wild type), 3 decimals |
| `variation_id`, `variation_ids`, `n_variation_ids` | representative ClinVar VariationID; all merged IDs (`;`-separated) and their number |
| `review_status`, `clinsig`, `last_evaluated` | of the representative VariationID, from the current `variant_summary` |
| `length`, `over_1022` | benchmark sequence length; length > 1,022 |
| `np_accession`, `np_version_match` | RefSeq accession and version of the ClinVar expression used; equal to the benchmark version |
| `brandes_entry` | the Brandes et al. entry used (UniProt accession or isoform) |
| `in_2023_07_release` | some merged VariationID is already in the 2023-07-06 release (138 rows) |
| `aa_change_in_2023_06_release` | another VariationID that is in the 2023-06-01 release maps to the same (protein, change) (356 rows) |

## Checks

**Orientation.** On the benchmark variants of the 2,210 matched proteins,
`-LLR` from the Brandes files is compared with the benchmark's oriented `ESM1b`
column:

| Subset | Proteins | Variants | Max abs. difference | Median abs. difference | Pearson r |
|---|---|---|---|---|---|
| Length <= 1,022 | 1,736 | 35,805 | 0.00053 | 0.00025 | 1.0000 |
| Length > 1,022 | 474 | 22,521 | 13.16 | 0.0146 | 0.9987 |
| All | 2,210 | 58,326 | 13.16 | 0.0004 | 0.9995 |

For proteins up to 1,022 residues the two agree to the 3-decimal rounding of
the Brandes files, which confirms `esm1b_oriented = -LLR` and the sign. For
longer proteins the values differ (1,602 of 22,521 by more than 0.1). Since
the differences occur only beyond the model's 1,022-residue context, the two
sources evidently handle long sequences differently. **For those 474
proteins, `esm1b_oriented` is not the same function as the benchmark `ESM1b`
column**, which matters if one is calibrated on the benchmark and tested here.

**Predictor sanity check** (per-protein AUC over proteins with both classes;
overall AUC pools all variants):

| Set | Per-protein mean AUC | Proteins | Overall AUC | Variants |
|---|---|---|---|---|
| External, all | 0.927 | 1,233 | 0.927 | 25,981 |
| External, length <= 1,022 | 0.930 | 927 | 0.919 | 14,827 |
| Benchmark, same 2,210 proteins | 0.893 | 2,210 | 0.909 | 58,326 |
| Benchmark, same proteins, length <= 1,022 | 0.899 | 1,736 | 0.899 | 35,805 |

## Caveats

* **Temporal separation is about ClinVar availability, not classification
  date.** 3,329 rows have a `last_evaluated` date before 2023-06-01 (491 have
  none): they were classified earlier but first submitted to ClinVar later.
* 356 rows (`aa_change_in_2023_06_release`) have an amino-acid change that was
  already in ClinVar on 2023-06-01 under another VariationID (a different
  nucleotide change) but is not in the benchmark.
* ACMG/AMP evidence codes PP3 and BP4 let submitters use computational
  predictions as supporting evidence, so recent classifications may partly
  depend on predictors like the one scored here. The higher AUC on this set
  than on the benchmark is consistent with that, but we did not test it.
* The class balance differs from the benchmark's, and many proteins have
  only one class.
* Six VariationIDs lie in exons shared by several benchmark proteins and appear
  under more than one protein (28 rows).
* No filter on allele origin is applied; `ClinicalSignificance` is ClinVar's
  germline classification (somatic classifications are in separate columns).
* LLRs are stored with 3 decimals, so ties are more frequent than in the
  benchmark's `ESM1b` column.
