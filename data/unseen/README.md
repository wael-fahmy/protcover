# Unseen-protein test set

`scripts/build_unseen.py` builds a test set of ClinVar missense variants in
human proteins that are **not among the 2,525 proteins of the ProteinGym v1.3
clinical benchmark** used for development. Variants are labelled pathogenic (1)
or benign (0) and scored with the precomputed ESM-1b log-likelihood ratios of
Brandes et al. (2023). The new proteins are grouped into families with the
benchmark's MMseqs2 parameters, and every new protein is searched against every
benchmark protein, so that its distance to any set of training proteins can be
computed later.

The set was built before any method was run on it. The build computes no
label-dependent performance statistic (no AUC, accuracy, coverage or score
distribution split by label), runs no conformal method and fits no base model.
It reports counts and class balance only. Every number below is copied from
`build_summary.json`, which the build writes, or counted in the output tables.

## Rebuild

```bash
bash data/external/download.sh                                # ClinVar files (shared with the temporal external set)
bash data/unseen/download.sh                                  # HGNC and RefSeq protein files into data/raw/unseen/ (about 52 MB)
.venv/Scripts/python.exe -I scripts/build_unseen.py           # tables, families, homology, summary (.venv/bin/python on Linux or macOS)
.venv/Scripts/python.exe -I scripts/embed_esm2_unseen.py      # ESM2 embeddings of the new proteins
```

The build needs `data/processed/proteins.parquet` and `variants.parquet`
(`scripts/build_dataset.py`) and MMseqs2 (`tools/mmseqs`). It fetches the
ESM-1b entries it needs by HTTP range requests (`scripts/external_esm1b.py`)
and caches them in `data/raw/external/esm1b/` (about 1.6 GB for the 15,962
entries now cached; the first fetch took about 25 minutes). With the cache in
place the build takes 7 to 10 minutes on 4 threads. Rerunning it gives the same
tables. The embeddings took about two hours on 4 CPU threads (7,029 s, with
other jobs running on the machine); a rerun skips them while the stored
sequence checksums match.

Files written:

| File | Content |
|---|---|
| `data/processed/unseen_proteins.parquet` | one row per new protein |
| `data/processed/unseen_variants.parquet` | one row per (protein, amino-acid change) |
| `data/processed/unseen_families.parquet` | one row per family of new proteins, with benchmark-homolog flags |
| `data/processed/unseen_homology.parquet` | MMseqs2 hits of every new protein (query) against all benchmark proteins (target) |
| `data/processed/unseen_benchmark_esm1b.parquet` | Brandes et al. ESM-1b score of every benchmark variant (see below) |
| `data/esm2_t12_35M_meanpool_unseen.npz` | ESM2 embeddings of the new proteins, same format as `data/esm2_t12_35M_meanpool.npz` |
| `data/unseen/build_summary.json` | input checksums, counts after every step, checks |
| `data/unseen/gene_mapping.tsv` | per candidate gene: UniProt candidates, their status, the entry chosen |
| `data/unseen/benchmark_esm1b_matches.tsv` | per benchmark protein: the Brandes et al. entry with exactly its sequence, or why there is none |
| `data/unseen/esm2_embed_summary.json` | embedding settings and the check that benchmark embeddings are reproduced |

## Sources

| Input | Location | Date | SHA-256 |
|---|---|---|---|
| Labels | ClinVar `pub/clinvar/tab_delimited/variant_summary.txt.gz` | Last-Modified 2026-10-26 13:50:58 GMT | `657b47d0a00ba3ad4b3ddf03e160835c8a4a3c7a2e130c92c904b70021a35703` |
| Protein HGVS | ClinVar `pub/clinvar/tab_delimited/hgvs4variation.txt.gz` | Last-Modified 2026-10-25 03:05:32 GMT | `3ba31904f004573af2bc2222f38c4ad3bfd0e995d636f8ff95bd650c3f042f17` |
| Gene, GeneID and UniProt accessions | HGNC monthly archive `hgnc_complete_set_2026-10-06.tsv` (storage.googleapis.com/public-download-files/hgnc/archive/archive/monthly/tsv/) | 2026-10-06 | `8bf6f686e640dbc864a346484626953621b0500d900d26cd085a35c2782d725d` |
| RefSeq protein sequences | NCBI `refseq/H_sapiens/mRNA_Prot/human.{1..19}.protein.faa.gz`, RefSeq release 237 (70,997 NP sequences) | Last-Modified 2026-10-27 | per file in `build_summary.json` and `data/raw/unseen/SHA256SUMS` |
| ESM-1b LLRs | Hugging Face Space `ntranoslab/esm_variants`, revision `90e7e89167f7ba5ccce395855e3bc8dc258fa1f4`, `ALL_hum_isoforms_ESM1b_LLR.zip` (42,286 entries, 20,335 canonical) and `isoform_list.csv` | Space commit of 2023-10-24 | Git LFS `42b90ce74a007ce7147f8e4be4035fe7f11881215fecce0983441cc229207cd0` |

The ClinVar files are the ones used for the temporal external set
(`data/external/README.md`). NCBI overwrites them, and the RefSeq protein files,
at every release; keep `data/raw/` to rebuild the same tables. The HGNC file is
a permanent monthly archive. ESM-1b scores: N. Brandes, G. Goldman, C. H. Wang,
C. J. Ye, V. Ntranos, *Nature Genetics* 55, 1512-1522 (2023); the Space declares
the licence CC BY-NC 4.0.

## Construction

### Variants

The variant filters and labels are those of `scripts/build_external.py`
(imported from it), without its date filter: every VariationID of the current
release counts.

| Step | VariationIDs | Pathogenic | Benign |
|---|---|---|---|
| 1. Current `variant_summary`, all rows | 4,690,666 | | |
| 2. Has a GRCh38 row | 4,620,295 | | |
| 3. `Type` is `single nucleotide variant` | 4,270,212 | | |
| 4. Review status of at least one star | 3,774,396 | | |
| 5. P/LP or B/LB | 1,490,093 | 179,877 | 1,310,216 |
| 6. Has a protein HGVS on a RefSeq `NP_` accession | 982,016 | 135,428 | 846,588 |
| 7. Missense on some RefSeq protein | 209,525 | 62,162 | 147,363 |

Steps 1 to 5 are exactly steps 1 to 5 of the temporal set: GRCh38 rows, single
nucleotide variants, review status `criteria provided, single submitter`,
`criteria provided, multiple submitters, no conflicts`, `reviewed by expert
panel` or `practice guideline`, and a germline classification of exactly
`Pathogenic`, `Likely pathogenic` or `Pathogenic/Likely pathogenic` (label 1) or
`Benign`, `Likely benign` or `Benign/Likely benign` (label 0). A missense change
is `p.Xxx123Yyy` (optionally in parentheses) with two different standard amino
acids.

### Proteins: genes outside the benchmark, one protein per gene

| Step | Genes |
|---|---|
| a. Approved HGNC protein-coding genes | 19,299 |
| b. With a labelled missense ClinVar allele (by the allele's ClinVar GeneID) | 15,323 |
| c. Not a benchmark gene | 12,809 |
| d. Has a UniProt accession scored by Brandes et al. as a canonical entry | 12,777 |
| e. That entry has a RefSeq protein with the same numbering | 12,385 |
| f. At least one ClinVar change placed on it | 12,253 |

**Benchmark genes (step c).** A gene is excluded if any of the following
holds: its symbol is a benchmark gene symbol (from `proteins.parquet`, or the
ClinVar gene of a benchmark RefSeq protein); its NCBI GeneID is the ClinVar gene
of a benchmark RefSeq protein (the GeneID of most ClinVar alleles described on
that accession); one of its UniProt accessions is a benchmark accession (from
`data/uniprot_map.tsv`, or the Brandes et al. entry whose sequence equals the
benchmark sequence, isoform suffix removed); or, through HGNC, it shares a
GeneID or UniProt accession with such a gene. 2,514 genes were excluded, all by
symbol (2,511 also by GeneID, 2,486 also by UniProt accession); no gene was
excluded by one criterion alone. The ClinVar gene matters for the 19
benchmark proteins without a gene symbol in `proteins.parquet` (18 of them also
lack a UniProt mapping; for example the benchmark's PTEN, PAX6, WT1 and TAP1
isoforms): their gene, and so its exclusion, comes from ClinVar. No new protein shares a
sequence, gene symbol or UniProt accession with a benchmark protein (checked in
the build).

**Choice of the protein (step d).** The protein of a gene is the UniProt
accession listed for it by HGNC that Brandes et al. scored as a canonical
sequence, that is an archive entry without an isoform suffix. Brandes et al.
scored the canonical sequence and the isoforms of every reviewed human UniProt
entry of their UniProt release, so the canonical entry is the UniProt canonical
isoform of that release. The protein sequence is the sequence Brandes et al.
scored, read from the column headers of their entry. 29 genes have no canonical
entry (mostly selenoproteins, which Brandes et al. did not score) and 3 have no
UniProt accession in HGNC. Nine genes have two usable canonical entries (for
example CALCA with calcitonin and CGRP); the one onto which more ClinVar
VariationIDs can be placed is used, then the longer one. Seven candidate
entries have exactly the sequence of another candidate under a different
accession and are merged into it (named by the smallest accession); 17 entries
serve more than one gene (duplicated genes with one UniProt entry, such as
CALM1, CALM2 and CALM3 or AMY1A and AMY1B), so the 12,225
final proteins come from 12,246 genes.

**Placement of ClinVar changes (step e).** ClinVar describes a change on
RefSeq proteins, while the scores refer to the UniProt sequence. A change is
placed only through a RefSeq protein (release 237) with the same numbering as
the entry, so no position is lifted over an alignment:

* a RefSeq protein whose sequence is identical to the entry (12,190 entries); or
* if there is none, a RefSeq protein of the same gene (its ClinVar GeneID) with
  the same length that differs from the entry only by substitutions at no more
  than max(2, 1% of) positions (191 entries; 153 differ at one position, for
  example BRCA2, ATRX and APOB, and NEB at two; here UniProt's canonical
  sequence has a different residue than every RefSeq protein of the gene).

For each (VariationID, protein), the rule of the temporal set is applied to the
expressions on those RefSeq accessions: if ClinVar gives an expression on the
release-237 version of the accession, only those are used, otherwise
expressions on other versions are used. The change must be missense, **its
wild-type residue must equal the entry's residue at that position**, and all
retained expressions must agree on one change. 392 genes are dropped because
their canonical entry has no RefSeq protein with the same numbering (isoforms
that differ by insertions or deletions, for example ARID1B, MACF1, DST, BPTF
and ZNF469), and 132 because no labelled change falls on such a RefSeq protein.

| Step | (VariationID, protein) pairs or rows | Pathogenic | Benign | Proteins |
|---|---|---|---|---|
| 8. Pairs with a protein HGVS on a RefSeq protein with the entry's numbering | 421,417 | | | 12,232 |
| 9. Missense there, wild type matches the entry, change unambiguous | 101,865 | 10,038 | 91,827 | 12,232 |
| 10. Not at position 1 (start-loss) | 101,533 | 9,751 | 91,782 | 12,225 |
| 11. One row per (protein, change); label conflicts dropped | 101,197 | 9,555 | 91,642 | 12,225 |
| 12. ESM-1b LLR present, wild type of the entry matches | 101,197 | 9,555 | 91,642 | 12,225 |

At step 9, 319,484 pairs were not missense on the entry (synonymous, nonsense,
frameshift and other changes), **68 pairs had a wild-type mismatch** (a change
at a position where the RefSeq protein and the entry differ, or an expression
on an older accession version) and none were ambiguous. Step 11 merges 324
rows with more than one VariationID; no (protein, change) had conflicting
labels. The representative VariationID has the highest review status, then the
smallest ID; all are kept in `variation_ids`. Step 12 repeats the wild-type
check against the LLR entry itself; it dropped nothing.

Of the 105,524 labelled missense VariationIDs whose ClinVar gene is one of the
final genes, 101,853 were placed, 2,059 have no expression on a RefSeq protein
with the entry's numbering (they are described only on other isoforms) and
1,612 have one there but were dropped at step 9. Three placed
VariationIDs belong to an overlapping gene. Nine VariationIDs lie in exons of
two final proteins and appear under both (18 rows).

### Scores

`esm1b_oriented = -esm1b_llr`, where `esm1b_llr` is the Brandes et al. value
log p(mutant) - log p(wild type) at that position of the entry (3 decimals).
Higher means more pathogenic, as in the benchmark's `ESM1b` column. For
proteins up to 1,022 residues the Brandes et al. values equal the benchmark's
`ESM1b` column to rounding; for longer proteins they differ (see the benchmark
scores below), so the same source must be used on both sides.

### Families and homology to the benchmark

Families are MMseqs2 clusters of the **new proteins among themselves** with the
benchmark parameters (`protcover.families.cluster`: minimum identity 0.3,
coverage 0.8, coverage mode 0). Benchmark families are not touched. Family IDs
are the MMseqs2 representative protein, a UniProt accession, so they cannot
collide with benchmark family IDs (RefSeq accessions).

Homology: `protcover.families.search` of every new protein (query) against all
2,525 benchmark proteins (target) with the settings of `homology.parquet`
(sensitivity 7.5, hits stored up to E <= 10, at most 500 targets per query),
with identity, alignment length, query and target coverage, E-value and bit
score. Each family is flagged

* `has_benchmark_homolog`: some member has a hit at E <= 1e-3 with an
  identity of at least 0.3 and coverage of at least 0.5 of both sequences (the
  benchmark's definition of a significant homolog, `families.homology` in
  `configs/default.yaml`);
* `has_benchmark_hit_e1e-3`: some member has any hit at E <= 1e-3, with no
  identity or coverage threshold (the strictest definition: a shared domain
  suffices);
* `has_benchmark_hit_cluster_thresholds`: some member has a hit at E <= 1e-3
  with identity >= 0.3 and both coverages >= 0.8, so it could have joined a
  benchmark family had the two sets been clustered together.

`max_id_to_benchmark` is the highest identity of a hit at E <= 1e-3 (no
coverage threshold), which is how `protcover.families.max_identity` defines
`max_id_to_train`. For a subset of benchmark proteins (for example the training
families of one split), filter `unseen_homology.parquet` by `target`. The
E-values were computed against all 2,525 benchmark proteins; a search against a
smaller target set gives slightly smaller E-values, so a few hits just above
1e-3 here would pass there. For exact agreement with `max_identity`, rerun it
on the subset.

### ESM2 embeddings

`scripts/embed_esm2_unseen.py` imports the functions of `scripts/embed_esm2.py`
(esm2_t12_35M_UR50D, layer 12, windows of 1,022 residues with stride 511 for
longer sequences, residue representations averaged over windows, then mean
pooled) and writes the same keys as the benchmark file plus `sequence_sha1`.
Before embedding, it re-embeds six benchmark proteins (four short, two just over
1,022 residues) and checks them against `data/esm2_t12_35M_meanpool.npz`
(largest absolute difference 1.2e-6).

## The final tables

**101,197 variants in 12,225 proteins (12,246 genes): 9,555 pathogenic and
91,642 benign (fraction pathogenic 0.094).** 818 proteins have both classes,
186 only pathogenic and 11,221 only benign. The median protein has 4 variants
(maximum 1,461, FBN1); 2,160 proteins have at least 10.

* Review status: 75,647 rows with one star, 24,899 with two, 651 with three
  (expert panel), none with four.
* Classification: 68,792 Likely benign, 15,486 Benign, 7,364 Benign/Likely
  benign, 5,247 Likely pathogenic, 2,395 Pathogenic, 1,913
  Pathogenic/Likely pathogenic.
* Placement: 97,266 rows through an identical RefSeq protein, 3,931 through a
  same-length RefSeq protein with a few substitutions (`np_match`).
* **Length.** 1,614 proteins are longer than 1,022 residues (51,436 variants:
  6,647 pathogenic, 44,789 benign). The longest benchmark protein has 2,302
  residues; **287 new proteins are longer than that, and they carry 32,659
  variants, including 6,313 of the 9,555 pathogenic ones** (FBN1, NF1, COL7A1,
  USH2A, NOTCH3, RYR2, RYR1, NSD1 and PKD1 are all among them).
* Pathogenic variants are concentrated: FBN1 has 1,395, NF1 405, COL7A1 294,
  USH2A 290, NOTCH3 255, RYR2 239, RYR1 222, ACTA1 188, NSD1 174, PKD1 154.

**Families.** 8,579 families, 6,976 of them singletons; the largest has 294
proteins (olfactory receptors). The median family has 4 variants (maximum
1,839); 2,204 families have at least 10 variants and 797 have both classes.

**Distance to the benchmark.**

| Families (of 8,579) | With the property | Without | Variants in families without |
|---|---|---|---|
| Benchmark homolog: E <= 1e-3, identity >= 0.3, both coverages >= 0.5 (`has_benchmark_homolog`) | 1,161 | 7,418 | 84,490 |
| Any benchmark hit at E <= 1e-3 (`has_benchmark_hit_e1e-3`) | 4,131 | 4,448 | 33,998 |
| Hit at the clustering thresholds: identity >= 0.3, both coverages >= 0.8 (`has_benchmark_hit_cluster_thresholds`) | 656 | 7,923 | 89,327 |

At the protein level, 2,459 new proteins have a benchmark homolog under the
first definition, 6,775 have some hit at E <= 1e-3, and 1,581 have a hit at the
clustering thresholds. 11,160 proteins have at least one stored hit (E <= 10).
The highest identity to a benchmark protein is 0.98 (paralogs such as HBG1, ACTB,
tubulins and SFTPA1).

## Brandes et al. scores for the benchmark variants

For proteins longer than 1,022 residues, the ProteinGym `ESM1b` column and the
Brandes et al. values are different functions, so an evaluation that calibrates
on the benchmark and tests here should use Brandes et al. scores on both sides.
`unseen_benchmark_esm1b.parquet` has one row for each of the 62,727 benchmark
variants (keys `protein_id`, `variant`, as in `variants.parquet`; no labels).

The entry is the Brandes et al. entry whose sequence equals the benchmark
sequence exactly. Candidates, tried in order: the mapped UniProt accession and
its isoforms, entries listed under the same gene symbol (as in
`build_external.py`), then the isoforms of the HGNC UniProt accessions of the
protein's gene (by symbol or ClinVar GeneID), which covers benchmark proteins
without a UniProt mapping. Every variant is then checked: the wild type must
equal the entry's residue at that position.

| Match | Proteins | Variants |
|---|---|---|
| UniProt canonical | 2,139 | 52,835 |
| UniProt isoform | 265 | 6,772 |
| HGNC UniProt accession of the gene | 11 | 232 |
| Entry under the same gene symbol | 8 | 133 |
| **Scored (`scored`)** | **2,423** | **59,972** |
| No entry with exactly the benchmark sequence | 98 | 2,694 |
| No candidate entry at all (readthrough proteins such as FPGT-TNNI3K) | 4 | 61 |

No scored variant failed the wild-type check. Of the 102 unscored proteins, 19
have no UniProt mapping and 24 are longer than 1,022 residues (1,036 unscored
variants are in proteins over 1,022 residues). As a secondary, separately named
column, `esm1b_brandes_oriented_near` gives a score for 35 of the unscored
proteins from a same-length entry that differs by substitutions at 1 (29
proteins), 2 (4) or 3 (2) positions (1,505 variants; variants at a differing
position stay empty); the remaining 1,250 benchmark variants have no Brandes
et al. score. The primary column `esm1b_brandes_oriented` never uses such an
entry.

Consistency of the score function (label-free; `ESM1b_benchmark` is the
benchmark's oriented `ESM1b` column):

| Subset | Proteins | Variants | Pearson r | Spearman rho | Max abs. difference | Median abs. difference | Abs. difference > 0.1 |
|---|---|---|---|---|---|---|---|
| Length <= 1,022 | 1,933 | 37,182 | 1.0000 | 1.0000 | 0.00053 | 0.00025 | 0 |
| Length > 1,022 | 490 | 22,790 | 0.9987 | 0.9990 | 13.16 | 0.0146 | 1,616 |
| All scored | 2,423 | 59,972 | 0.9995 | 0.9996 | 13.16 | 0.0004 | 1,616 |
| Secondary near match | 35 | 1,505 | 0.9996 | 0.9994 | 3.84 | 0.0296 | 298 |

Up to 1,022 residues the two sources agree to the 3-decimal rounding of the
Brandes et al. files; beyond that they differ, as already seen for the temporal
external set.

## Columns

`unseen_proteins.parquet`

| Column | Meaning |
|---|---|
| `protein_id`, `uniprot`, `brandes_entry` | UniProt accession of the canonical entry scored by Brandes et al. (the three are equal) |
| `gene`, `hgnc_id`, `entrez_id` | HGNC symbol, HGNC ID and NCBI GeneID (`;`-separated when one entry serves several genes) |
| `length`, `sequence` | the sequence Brandes et al. scored |
| `over_1022`, `over_benchmark_max_length` | length > 1,022; length > 2,302 (the longest benchmark protein) |
| `n_variants`, `n_path`, `n_benign` | rows in `unseen_variants.parquet` |
| `family` | MMseqs2 family among the new proteins (representative accession) |
| `max_id_to_benchmark` | highest identity of a hit at E <= 1e-3 to any benchmark protein (0 if none) |
| `has_benchmark_homolog`, `has_benchmark_hit_e1e-3` | the protein-level versions of the family flags |
| `refseq_match`, `refseq_proteins`, `refseq_n_diff` | `identical` or `same_length_substitutions`; the RefSeq accessions used for placement; number of differing positions |

`unseen_variants.parquet`

| Column | Meaning |
|---|---|
| `protein_id`, `uniprot`, `gene`, `family` | as above |
| `mutant`, `position`, `wt`, `mt` | benchmark format (`F55L`), 1-based position in the entry sequence |
| `label` | 1 = P/LP, 0 = B/LB |
| `esm1b_oriented`, `esm1b_llr` | `-esm1b_llr` (higher = more pathogenic); Brandes et al. LLR, 3 decimals |
| `variation_id`, `variation_ids`, `n_variation_ids` | representative ClinVar VariationID; all merged IDs (`;`-separated) and their number |
| `review_stars`, `review_status`, `clinsig`, `last_evaluated` | of the representative VariationID |
| `clinvar_gene_symbol` | ClinVar's `GeneSymbol` of the representative VariationID |
| `np_accession`, `np_version_match`, `np_match` | RefSeq accession and version of the ClinVar expression used; whether it is the release-237 version; how that RefSeq protein matches the entry |
| `length`, `over_1022`, `over_benchmark_max_length` | as above |

`unseen_families.parquet`: `family`, `members` (`;`-separated), `n_proteins`,
`n_variants`, `n_path`, `n_benign`, `max_length`, the three flags above,
`max_id_to_benchmark` (any hit at E <= 1e-3) and `max_id_to_benchmark_hc` (hits
that also meet identity >= 0.3 and both coverages >= 0.5), each the maximum
over members.

`unseen_homology.parquet`: `query` (new protein), `target` (benchmark
`protein_id`), `fident`, `alnlen`, `qcov`, `tcov`, `evalue`, `bits`, as in
`homology.parquet`.

`unseen_benchmark_esm1b.parquet`: `protein_id`, `variant`, `position`,
`brandes_entry`, `match_type`, `wt_ok`, `esm1b_brandes_llr`,
`esm1b_brandes_oriented` (primary), `scored`, `miss_reason` (empty when
scored), `near_entry`, `near_n_diff`, `esm1b_brandes_oriented_near`
(secondary), `ESM1b_benchmark`, `length`, `over_1022`.

## Caveats and open problems

* **Length and class balance differ from the benchmark.** A third of the
  variants and two thirds of the pathogenic ones are in proteins longer than
  any benchmark protein, and ESM-1b's context is 1,022 residues. Whether to
  include proteins over 1,022 or over 2,302 residues has to be decided in the
  pre-registration; the flags make either restriction a filter. Overall only
  9.4% of the variants are pathogenic, most proteins have a single class, and
  a few genes hold many of the pathogenic variants.
* **No MSA descriptors.** The benchmark's support descriptors `log_neff` and
  `log_neff_L` come from the ProteinGym MSAs. No such MSAs exist for the new
  proteins, and building them would need UniRef, which we did not download.
  `max_id_to_train` (from `unseen_homology.parquet`) and `emb_dist_to_train`
  (from the ESM2 embeddings) can be computed; `log_neff` and `log_neff_L`
  cannot, which affects the pre-registered feature class of the conditional
  method.
* **Families are new-only.** The new proteins were clustered among
  themselves, as specified, so a new protein with a benchmark hit at the
  clustering thresholds (1,581 proteins in 656 families) would have shared a
  family with a benchmark protein under joint clustering. The flags and
  `max_id_to_benchmark` let an analysis exclude or stratify them. Paralogs of
  benchmark genes are not excluded, only flagged.
* **Selection by annotation.** Genes are lost when their UniProt canonical
  sequence, as scored by Brandes et al., has no RefSeq protein with the same
  numbering (392 genes, for example ARID1B, MACF1, DST, BPTF and ZNF469), and
  2,059 labelled missense VariationIDs of the final genes are described only on
  other isoforms. A lift-over through an alignment could recover them; we kept
  positions exact instead. These losses do not depend on the labels.
* The Brandes et al. canonical sequences come from the UniProt release they
  used, which can differ from today's UniProt canonical sequence; the scored
  sequence is what defines the protein here, and every placed change matches
  it at the wild type.
* ACMG/AMP evidence codes PP3 and BP4 let submitters use computational
  predictions as supporting evidence, so some classifications may partly
  depend on predictors like ESM-1b.
* No filter on allele origin is applied, as in the temporal set.
* LLRs are stored with 3 decimals, so ties are more frequent than in the
  benchmark's `ESM1b` column.
