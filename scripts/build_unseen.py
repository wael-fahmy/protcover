"""Build the unseen-protein ClinVar test set (proteins outside the benchmark).

Outputs
  data/processed/unseen_proteins.parquet      one row per new protein
  data/processed/unseen_variants.parquet      one row per (protein, amino-acid change)
  data/processed/unseen_families.parquet      MMseqs2 families of the new proteins, homolog flags
  data/processed/unseen_homology.parquet      MMseqs2 hits of every new protein against all benchmark proteins
  data/processed/unseen_benchmark_esm1b.parquet  Brandes et al. ESM-1b scores of every benchmark variant
  data/unseen/build_summary.json              input checksums, counts after every filter, checks
  data/unseen/gene_mapping.tsv                per-gene decision (candidate entries, chosen entry, status)
  data/unseen/benchmark_esm1b_matches.tsv     which Brandes et al. entry scores each benchmark protein
The ESM2 embeddings of the new proteins are computed by scripts/embed_esm2_unseen.py.

Variants: the current ClinVar release (all dates), with the variant filters and
labels of scripts/build_external.py: GRCh38 row, single nucleotide variant,
review status of at least one star, germline classification exactly P, LP or
P/LP (label 1) or B, LB or B/LB (label 0), a missense protein HGVS
(p.Xxx123Yyy) on a RefSeq protein, not at position 1, and one row per
(protein, change) with label conflicts dropped.

Proteins: approved HGNC protein-coding genes with ClinVar labelled missense
alleles, minus every gene of the benchmark. A gene is a benchmark gene if its
symbol is a benchmark gene symbol, its NCBI GeneID is the gene of a benchmark
RefSeq protein in ClinVar, or one of its UniProt accessions is a benchmark
UniProt accession (UniProt map, or the Brandes et al. entry whose sequence equals
the benchmark sequence). Each remaining gene gets one protein: its HGNC UniProt
accession scored by Brandes et al. as a canonical sequence (entry without an
isoform suffix); if a gene lists two such accessions, the one onto which more
ClinVar variants can be placed is used. The protein sequence is the sequence
Brandes et al. scored, read from their entry. A ClinVar protein change is placed
on it only through a RefSeq protein (release 237) with the same numbering, so
positions need no lift-over: a RefSeq protein whose sequence is identical or,
if there is none, a RefSeq protein of the same gene with the same length that
differs only by substitutions at no more than max(2, 1% of) positions. The
wild-type residue is checked against the entry at that position, and mismatches
are dropped and counted. Genes whose entry has no such RefSeq protein (isoforms
differing by insertions or deletions) are dropped and counted. Proteins whose
sequence equals a benchmark sequence are dropped. No length cap: Brandes et al.
scored proteins longer than 1,022 residues too.

Scores: ESM-1b LLRs of Brandes et al. (2023), oriented like the benchmark ESM1b
column (higher = more pathogenic, i.e. -LLR).

Families: MMseqs2 clustering of the new proteins among themselves with the
benchmark parameters (30% identity, 80% coverage, coverage mode 0); benchmark
families are not touched. Homology: MMseqs2 search of every new protein against
all benchmark proteins (sensitivity 7.5, hits stored up to E <= 10), as for
data/processed/homology.parquet.

This script computes no label-dependent performance statistic and runs no
conformal method and no base model. Label counts and class balance are reported.

Inputs: bash data/external/download.sh (ClinVar) and bash data/unseen/download.sh
(HGNC, RefSeq proteins); ESM-1b entries are fetched by HTTP range requests on
first use and cached in data/raw/external/esm1b/.

Usage: python -I scripts/build_unseen.py [--config configs/default.yaml] [--threads 4]
"""
import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_external import (AA3, BENIGN, CURRENT_HGVS, CURRENT_VS, MISSENSE_RE, PATHOGENIC,  # noqa: E402
                            STARS, read_tsv, sha256, step)
from external_esm1b import HF_REVISION, HF_SPACE, ZIP_LFS_SHA256, ZIP_NAME, ZIP_SIZE, BrandesLLR  # noqa: E402

from protcover.config import ROOT, load_config, processed_dir  # noqa: E402
from protcover.data import load_proteins, load_variants  # noqa: E402
from protcover.families import cluster, search  # noqa: E402

EXT_RAW = ROOT / "data" / "raw" / "external"
RAW = ROOT / "data" / "raw" / "unseen"
DOC = ROOT / "data" / "unseen"
HGNC_FILE = RAW / "hgnc" / "hgnc_complete_set_2026-10-06.tsv"
REFSEQ_DIR = RAW / "refseq_protein"
NP_RE = r"^(?P<acc>NP_\d+)\.(?P<ver>\d+):(?P<chg>.+)$"


# --------------------------------------------------------------------------- inputs

def read_labels(log):
    """ClinVar labels, exactly the variant filters and labels of build_external.py."""
    vs_cols = ["VariationID", "Type", "Assembly", "ClinicalSignificance", "ReviewStatus", "GeneSymbol", "LastEvaluated"]
    vs = read_tsv(CURRENT_VS, vs_cols).to_pandas()
    vs["VariationID"] = vs["VariationID"].astype(np.int64)
    step(log, "current variant_summary, all rows", vids=vs["VariationID"].unique())
    vs = vs[vs["Assembly"] == "GRCh38"]
    step(log, "GRCh38 rows", vids=vs["VariationID"].unique())
    vs = vs[vs["Type"] == "single nucleotide variant"]
    step(log, "single nucleotide variant", vids=vs["VariationID"].unique())
    vs = vs[vs["ReviewStatus"].isin(STARS)]
    step(log, "review status >= 1 star", vids=vs["VariationID"].unique())
    vs["label"] = np.where(vs["ClinicalSignificance"].isin(PATHOGENIC), 1, np.where(vs["ClinicalSignificance"].isin(BENIGN), 0, -1))
    vs = vs[vs["label"] >= 0]
    n_dup = int(vs["VariationID"].duplicated().sum())
    vs = vs.drop_duplicates("VariationID")
    step(log, "P/LP or B/LB", vids=vs["VariationID"], df=vs, note=f"{n_dup} duplicate GRCh38 rows of a VariationID collapsed")
    return vs.drop(columns=["Type", "Assembly"]).reset_index(drop=True)


def read_np_expressions(path, vids):
    """Every protein HGVS on a RefSeq NP accession for the given VariationIDs.

    Columns: VariationID, gene_id (ClinVar's gene of the allele), acc, ver,
    missense (bool) and, for missense changes, wt, pos, mt (one-letter codes).
    """
    with gzip.open(path, "rt") as g:
        skip = next(i for i, line in enumerate(g) if line.startswith("#Symbol"))
    cols = ["GeneID", "VariationID", "ProteinExpression"]
    reader = pacsv.open_csv(
        pa.input_stream(str(path), compression="gzip"),
        read_options=pacsv.ReadOptions(skip_rows=skip, block_size=1 << 26),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(include_columns=cols, column_types={c: pa.string() for c in cols}),
    )
    ids = pa.array(np.sort(np.asarray(list(vids), dtype=np.int64)).astype(str))
    parts, n_rows = [], 0
    for b in reader:
        n_rows += b.num_rows
        m = pc.extract_regex(b.column("ProteinExpression"), NP_RE)
        keep = pc.fill_null(pc.and_(pc.is_valid(m), pc.is_in(b.column("VariationID"), value_set=ids)), False)
        if not pc.any(keep).as_py():
            continue
        m = pc.filter(m, keep)
        x = pc.extract_regex(pc.struct_field(m, "chg"), MISSENSE_RE)
        parts.append(pa.table({
            "VariationID": pc.cast(pc.filter(b.column("VariationID"), keep), pa.int64()),
            "gene_id": pc.filter(b.column("GeneID"), keep),
            "acc": pc.struct_field(m, "acc"), "ver": pc.cast(pc.struct_field(m, "ver"), pa.int16()),
            "wt3": pc.struct_field(x, "wt3"), "pos": pc.struct_field(x, "pos"), "mt3": pc.struct_field(x, "mt3"),
        }))
    hg = pa.concat_tables(parts).to_pandas(strings_to_categorical=True)
    hg["gene_id"] = pd.to_numeric(hg["gene_id"].astype(object), errors="coerce").astype("Int64")
    hg["wt"] = hg["wt3"].astype(object).map(AA3)
    hg["mt"] = hg["mt3"].astype(object).map(AA3)
    hg["missense"] = hg["wt"].notna() & hg["mt"].notna() & (hg["wt"] != hg["mt"])
    hg["pos"] = pd.to_numeric(hg["pos"].astype(object), errors="coerce").where(hg["missense"]).astype("Int64")
    hg = hg.drop(columns=["wt3", "mt3"]).drop_duplicates().reset_index(drop=True)
    return hg, n_rows


def read_refseq():
    """RefSeq release 237 human NP_ sequences: accession -> (version, sequence)."""
    out = {}
    for f in sorted(REFSEQ_DIR.glob("human.*.protein.faa.gz")):
        acc, buf = None, []
        with gzip.open(f, "rt") as g:
            for line in g:
                if line.startswith(">"):
                    if acc is not None:
                        out[acc[0]] = (acc[1], "".join(buf))
                    head = line[1:].split(" ", 1)[0]
                    a, v = head.split(".")
                    acc, buf = ((a, int(v)) if a.startswith("NP_") else None), []
                elif acc is not None:
                    buf.append(line.strip())
            if acc is not None:
                out[acc[0]] = (acc[1], "".join(buf))
    return out


def read_hgnc():
    h = pd.read_csv(HGNC_FILE, sep="\t", dtype=str)
    h["entrez_id"] = pd.to_numeric(h["entrez_id"], errors="coerce").astype("Int64")
    h["uniprot_list"] = h["uniprot_ids"].fillna("").str.split("|").map(lambda l: [a for a in l if a])
    return h


# ----------------------------------------------------------- benchmark <-> Brandes et al.

def match_benchmark(esm, prot, hgnc):
    """Brandes et al. entry whose sequence equals each benchmark sequence exactly.

    Candidates, tried in order: the mapped UniProt accession and its isoforms,
    entries listed under the same gene symbol (both as in build_external.py), then
    the isoforms of the HGNC UniProt accessions of the protein's gene (symbol or
    ClinVar GeneID), which covers benchmark proteins without a UniProt mapping.
    """
    sym2up = {s: l for s, l in zip(hgnc["symbol"], hgnc["uniprot_list"])}
    gid2up = {g: l for g, l in zip(hgnc["entrez_id"], hgnc["uniprot_list"]) if pd.notna(g)}
    order, src = {}, {}
    for r in prot.itertuples():
        first, fallback = esm.candidates(r.uniprot, r.gene)
        ups = []
        for s in (r.gene.split(";") if isinstance(r.gene, str) else []):
            ups += sym2up.get(s.strip(), [])
        if pd.notna(r.clinvar_gene_id):
            ups += gid2up.get(int(r.clinvar_gene_id), [])
        extra = []
        for u in dict.fromkeys(ups):
            extra += [a for a in esm.candidates(u, None)[0] if a not in first and a not in fallback and a not in extra]
        order[r.protein_id] = first + fallback + extra
        src[r.protein_id] = (set(first), set(fallback))
    seq = prot.set_index("protein_id")["sequence"]
    up = prot.set_index("protein_id")["uniprot"]
    match, tried, pending, k = {}, {}, set(prot["protein_id"]), 0
    while pending:
        batch = {pid: order[pid][k] for pid in pending if k < len(order[pid])}
        esm.prefetch(batch.values())
        for pid in list(pending):
            if pid not in batch:
                pending.discard(pid)
                continue
            tried[pid] = k + 1
            if esm.sequence(batch[pid]) == seq[pid]:
                a = batch[pid]
                first, fallback = src[pid]
                kind = ("uniprot_canonical" if a == up[pid] else "uniprot_isoform" if a in first
                        else "gene_symbol_fallback" if a in fallback else "hgnc_uniprot")
                match[pid] = (a, kind)
                pending.discard(pid)
        k += 1
    # Secondary, flagged separately: for proteins without an exact match, the candidate of the same
    # length that differs only by substitutions at the fewest (<= max(2, 1% of)) positions.
    near = {}
    for pid in prot["protein_id"]:
        if pid in match:
            continue
        s = np.frombuffer(seq[pid].encode(), "S1")
        best = None
        for a in order[pid]:
            q = esm.sequence(a)
            if len(q) == len(s):
                nd = int((np.frombuffer(q.encode(), "S1") != s).sum())
                if nd <= max(2, 0.01 * len(s)) and (best is None or nd < best[1]):
                    best = (a, nd)
        if best:
            near[pid] = best
    mt = pd.DataFrame([{"protein_id": pid, "uniprot": up[pid], "gene": g, "length": len(seq[pid]),
                        "n_candidates": len(order[pid]), "n_tried": tried.get(pid, 0),
                        "brandes_entry": match.get(pid, (None, None))[0],
                        "match_type": match.get(pid, (None, "no_candidate" if not order[pid] else "no_exact_match"))[1],
                        "near_entry": near.get(pid, (None, None))[0], "near_n_diff": near.get(pid, (None, np.nan))[1]}
                       for pid, g in zip(prot["protein_id"], prot["gene"])])
    return mt


def _lookup(esm, entry, wt, pos, mtt):
    tab, s = esm.load(entry)
    llr, ok = np.full(len(wt), np.nan), np.zeros(len(wt), dtype=bool)
    for i, (w, p, x) in enumerate(zip(wt, pos, mtt)):
        ok[i] = p <= len(s) and s[p - 1] == w
        if ok[i]:
            llr[i] = float(tab.at[x, f"{w} {p}"])
    return llr, ok


def benchmark_scores(esm, bench, mt):
    """Brandes et al. oriented ESM-1b score of every benchmark variant (no labels).

    esm1b_brandes_oriented uses only the entry whose sequence equals the benchmark
    sequence. esm1b_brandes_oriented_near is filled only for proteins without such an
    entry, from a same-length entry that differs by a few substitutions (variants at
    a differing position fail the wild-type check and stay empty).
    """
    m = mt.set_index("protein_id")
    rows = []
    for pid, g in bench.groupby("protein_id", sort=True):
        entry, kind, near = m.at[pid, "brandes_entry"], m.at[pid, "match_type"], m.at[pid, "near_entry"]
        wt, mtt, pos = g["variant"].str[0].to_numpy(), g["variant"].str[-1].to_numpy(), g["position"].to_numpy()
        llr, wt_ok = np.full(len(g), np.nan), np.zeros(len(g), dtype=bool)
        llr_near = np.full(len(g), np.nan)
        if isinstance(entry, str):
            llr, wt_ok = _lookup(esm, entry, wt, pos, mtt)
            reason = np.where(wt_ok, "", "wt_mismatch")
        else:
            reason = np.full(len(g), kind, dtype=object)
            if isinstance(near, str):
                llr_near, _ = _lookup(esm, near, wt, pos, mtt)
        rows.append(pd.DataFrame({"protein_id": pid, "variant": g["variant"].to_numpy(), "position": pos,
                                  "brandes_entry": entry, "match_type": kind, "wt_ok": wt_ok,
                                  "esm1b_brandes_llr": llr, "esm1b_brandes_oriented": -llr,
                                  "scored": ~np.isnan(llr), "miss_reason": reason,
                                  "near_entry": near if isinstance(near, str) else None, "near_n_diff": m.at[pid, "near_n_diff"],
                                  "esm1b_brandes_oriented_near": -llr_near,
                                  "ESM1b_benchmark": g["ESM1b"].to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def score_agreement(a):
    """Agreement of two score columns (label-free consistency check)."""
    d = (a["ESM1b_benchmark"] - a["esm1b_brandes_oriented"]).abs()
    return {"n_proteins": int(a["protein_id"].nunique()), "n_variants": int(len(a)),
            "pearson_r": float(pearsonr(a["ESM1b_benchmark"], a["esm1b_brandes_oriented"])[0]),
            "spearman_rho": float(spearmanr(a["ESM1b_benchmark"], a["esm1b_brandes_oriented"])[0]),
            "max_abs_diff": float(d.max()), "median_abs_diff": float(d.median()),
            "n_abs_diff_gt_1e-3": int((d > 1e-3).sum()), "n_abs_diff_gt_0.1": int((d > 0.1).sum())}


# -------------------------------------------------------------- placement on new proteins

def place_changes(hg, npmap, seqs):
    """One amino-acid change per (VariationID, protein), mirroring build_external.resolve_changes.

    npmap: acc, protein_id, ref_ver (RefSeq accessions whose release-237 sequence
    equals the protein's sequence). If ClinVar gives an expression on the release-237
    version of one of these accessions, only such expressions are used for the pair;
    otherwise expressions on other versions are used. The change must be missense, its
    wild type must equal the protein sequence at that position, and all retained
    expressions must agree on one change.
    """
    hg = hg.merge(npmap, on="acc")
    key = ["VariationID", "protein_id"]
    hg["exact"] = hg["ver"] == hg["ref_ver"]
    has_exact = hg.groupby(key)["exact"].transform("any")
    hg = hg[hg["exact"] | ~has_exact].copy()
    status = hg[key].drop_duplicates().assign(status="not_missense")
    mis = hg[hg["missense"]].copy()
    mis["pos"] = mis["pos"].astype(int)
    s = mis["protein_id"].map(seqs).to_numpy()
    mis["wt_ok"] = [p <= len(q) and q[p - 1] == w for q, p, w in zip(s, mis["pos"], mis["wt"])]
    pairs_mis = mis[key].drop_duplicates()
    ok = mis[mis["wt_ok"]]
    pairs_wt = ok[key].drop_duplicates()
    nchg = ok.drop_duplicates(key + ["pos", "mt"]).groupby(key).size().rename("n_changes").reset_index()
    good = nchg.loc[nchg["n_changes"] == 1, key]
    res = ok.merge(good, on=key).sort_values(key + ["exact", "ver"], ascending=[True, True, False, False])
    res = res.drop_duplicates(key).copy()
    res["mutant"] = res["wt"] + res["pos"].astype(str) + res["mt"]
    res["np_accession"] = res["acc"].astype(str) + "." + res["ver"].astype(str)
    status = status.set_index(key)
    status.loc[pd.MultiIndex.from_frame(pairs_mis), "status"] = "wt_mismatch"
    status.loc[pd.MultiIndex.from_frame(pairs_wt), "status"] = "ambiguous"
    status.loc[pd.MultiIndex.from_frame(good), "status"] = "ok"
    cols = ["VariationID", "protein_id", "mutant", "pos", "wt", "mt", "np_accession", "exact"]
    return res[cols].rename(columns={"pos": "position", "exact": "np_version_match"}), status.reset_index()


def cluster_level_hits(hits, hom, min_id=0.3, min_cov=0.8):
    """Hits that meet the family clustering thresholds (identity >= 0.3, both coverages >= 0.8,
    at the significance E-value): such a new protein could have joined a benchmark family had
    the two sets been clustered together."""
    return hits[(hits["evalue"] <= hom["evalue"]) & (hits["fident"] >= min_id) & (np.minimum(hits["qcov"], hits["tcov"]) >= min_cov)]


def homolog_flags(fam, hits, hom):
    """Per family: does any member hit any benchmark protein significantly?"""
    f = fam.set_index("protein_id")["family"]
    sig = hits[hits["evalue"] <= hom["evalue"]]
    hc = sig[(sig["fident"] >= hom["min_id"]) & (np.minimum(sig["qcov"], sig["tcov"]) >= hom["min_cov"])]
    out = pd.DataFrame(index=pd.Index(sorted(f.unique()), name="family"))
    out["has_benchmark_homolog"] = out.index.isin(set(hc["query"].map(f)))
    out["has_benchmark_hit_e1e-3"] = out.index.isin(set(sig["query"].map(f)))
    out["has_benchmark_hit_cluster_thresholds"] = out.index.isin(set(cluster_level_hits(hits, hom)["query"].map(f)))
    for col, h in [("max_id_to_benchmark", sig), ("max_id_to_benchmark_hc", hc)]:
        best = f.index.to_series().map(h.groupby("query")["fident"].max()).fillna(0.0)
        out[col] = best.groupby(f).max()
    return out


# ---------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    fc, hom = cfg["families"], cfg["families"]["homology"]
    t_start = time.time()
    DOC.mkdir(parents=True, exist_ok=True)
    log, gene_log, pair_log = [], [], []

    prot = load_proteins(cfg)
    bench = load_variants(cfg)
    prot["acc"] = prot["protein_id"].str.split(".").str[0]
    hgnc = read_hgnc()

    # 1. ClinVar labels and protein HGVS (variant filters of build_external.py, no date filter).
    print("reading variant_summary", flush=True)
    vs = read_labels(log)
    print("reading hgvs4variation", flush=True)
    hg, n_hgvs_rows = read_np_expressions(CURRENT_HGVS, vs["VariationID"])
    step(log, "has a protein HGVS on a RefSeq NP accession", vids=hg["VariationID"].unique(),
         df=vs[vs["VariationID"].isin(set(hg["VariationID"]))])
    mis_ids = set(hg.loc[hg["missense"], "VariationID"])
    step(log, "missense (p.Xxx123Yyy, two different standard amino acids) on some RefSeq NP", vids=mis_ids,
         df=vs[vs["VariationID"].isin(mis_ids)])

    # 2. Benchmark genes, GeneIDs, UniProt accessions and sequences.
    on_b = hg[hg["acc"].isin(set(prot["acc"]))]
    on_b = on_b.assign(acc=on_b["acc"].astype(str))
    dom = (on_b.dropna(subset=["gene_id"]).groupby(["acc", "gene_id"], observed=True).size().rename("n").reset_index()
           .sort_values(["acc", "n"], ascending=[True, False]).drop_duplicates("acc"))
    prot["clinvar_gene_id"] = prot["acc"].map(dom.set_index("acc")["gene_id"]).astype("Int64")
    gid2sym = dict(zip(hgnc["entrez_id"], hgnc["symbol"]))
    prot["clinvar_gene"] = prot["clinvar_gene_id"].map(lambda g: gid2sym.get(g) if pd.notna(g) else None)
    print("matching benchmark proteins to Brandes et al. entries", flush=True)
    esm = BrandesLLR(EXT_RAW / "esm1b")
    bmt = match_benchmark(esm, prot, hgnc)
    bmt["clinvar_gene_id"] = bmt["protein_id"].map(prot.set_index("protein_id")["clinvar_gene_id"])
    bmt.to_csv(DOC / "benchmark_esm1b_matches.tsv", sep="\t", index=False)

    # Direct links: symbols (proteins.parquet and the ClinVar gene of each benchmark RefSeq protein),
    # GeneIDs (ClinVar gene of each benchmark RefSeq protein), UniProt accessions (UniProt map and
    # the Brandes et al. entry matching the benchmark sequence). Closure through HGNC: every GeneID
    # and UniProt accession HGNC lists for a gene with such a symbol or GeneID.
    bench_sym = {s.strip() for g in prot["gene"].dropna() for s in g.split(";")} | set(prot["clinvar_gene"].dropna())
    direct_gid = set(prot["clinvar_gene_id"].dropna().astype(int))
    direct_up = set(prot["uniprot"].dropna()) | set(bmt["brandes_entry"].dropna().str.split("-").str[0])
    bench_gid = direct_gid | set(hgnc.loc[hgnc["symbol"].isin(bench_sym), "entrez_id"].dropna().astype(int))
    bench_up = direct_up | {a for l in hgnc.loc[hgnc["symbol"].isin(bench_sym) | hgnc["entrez_id"].isin(bench_gid), "uniprot_list"] for a in l}
    bench_seqs = set(prot["sequence"])
    # ClinVar gene of each RefSeq protein: the GeneID of most of its alleles.
    npg = (hg.dropna(subset=["gene_id"]).groupby(["acc", "gene_id"], observed=True).size().rename("n").reset_index()
           .sort_values(["acc", "n"], ascending=[True, False]).drop_duplicates("acc"))
    np_gene = dict(zip(npg["acc"].astype(str), npg["gene_id"].astype(int)))

    # 3. Benchmark variants scored with Brandes et al. (same score function on both sides).
    print("scoring benchmark variants with Brandes et al. entries", flush=True)
    bs = benchmark_scores(esm, bench, bmt)
    bs["length"] = bs["protein_id"].map(prot.set_index("protein_id")["length"]).astype(int)
    bs["over_1022"] = bs["length"] > 1022
    sc = bs[bs["scored"]]
    bench_check = {
        "n_benchmark_proteins": int(len(bmt)), "n_benchmark_variants": int(len(bs)),
        "match_types": bmt["match_type"].value_counts().to_dict(),
        "n_proteins_scored": int(sc["protein_id"].nunique()), "n_variants_scored": int(len(sc)),
        "n_variants_not_scored": int((~bs["scored"]).sum()),
        "variants_not_scored_by_reason": bs.loc[~bs["scored"], "miss_reason"].value_counts().to_dict(),
        "proteins_not_scored_by_reason": bmt.loc[bmt["brandes_entry"].isna(), "match_type"].value_counts().to_dict(),
        "n_proteins_not_scored_without_uniprot_map": int(bmt.loc[bmt["brandes_entry"].isna(), "uniprot"].isna().sum()),
        "n_proteins_not_scored_over_1022": int((bmt.loc[bmt["brandes_entry"].isna(), "length"] > 1022).sum()),
        "secondary_near_match": {
            "n_proteins": int(bmt["near_entry"].notna().sum()),
            "near_n_diff": {str(int(k)): int(v) for k, v in bmt["near_n_diff"].dropna().value_counts().sort_index().items()},
            "n_variants_scored_near": int(bs["esm1b_brandes_oriented_near"].notna().sum()),
            "n_variants_still_unscored": int((~bs["scored"] & bs["esm1b_brandes_oriented_near"].isna()).sum())},
        "agreement_with_benchmark_ESM1b": {"all": score_agreement(sc), "length_le_1022": score_agreement(sc[~sc["over_1022"]]),
                                           "length_gt_1022": score_agreement(sc[sc["over_1022"]])},
    }
    nr = bs[bs["esm1b_brandes_oriented_near"].notna()]
    if len(nr) > 2:
        bench_check["secondary_near_match"]["agreement_with_benchmark_ESM1b"] = score_agreement(
            nr.drop(columns="esm1b_brandes_oriented").rename(columns={"esm1b_brandes_oriented_near": "esm1b_brandes_oriented"}))
    bs.to_parquet(processed_dir(cfg) / "unseen_benchmark_esm1b.parquet", index=False)
    print(json.dumps(bench_check, indent=2), flush=True)

    # 4. Candidate genes: approved HGNC protein-coding genes with labelled missense alleles, not in the benchmark.
    can_entries = {a for a in esm.entries if "-" not in a}
    g = hgnc[hgnc["locus_group"] == "protein-coding gene"].copy()
    gene_log.append({"step": "HGNC approved protein-coding genes", "n_genes": int(len(g))})
    allele_genes = set(hg.loc[hg["missense"], "gene_id"].dropna().astype(int))
    n_allele_gene_not_pc = len(allele_genes - set(g["entrez_id"].dropna().astype(int)))
    g = g[g["entrez_id"].isin(allele_genes)]
    gene_log.append({"step": "has a ClinVar labelled missense allele (ClinVar GeneID of the allele)", "n_genes": int(len(g)),
                     "note": f"{n_allele_gene_not_pc} ClinVar GeneIDs with such alleles are not HGNC protein-coding genes"})
    g["ex_symbol"] = g["symbol"].isin(bench_sym)
    g["ex_gene_id"] = g["entrez_id"].isin(direct_gid)
    g["ex_uniprot"] = g["uniprot_list"].map(lambda l: bool(set(l) & direct_up))
    g["ex_hgnc_closure"] = g["entrez_id"].isin(bench_gid) | g["uniprot_list"].map(lambda l: bool(set(l) & bench_up))
    ex = g["ex_symbol"] | g["ex_gene_id"] | g["ex_uniprot"] | g["ex_hgnc_closure"]
    direct = g["ex_symbol"] | g["ex_gene_id"] | g["ex_uniprot"]
    gene_log.append({"step": "not a benchmark gene (symbol, GeneID or UniProt accession)", "n_genes": int((~ex).sum()),
                     "excluded": {"total": int(ex.sum()), "by_symbol": int(g["ex_symbol"].sum()),
                                  "by_gene_id": int(g["ex_gene_id"].sum()), "by_uniprot": int(g["ex_uniprot"].sum()),
                                  "by_symbol_only": int((g["ex_symbol"] & ~g["ex_gene_id"] & ~g["ex_uniprot"]).sum()),
                                  "by_gene_id_only": int((g["ex_gene_id"] & ~g["ex_symbol"] & ~g["ex_uniprot"]).sum()),
                                  "by_uniprot_only": int((g["ex_uniprot"] & ~g["ex_symbol"] & ~g["ex_gene_id"]).sum()),
                                  "by_hgnc_closure_only": int((g["ex_hgnc_closure"] & ~direct).sum()),
                                  "genes_by_hgnc_closure_only": sorted(g.loc[g["ex_hgnc_closure"] & ~direct, "symbol"])}})
    g = g[~ex].copy()
    g["candidates"] = g["uniprot_list"].map(lambda l: [a for a in l if a in can_entries])
    g["status"] = np.where(g["uniprot_list"].map(len) == 0, "no_uniprot_in_hgnc",
                           np.where(g["candidates"].map(len) == 0, "no_brandes_canonical_entry", ""))
    gene_log.append({"step": "has a UniProt accession scored by Brandes et al. as a canonical entry",
                     "n_genes": int((g["status"] == "").sum()),
                     "excluded": g.loc[g["status"] != "", "status"].value_counts().to_dict()})

    # 5. Sequences of the candidate entries (as scored by Brandes et al.).
    entries = sorted({a for l in g.loc[g["status"] == "", "candidates"] for a in l})
    print(f"fetching {len(entries)} Brandes et al. entries", flush=True)
    for i in range(0, len(entries), 1000):
        esm.prefetch(entries[i:i + 1000])
    eseq = {a: esm.sequence(a) for a in entries}
    same_as_bench = {a for a, s in eseq.items() if s in bench_seqs}
    nonstd = {a for a, s in eseq.items() if set(s) - set(AA3.values())}
    # Identical sequences under different accessions are one protein, named by the smallest accession.
    first_acc = {}
    for a in sorted(eseq):
        first_acc.setdefault(eseq[a], a)
    dup_seq = {a for a in eseq if first_acc[eseq[a]] != a}
    g.loc[g["status"] == "", "candidates"] = g.loc[g["status"] == "", "candidates"].map(
        lambda l: list(dict.fromkeys(first_acc[eseq[a]] for a in l)))

    # 6. RefSeq proteins identical to each entry, and placement of ClinVar changes.
    print("reading RefSeq proteins", flush=True)
    refseq = read_refseq()
    by_seq = {}
    for acc, (ver, s) in refseq.items():
        by_seq.setdefault(s, []).append((acc, ver))
    usable = [a for a in entries if a not in same_as_bench and a not in dup_seq]
    rows = [{"acc": acc, "protein_id": a, "ref_ver": ver, "np_match": "identical", "n_diff": 0}
            for a in usable for acc, ver in by_seq.get(eseq[a], [])]
    # Entries without an identical RefSeq protein: a RefSeq protein of the same gene (its ClinVar
    # GeneID) with the same length that differs only by substitutions at <= max(2, 1% of) positions
    # has the same numbering; changes at the differing positions fail the wild-type check.
    has_identical = {r["protein_id"] for r in rows}
    entry_gids = {}
    for r in g.loc[g["status"] == ""].itertuples():
        for a in r.candidates:
            entry_gids.setdefault(a, set()).add(int(r.entrez_id))
    by_len = {}
    for acc, (ver, s) in refseq.items():
        if acc in np_gene:
            by_len.setdefault((np_gene[acc], len(s)), []).append((acc, ver, s))
    for a in usable:
        if a in has_identical:
            continue
        e = np.frombuffer(eseq[a].encode(), "S1")
        for gid in sorted(entry_gids.get(a, ())):
            for acc, ver, s in by_len.get((gid, len(eseq[a])), []):
                nd = int((np.frombuffer(s.encode(), "S1") != e).sum())
                if nd <= max(2, 0.01 * len(e)):
                    rows.append({"acc": acc, "protein_id": a, "ref_ver": ver, "np_match": "same_length_substitutions", "n_diff": nd})
    npmap = pd.DataFrame(rows).drop_duplicates(["acc", "protein_id"])
    # An identical match wins; a RefSeq protein close to two entries goes to the closer one.
    ident_acc = set(npmap.loc[npmap["np_match"] == "identical", "acc"])
    npmap = npmap[(npmap["np_match"] == "identical") | ~npmap["acc"].isin(ident_acc)]
    npmap = npmap.sort_values(["n_diff", "protein_id"]).drop_duplicates("acc").sort_values(["protein_id", "acc"]).reset_index(drop=True)
    npmap["ref_ver"] = npmap["ref_ver"].astype(np.int16)
    n_np = npmap.groupby("protein_id").size()
    hg_c = hg[hg["acc"].isin(set(npmap["acc"]))].copy()
    hg_c["acc"] = hg_c["acc"].astype(str)
    res_all, status_all = place_changes(hg_c, npmap, eseq)
    n_ok = res_all.groupby("protein_id")["VariationID"].nunique()

    # 7. One protein per gene.
    def entry_status(a):
        if a in same_as_bench:
            return "sequence_equals_a_benchmark_sequence"
        if a not in n_np.index:
            return "no_refseq_protein_with_the_same_numbering"
        if a not in n_ok.index:
            return "no_clinvar_change_placed"
        return "ok"
    est = {a: entry_status(a) for a in usable + sorted(same_as_bench - dup_seq)}
    chosen, gm = {}, []
    for r in g.itertuples():
        if r.status:
            gm.append({"gene": r.symbol, "hgnc_id": r.hgnc_id, "entrez_id": r.entrez_id, "uniprot_ids": r.uniprot_ids,
                       "candidates": "", "candidate_status": "", "chosen": None, "status": r.status})
            continue
        ok = [a for a in r.candidates if est[a] == "ok"]
        # Most placed VariationIDs, then longer sequence, then smaller accession.
        best = sorted(ok, key=lambda a: (-n_ok[a], -len(eseq[a]), a))[0] if ok else None
        if best is not None:
            chosen.setdefault(best, []).append(r)
        status = "ok" if best else ("; ".join(sorted({est[a] for a in r.candidates})))
        gm.append({"gene": r.symbol, "hgnc_id": r.hgnc_id, "entrez_id": r.entrez_id, "uniprot_ids": r.uniprot_ids,
                   "candidates": ";".join(r.candidates), "candidate_status": ";".join(est[a] for a in r.candidates),
                   "chosen": best, "status": status})
    gm = pd.DataFrame(gm)
    gm.to_csv(DOC / "gene_mapping.tsv", sep="\t", index=False)
    cs = gm.loc[gm["candidates"] != "", "candidate_status"].str.split(";")
    gene_log.append({"step": "some candidate entry differs in sequence from every benchmark protein",
                     "n_genes": int(cs.map(lambda l: any(x != "sequence_equals_a_benchmark_sequence" for x in l)).sum())})
    tier = npmap.groupby("protein_id")["np_match"].first()
    gene_log.append({"step": "a RefSeq protein (release 237) has exactly the sequence of such an entry, or the same length "
                             "and only substitutions at <= max(2, 1% of) positions",
                     "n_genes": int(cs.map(lambda l: any(x in ("ok", "no_clinvar_change_placed") for x in l)).sum()),
                     "n_entries_identical": int((tier == "identical").sum()),
                     "n_entries_same_length_substitutions": int((tier == "same_length_substitutions").sum())})
    gene_log.append({"step": "at least one ClinVar change placed on such an entry (one entry chosen per gene)",
                     "n_genes": int((gm["status"] == "ok").sum()),
                     "status_of_dropped_genes": gm.loc[gm["status"] != "ok", "status"].value_counts().to_dict(),
                     "n_genes_with_2plus_usable_entries": int(cs.map(lambda l: l.count("ok") >= 2).sum()),
                     "n_entries_chosen_by_2plus_genes": int(sum(len(v) > 1 for v in chosen.values())),
                     "n_proteins": len(chosen)})

    keep = set(chosen)
    res = res_all[res_all["protein_id"].isin(keep)].copy()
    st = status_all[status_all["protein_id"].isin(keep)]
    pstat = st["status"].value_counts().to_dict()
    lab = vs.set_index("VariationID")
    res = res.join(lab[["label", "ClinicalSignificance", "ReviewStatus", "LastEvaluated", "GeneSymbol"]], on="VariationID")
    pair_log.append({"step": "(VariationID, protein) pairs with a protein HGVS on a RefSeq protein with the numbering of the entry",
                     "n_pairs": int(len(st)), "n_variation_ids": int(st["VariationID"].nunique()), "n_proteins": int(st["protein_id"].nunique()),
                     "pair_status": pstat})
    step(log, "missense on the chosen entry, wild type matches the entry, change unambiguous", vids=res["VariationID"].unique(), df=res,
         note=f"dropped pairs: not_missense={pstat.get('not_missense', 0)}, wt_mismatch={pstat.get('wt_mismatch', 0)}, ambiguous={pstat.get('ambiguous', 0)}")

    # Labelled missense alleles of the final genes that could not be placed on the chosen entry.
    fin_gid = {int(r.entrez_id) for v in chosen.values() for r in v}
    own = hg[hg["missense"] & hg["gene_id"].isin(fin_gid)]
    own_ids = set(own["VariationID"])
    placed_ids = set(res["VariationID"])
    tried_ids = set(st["VariationID"])
    not_placed = {
        "n_labelled_missense_variation_ids_of_final_genes": len(own_ids),
        "placed": len(own_ids & placed_ids),
        "no_expression_on_a_refseq_protein_with_the_entry_numbering": len(own_ids - tried_ids),
        "expression_on_such_a_refseq_protein_but_dropped": len((own_ids & tried_ids) - placed_ids),
        "placed_variation_ids_whose_allele_gene_is_another_gene": len(placed_ids - own_ids),
    }

    n_p1 = int((res["position"] == 1).sum())
    res = res[res["position"] > 1]
    step(log, "not at position 1 (start-loss)", vids=res["VariationID"].unique(), df=res, note=f"{n_p1} rows removed")

    res["stars"] = res["ReviewStatus"].map(STARS)
    nlab = res.groupby(["protein_id", "mutant"])["label"].transform("nunique")
    n_conflict = int(res.loc[nlab > 1, ["protein_id", "mutant"]].drop_duplicates().shape[0])
    n_conflict_rows = int((nlab > 1).sum())
    res = res[nlab == 1].copy()
    grp = res.sort_values("VariationID").groupby(["protein_id", "mutant"])
    ids = grp["VariationID"].agg(lambda s: ";".join(map(str, s))).rename("variation_ids")
    nids = grp["VariationID"].nunique().rename("n_variation_ids")
    rep = res.sort_values(["stars", "VariationID"], ascending=[False, True]).drop_duplicates(["protein_id", "mutant"])
    rep = rep.merge(ids, on=["protein_id", "mutant"]).merge(nids, on=["protein_id", "mutant"])
    step(log, "one row per (protein, change); label conflicts dropped", df=rep,
         note=f"{n_conflict} (protein, change) with conflicting labels across VariationIDs dropped ({n_conflict_rows} rows); "
              f"{int((rep['n_variation_ids'] > 1).sum())} rows merge >1 VariationID")

    # 8. ESM-1b scores of the new variants, with a wild-type check against the entry.
    print("reading ESM-1b scores of the new variants", flush=True)
    vals, wt_bad = {}, 0
    for pid, gg in rep.groupby("protein_id"):
        tab, s = esm.load(pid)
        assert s == eseq[pid]
        for i, w, p, m in zip(gg.index, gg["wt"], gg["position"], gg["mt"]):
            if p <= len(s) and s[p - 1] == w:
                vals[i] = float(tab.at[m, f"{w} {p}"])
            else:
                wt_bad += 1
    rep["esm1b_llr"] = pd.Series(vals)
    rep = rep[rep["esm1b_llr"].notna()].copy()
    rep["esm1b_oriented"] = -rep["esm1b_llr"]
    step(log, "ESM-1b LLR present and wild type of the entry matches", df=rep, note=f"{wt_bad} rows dropped")

    # 9. Protein table.
    pg = {a: v for a, v in chosen.items() if a in set(rep["protein_id"])}
    P = pd.DataFrame({"protein_id": sorted(pg)})
    P["uniprot"] = P["protein_id"]
    P["gene"] = P["protein_id"].map(lambda a: ";".join(sorted(r.symbol for r in pg[a])))
    P["hgnc_id"] = P["protein_id"].map(lambda a: ";".join(sorted(r.hgnc_id for r in pg[a])))
    P["entrez_id"] = P["protein_id"].map(lambda a: ";".join(sorted(str(r.entrez_id) for r in pg[a])))
    P["sequence"] = P["protein_id"].map(eseq)
    P["length"] = P["sequence"].str.len()
    P["over_1022"] = P["length"] > 1022
    # Longer than every benchmark protein (the benchmark has no protein above this length).
    bench_max_len = int(prot["length"].max())
    P["over_benchmark_max_length"] = P["length"] > bench_max_len
    P["refseq_match"] = P["protein_id"].map(tier)
    P["refseq_proteins"] = P["protein_id"].map(npmap.groupby("protein_id")["acc"].agg(lambda s: ";".join(sorted(s))))
    P["refseq_n_diff"] = P["protein_id"].map(npmap.groupby("protein_id")["n_diff"].min()).astype(int)
    cnt = rep.groupby("protein_id")["label"].agg(n_variants="size", n_path="sum")
    P = P.merge(cnt, left_on="protein_id", right_index=True)
    P["n_benign"] = P["n_variants"] - P["n_path"]
    P["brandes_entry"] = P["protein_id"]
    assert not set(P["sequence"]) & bench_seqs and not set(P["uniprot"]) & bench_up

    # 10. Families (new proteins only) and homology to the benchmark.
    print(f"clustering {len(P)} new proteins", flush=True)
    fam = cluster(P, fc["min_seq_id"], fc["coverage"], fc["cov_mode"], threads=args.threads).rename(columns={"family_id": "family"})
    assert len(fam) == len(P) and set(fam["protein_id"]) == set(P["protein_id"])
    P = P.merge(fam, on="protein_id")
    print("searching new proteins against the benchmark", flush=True)
    hits = search(P[["protein_id", "sequence"]], prot[["protein_id", "sequence"]], fc["search_sensitivity"], fc["search_evalue"], threads=args.threads)
    assert set(hits["query"]) <= set(P["protein_id"]) and set(hits["target"]) <= set(prot["protein_id"])
    hits = hits.sort_values(["query", "evalue", "target"]).reset_index(drop=True)
    hits.to_parquet(processed_dir(cfg) / "unseen_homology.parquet", index=False)

    F = P.groupby("family").agg(members=("protein_id", lambda s: ";".join(sorted(s))), n_proteins=("protein_id", "size"),
                                n_variants=("n_variants", "sum"), n_path=("n_path", "sum"), n_benign=("n_benign", "sum"),
                                max_length=("length", "max"))
    F = F.join(homolog_flags(P[["protein_id", "family"]], hits, hom)).reset_index()
    F.to_parquet(processed_dir(cfg) / "unseen_families.parquet", index=False)
    sig = hits[hits["evalue"] <= hom["evalue"]]
    hc = sig[(sig["fident"] >= hom["min_id"]) & (np.minimum(sig["qcov"], sig["tcov"]) >= hom["min_cov"])]
    P["max_id_to_benchmark"] = P["protein_id"].map(sig.groupby("query")["fident"].max()).fillna(0.0)
    P["has_benchmark_homolog"] = P["protein_id"].isin(set(hc["query"]))
    P["has_benchmark_hit_e1e-3"] = P["protein_id"].isin(set(sig["query"]))
    P = P[["protein_id", "uniprot", "gene", "hgnc_id", "entrez_id", "length", "over_1022", "over_benchmark_max_length", "sequence",
           "n_variants", "n_path", "n_benign",
           "family", "max_id_to_benchmark", "has_benchmark_homolog", "has_benchmark_hit_e1e-3", "refseq_match", "refseq_proteins",
           "refseq_n_diff", "brandes_entry"]].sort_values("protein_id").reset_index(drop=True)
    P.to_parquet(processed_dir(cfg) / "unseen_proteins.parquet", index=False)

    out = rep.rename(columns={"VariationID": "variation_id", "ReviewStatus": "review_status", "ClinicalSignificance": "clinsig",
                              "LastEvaluated": "last_evaluated", "GeneSymbol": "clinvar_gene_symbol"})
    out["review_stars"] = out["stars"].astype(int)
    out["gene"] = out["protein_id"].map(P.set_index("protein_id")["gene"])
    out["uniprot"] = out["protein_id"]
    out["family"] = out["protein_id"].map(P.set_index("protein_id")["family"])
    out["length"] = out["protein_id"].map(P.set_index("protein_id")["length"])
    out["over_1022"] = out["length"] > 1022
    out["over_benchmark_max_length"] = out["length"] > bench_max_len
    out["np_match"] = out["np_accession"].str.split(".").str[0].map(npmap.set_index("acc")["np_match"])
    cols = ["protein_id", "uniprot", "gene", "family", "mutant", "position", "wt", "mt", "label", "esm1b_oriented", "esm1b_llr",
            "variation_id", "variation_ids", "n_variation_ids", "review_stars", "review_status", "clinsig", "last_evaluated",
            "clinvar_gene_symbol", "np_accession", "np_version_match", "np_match", "length", "over_1022", "over_benchmark_max_length"]
    out = out[cols].sort_values(["protein_id", "position", "mutant"]).reset_index(drop=True)
    out["position"] = out["position"].astype(int)
    out["label"] = out["label"].astype(int)
    assert out["esm1b_oriented"].notna().all() and not out.duplicated(["protein_id", "mutant"]).any()
    out.to_parquet(processed_dir(cfg) / "unseen_variants.parquet", index=False)

    # 11. Summary.
    gv = out.groupby("protein_id")["label"].agg(["min", "max", "size"])
    fv = F.set_index("family")
    shared = out[out["variation_id"].duplicated(keep=False)]
    refseq_files = sorted(REFSEQ_DIR.glob("human.*.protein.faa.gz"))
    bdir = processed_dir(cfg)
    summary = {
        "inputs": {
            "current_variant_summary": {"file": "pub/clinvar/tab_delimited/variant_summary.txt.gz",
                                        "last_modified": (CURRENT_VS.parent / (CURRENT_VS.name + ".last_modified")).read_text().strip(),
                                        "sha256": sha256(CURRENT_VS)},
            "current_hgvs4variation": {"file": "pub/clinvar/tab_delimited/hgvs4variation.txt.gz",
                                       "last_modified": (CURRENT_HGVS.parent / (CURRENT_HGVS.name + ".last_modified")).read_text().strip(),
                                       "sha256": sha256(CURRENT_HGVS), "n_rows": n_hgvs_rows,
                                       "n_np_rows_for_labelled_variation_ids": int(len(hg))},
            "hgnc": {"file": "hgnc/archive/archive/monthly/tsv/" + HGNC_FILE.name, "sha256": sha256(HGNC_FILE), "n_rows": int(len(hgnc))},
            "refseq_protein": {"dir": "refseq/H_sapiens/mRNA_Prot", "release": (REFSEQ_DIR / "RELEASE_NUMBER").read_text().strip(),
                               "last_modified": (REFSEQ_DIR / (refseq_files[0].name + ".last_modified")).read_text().strip(),
                               "n_np_sequences": len(refseq),
                               "sha256": {f.name: sha256(f) for f in refseq_files}},
            "esm1b": {"space": HF_SPACE, "revision": HF_REVISION, "file": ZIP_NAME, "size": ZIP_SIZE, "lfs_sha256": ZIP_LFS_SHA256,
                      "n_entries": len(esm.entries), "n_canonical_entries": len(can_entries),
                      "central_directory_sha256": sha256(EXT_RAW / "esm1b" / "central_directory.bin"),
                      "isoform_list_sha256": sha256(EXT_RAW / "esm1b" / "isoform_list.csv")},
            "benchmark": {f: sha256(bdir / f) for f in ["proteins.parquet", "variants.parquet", "families.parquet", "homology.parquet"]},
            "config": {"families": fc},
        },
        "variant_filter_steps": log,
        "gene_selection_steps": gene_log,
        "pair_steps": pair_log,
        "labelled_missense_alleles_of_final_genes": not_placed,
        "benchmark_exclusion_sets": {"n_symbols": len(bench_sym), "n_gene_ids": len(bench_gid), "n_uniprot_accessions": len(bench_up),
                                     "n_sequences": len(bench_seqs)},
        "candidate_entries": {"n": len(entries), "sequence_equals_a_benchmark_sequence": sorted(same_as_bench),
                              "same_sequence_as_another_candidate": sorted(dup_seq),
                              "with_nonstandard_residues": sorted(nonstd),
                              "n_with_refseq_protein_with_same_numbering": int(len(n_np))},
        "final": {
            "n_variants": int(len(out)), "n_proteins": int(len(P)), "n_genes": int(sum(len(v) for v in pg.values())),
            "n_pathogenic": int(out["label"].sum()), "n_benign": int((1 - out["label"]).sum()),
            "fraction_pathogenic": float(out["label"].mean()),
            "n_proteins_with_both_classes": int(((gv["min"] == 0) & (gv["max"] == 1)).sum()),
            "n_proteins_only_pathogenic": int((gv["min"] == 1).sum()), "n_proteins_only_benign": int((gv["max"] == 0).sum()),
            "variants_per_protein_median": float(gv["size"].median()), "variants_per_protein_max": int(gv["size"].max()),
            "n_proteins_ge10_variants": int((gv["size"] >= 10).sum()),
            "n_proteins_by_refseq_match": P["refseq_match"].value_counts().to_dict(),
            "n_variants_by_refseq_match": out["np_match"].value_counts().to_dict(),
            "n_proteins_over_1022": int(P["over_1022"].sum()), "n_variants_in_proteins_over_1022": int(out["over_1022"].sum()),
            "labels_in_proteins_over_1022": {"pathogenic": int(out.loc[out["over_1022"], "label"].sum()),
                                             "benign": int((1 - out.loc[out["over_1022"], "label"]).sum())},
            "benchmark_max_length": bench_max_len,
            "n_proteins_over_benchmark_max_length": int(P["over_benchmark_max_length"].sum()),
            "n_variants_in_proteins_over_benchmark_max_length": int(out["over_benchmark_max_length"].sum()),
            "labels_in_proteins_over_benchmark_max_length": {
                "pathogenic": int(out.loc[out["over_benchmark_max_length"], "label"].sum()),
                "benign": int((1 - out.loc[out["over_benchmark_max_length"], "label"]).sum())},
            "proteins_with_most_pathogenic_variants": {
                g: int(n) for g, n in out.groupby("gene")["label"].sum().sort_values(ascending=False).head(10).items()},
            "n_families": int(len(F)), "n_singleton_families": int((F["n_proteins"] == 1).sum()),
            "max_family_size": int(F["n_proteins"].max()),
            "n_families_with_both_classes": int(((F["n_path"] > 0) & (F["n_benign"] > 0)).sum()),
            "n_families_ge10_variants": int((F["n_variants"] >= 10).sum()),
            "variants_per_family_median": float(F["n_variants"].median()), "variants_per_family_max": int(F["n_variants"].max()),
            "n_families_with_benchmark_homolog_hc": int(fv["has_benchmark_homolog"].sum()),
            "n_families_without_benchmark_homolog_hc": int((~fv["has_benchmark_homolog"]).sum()),
            "n_families_with_benchmark_hit_cluster_thresholds": int(fv["has_benchmark_hit_cluster_thresholds"].sum()),
            "n_proteins_with_benchmark_hit_cluster_thresholds": int(cluster_level_hits(hits, hom)["query"].nunique()),
            "n_families_with_benchmark_hit_e1e-3": int(fv["has_benchmark_hit_e1e-3"].sum()),
            "n_families_without_benchmark_hit_e1e-3": int((~fv["has_benchmark_hit_e1e-3"]).sum()),
            "n_variants_in_families_without_benchmark_homolog_hc": int(fv.loc[~fv["has_benchmark_homolog"], "n_variants"].sum()),
            "n_variants_in_families_without_benchmark_hit_e1e-3": int(fv.loc[~fv["has_benchmark_hit_e1e-3"], "n_variants"].sum()),
            "n_homology_hits": int(len(hits)), "n_proteins_with_any_hit": int(hits["query"].nunique()),
            "n_wt_mismatch_pairs_dropped": int(pstat.get("wt_mismatch", 0)), "n_wt_mismatch_rows_at_scoring": int(wt_bad),
            "n_rows_np_version_differs": int((~out["np_version_match"]).sum()),
            "n_rows_merging_gt1_variation_id": int((out["n_variation_ids"] > 1).sum()),
            "n_rows_variation_id_shared_across_proteins": int(len(shared)),
            "n_variation_ids_shared_across_proteins": int(shared["variation_id"].nunique()),
            "review_stars": {str(k): int(v) for k, v in out["review_stars"].value_counts().sort_index().items()},
            "clinsig": out["clinsig"].value_counts().to_dict(),
        },
        "benchmark_brandes_esm1b": bench_check,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    with open(DOC / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=int)
    print(json.dumps(summary["final"], indent=2))


if __name__ == "__main__":
    main()
