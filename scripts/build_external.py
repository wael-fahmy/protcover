"""Build the external, temporally separated ClinVar test set.

Output: data/processed/external_temporal.parquet, plus data/external/build_summary.json
(counts after every filter, input checksums, sanity checks) and
data/external/esm1b_matches.tsv (which Brandes et al. entry scores each protein).

"New" variants are ClinVar VariationIDs that do not occur anywhere in the
archived variant_summary of 2023-06-01 (any assembly, type or classification).
Their current labels come from the current variant_summary.txt.gz and their
protein changes from the current hgvs4variation.txt.gz. Only the 2,525 proteins
of the ProteinGym v1.3 clinical benchmark are used, and any (protein, amino-acid
change) already in the benchmark is removed whatever its label. Scores are the
precomputed ESM-1b LLRs of Brandes et al. (2023), oriented like the benchmark's
ESM1b column (higher = more pathogenic, i.e. -LLR).

This script only builds the table and checks the predictor (AUC of the oriented
score, agreement with the benchmark ESM1b column). It evaluates no conformal method.

Inputs: bash data/external/download.sh (ClinVar files); the ESM-1b entries are
fetched by HTTP range requests on first use and cached in data/raw/external/esm1b/.

Usage: python scripts/build_external.py [--config configs/default.yaml]
"""
import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from external_esm1b import HF_REVISION, HF_SPACE, ZIP_LFS_SHA256, ZIP_NAME, ZIP_SIZE, BrandesLLR  # noqa: E402

from protcover.config import ROOT, load_config, processed_dir  # noqa: E402
from protcover.data import load_proteins, load_variants  # noqa: E402

EXT_RAW = ROOT / "data" / "raw" / "external"
EXT_DOC = ROOT / "data" / "external"
SNAPSHOT_FILE = EXT_RAW / "clinvar_archive" / "variant_summary_2023-06.txt.gz"
NEXT_RELEASE_FILE = EXT_RAW / "clinvar_archive" / "variant_summary_2023-07.txt.gz"
CURRENT_VS = EXT_RAW / "clinvar_current" / "variant_summary.txt.gz"
CURRENT_HGVS = EXT_RAW / "clinvar_current" / "hgvs4variation.txt.gz"

PATHOGENIC = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
BENIGN = {"Benign", "Likely benign", "Benign/Likely benign"}
# ClinVar review statuses with at least one star, mapped to their star count.
STARS = {
    "criteria provided, single submitter": 1,
    "criteria provided, multiple submitters, no conflicts": 2,
    "reviewed by expert panel": 3,
    "practice guideline": 4,
}
AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
       "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}
# p.Arg41Cys or p.(Arg41Cys); anything else (Ter, =, fs, del, ?) is not a missense change.
MISSENSE_RE = r"^p\.\(?(?P<wt3>[A-Z][a-z]{2})(?P<pos>\d+)(?P<mt3>[A-Z][a-z]{2})\)?$"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path, columns, skip_rows=0):
    opts = dict(
        read_options=pacsv.ReadOptions(skip_rows=skip_rows, block_size=1 << 26),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(include_columns=columns, column_types={c: pa.string() for c in columns}),
    )
    return pacsv.read_csv(path, **opts)


def variation_ids(path):
    t = read_tsv(path, ["VariationID"])
    return set(pd.to_numeric(t.column("VariationID").to_pandas(), errors="raise").astype(np.int64))


def hgvs_on_benchmark(path, bench_bases):
    """Protein HGVS rows whose RefSeq protein accession (any version) is a benchmark protein."""
    with gzip.open(path, "rt") as g:
        skip = next(i for i, line in enumerate(g) if line.startswith("#Symbol"))
    reader = pacsv.open_csv(
        pa.input_stream(str(path), compression="gzip"),
        read_options=pacsv.ReadOptions(skip_rows=skip, block_size=1 << 26),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(include_columns=["VariationID", "ProteinExpression"],
                                             column_types={"VariationID": pa.string(), "ProteinExpression": pa.string()}),
    )
    bases = pa.array(sorted(bench_bases))
    out, n_rows = [], 0
    for batch in reader:
        n_rows += batch.num_rows
        pe = batch.column("ProteinExpression")
        m = pc.extract_regex(pe, r"^(?P<acc>NP_\d+)\.(?P<ver>\d+):(?P<chg>.+)$")
        acc = pc.struct_field(m, "acc")
        keep = pc.fill_null(pc.is_in(acc, value_set=bases), False)
        if not pc.any(keep).as_py():
            continue
        out.append(pd.DataFrame({
            "VariationID": pc.filter(batch.column("VariationID"), keep).to_pandas(),
            "acc": pc.filter(acc, keep).to_pandas(),
            "ver": pc.filter(pc.struct_field(m, "ver"), keep).to_pandas(),
            "chg": pc.filter(pc.struct_field(m, "chg"), keep).to_pandas(),
        }))
    df = pd.concat(out, ignore_index=True).drop_duplicates()
    df["VariationID"] = df["VariationID"].astype(np.int64)
    df["ver"] = df["ver"].astype(int)
    return df, n_rows


def resolve_changes(hg, prot):
    """One amino-acid change per (VariationID, benchmark protein).

    If ClinVar gives an expression on the benchmark's exact accession version,
    only those expressions are used; otherwise expressions on other versions are
    used. The change must be a missense substitution whose wild-type residue
    equals the benchmark sequence at that position, and all retained expressions
    must agree on one change. Returns the resolved table and per-pair status.
    """
    hg = hg.merge(prot[["protein_id", "acc", "bench_ver"]], on="acc")
    key = ["VariationID", "protein_id"]
    hg["exact"] = hg["ver"] == hg["bench_ver"]
    has_exact = hg.groupby(key)["exact"].transform("any")
    hg = hg[hg["exact"] | ~has_exact].copy()
    status = hg[key].drop_duplicates().assign(status="not_missense")

    x = hg["chg"].str.extract(MISSENSE_RE)
    hg["wt"], hg["mt"] = x["wt3"].map(AA3), x["mt3"].map(AA3)
    hg["pos"] = pd.to_numeric(x["pos"], errors="coerce")
    mis = hg[hg["wt"].notna() & hg["mt"].notna() & (hg["wt"] != hg["mt"])].copy()
    mis["pos"] = mis["pos"].astype(int)
    seq = prot.set_index("protein_id")["sequence"]
    s = seq.reindex(mis["protein_id"]).to_numpy()
    mis["wt_ok"] = [p <= len(q) and q[p - 1] == w for q, p, w in zip(s, mis["pos"], mis["wt"])]
    pairs_mis = mis[key].drop_duplicates()
    ok = mis[mis["wt_ok"]]
    pairs_wt = ok[key].drop_duplicates()
    nchg = ok.drop_duplicates(key + ["pos", "mt"]).groupby(key).size().rename("n_changes").reset_index()
    good = nchg.loc[nchg["n_changes"] == 1, key]
    res = ok.merge(good, on=key).sort_values(key + ["exact", "ver"], ascending=[True, True, False, False])
    res = res.drop_duplicates(key)
    res["mutant"] = res["wt"] + res["pos"].astype(str) + res["mt"]
    res["np_accession"] = res["acc"] + "." + res["ver"].astype(str)

    status = status.set_index(key)
    status.loc[pd.MultiIndex.from_frame(pairs_mis), "status"] = "wt_mismatch"
    status.loc[pd.MultiIndex.from_frame(pairs_wt), "status"] = "ambiguous"
    status.loc[pd.MultiIndex.from_frame(good), "status"] = "ok"
    cols = ["VariationID", "protein_id", "mutant", "pos", "wt", "mt", "np_accession", "exact"]
    return res[cols].rename(columns={"pos": "position", "exact": "np_version_match"}), status.reset_index()


def step(log, name, vids=None, df=None, note=""):
    rec = {"step": name}
    if vids is not None:
        rec["n_variation_ids"] = int(len(vids))
    if df is not None:
        rec["n_rows"] = int(len(df))
        if "label" in df:
            rec["n_pathogenic"] = int((df["label"] == 1).sum())
            rec["n_benign"] = int((df["label"] == 0).sum())
        if "protein_id" in df:
            rec["n_proteins"] = int(df["protein_id"].nunique())
    if note:
        rec["note"] = note
    log.append(rec)
    print(json.dumps(rec), flush=True)


def per_protein_auc(df, score):
    both = df.groupby("protein_id")["label"].transform("nunique") == 2
    a = df[both].groupby("protein_id")[[score, "label"]].apply(lambda g: roc_auc_score(g["label"], g[score]))
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    log = []

    # Benchmark proteins and variants (protcover loaders).
    prot = load_proteins(cfg)
    bench = load_variants(cfg)
    prot["acc"] = prot["protein_id"].str.split(".").str[0]
    prot["bench_ver"] = prot["protein_id"].str.split(".").str[1].astype(int)
    assert prot["acc"].is_unique
    bench_pairs = set(zip(bench["protein_id"], bench["variant"]))

    # Snapshot boundary and next release.
    print("reading archives", flush=True)
    old_ids = variation_ids(SNAPSHOT_FILE)
    next_ids = variation_ids(NEXT_RELEASE_FILE)

    # Current labels.
    print("reading current variant_summary", flush=True)
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
    other_clinsig = vs.loc[vs["label"] == -1, "ClinicalSignificance"].value_counts().head(8).to_dict()
    vs = vs[vs["label"] >= 0]
    n_dup = int(vs["VariationID"].duplicated().sum())
    vs = vs.drop_duplicates("VariationID")
    step(log, "P/LP or B/LB", vids=vs["VariationID"], df=vs,
         note=f"{n_dup} duplicate GRCh38 rows of a VariationID collapsed")
    snapshot_labeled = vs.copy()
    vs = vs[~vs["VariationID"].isin(old_ids)]
    step(log, "VariationID absent from the 2023-06 archive (new)", vids=vs["VariationID"], df=vs)

    # Protein changes on benchmark proteins.
    print("reading hgvs4variation", flush=True)
    hg, n_hgvs_rows = hgvs_on_benchmark(CURRENT_HGVS, set(prot["acc"]))
    any_expr = set(hg["VariationID"])
    new = vs[vs["VariationID"].isin(any_expr)]
    step(log, "has a protein HGVS on a benchmark RefSeq accession (version ignored)", vids=new["VariationID"], df=new)

    res_all, status_all = resolve_changes(hg, prot)
    st = status_all[status_all["VariationID"].isin(new["VariationID"])]
    pair_status = st["status"].value_counts().to_dict()
    mis_ids = set(st.loc[st["status"] != "not_missense", "VariationID"])
    step(log, "missense on a benchmark protein (preferred version)", vids=mis_ids,
         note=f"(VariationID, protein) pair status: {pair_status}")
    res = res_all.merge(new[["VariationID", "label", "ClinicalSignificance", "ReviewStatus", "GeneSymbol", "LastEvaluated"]], on="VariationID")
    step(log, "wild type matches benchmark sequence and change unambiguous", vids=res["VariationID"].unique(), df=res,
         note=f"dropped pairs: wt_mismatch={pair_status.get('wt_mismatch', 0)}, ambiguous={pair_status.get('ambiguous', 0)}")

    # Changes of the initiator methionine are start-loss variants, not missense;
    # the benchmark has no position-1 variants.
    assert (bench["position"] > 1).all()
    n_p1 = int((res["position"] == 1).sum())
    res = res[res["position"] > 1]
    step(log, "not at position 1 (start-loss)", vids=res["VariationID"].unique(), df=res, note=f"{n_p1} rows removed")

    in_bench =[(p, m) in bench_pairs for p, m in zip(res["protein_id"], res["mutant"])]
    res = res[~np.array(in_bench, dtype=bool)]
    step(log, "(protein, change) not in the benchmark", vids=res["VariationID"].unique(), df=res,
         note=f"{int(np.sum(in_bench))} rows removed")

    # Collapse VariationIDs that give the same amino-acid change.
    res["stars"] = res["ReviewStatus"].map(STARS)
    nlab = res.groupby(["protein_id", "mutant"])["label"].transform("nunique")
    n_conflict = int(res.loc[nlab > 1, ["protein_id", "mutant"]].drop_duplicates().shape[0])
    res = res[nlab == 1].copy()
    res["in_next"] = res["VariationID"].isin(next_ids)
    grp = res.sort_values("VariationID").groupby(["protein_id", "mutant"])
    ids = grp["VariationID"].agg(lambda s: ";".join(map(str, s))).rename("variation_ids")
    nids = grp["VariationID"].nunique().rename("n_variation_ids")
    in_next = grp["in_next"].any().rename("in_2023_07_release")
    rep = res.sort_values(["stars", "VariationID"], ascending=[False, True]).drop_duplicates(["protein_id", "mutant"])
    rep = rep.merge(ids, on=["protein_id", "mutant"]).merge(nids, on=["protein_id", "mutant"]).merge(in_next, on=["protein_id", "mutant"])
    step(log, "one row per (protein, change); label conflicts dropped", df=rep,
         note=f"{n_conflict} (protein, change) with conflicting labels across VariationIDs dropped; "
              f"{int((rep['n_variation_ids'] > 1).sum())} rows merge >1 VariationID")

    # Flag changes that some VariationID already in ClinVar on 2023-06-01 maps to.
    old_res = res_all[res_all["VariationID"].isin(old_ids)]
    old_changes = set(zip(old_res["protein_id"], old_res["mutant"]))
    rep["aa_change_in_2023_06_release"] = [(p, m) in old_changes for p, m in zip(rep["protein_id"], rep["mutant"])]

    # ESM-1b LLRs: exact sequence match against the benchmark sequence.
    print("matching Brandes et al. ESM-1b entries", flush=True)
    esm = BrandesLLR(EXT_RAW / "esm1b")
    p = prot.set_index("protein_id")
    todo = sorted(rep["protein_id"].unique())
    cands = {pid: esm.candidates(p.at[pid, "uniprot"], p.at[pid, "gene"]) for pid in todo}
    order = {pid: a + b for pid, (a, b) in cands.items()}
    match, tried, pending, k = {}, {}, set(todo), 0
    while pending:
        batch = {pid: order[pid][k] for pid in pending if k < len(order[pid])}
        esm.prefetch(batch.values())
        for pid in list(pending):
            if pid not in batch:
                pending.discard(pid)
                continue
            tried[pid] = k + 1
            if esm.sequence(batch[pid]) == p.at[pid, "sequence"]:
                acc = batch[pid]
                kind = "uniprot_canonical" if acc == p.at[pid, "uniprot"] else ("uniprot_isoform" if acc in cands[pid][0] else "gene_symbol_fallback")
                match[pid] = (acc, kind)
                pending.discard(pid)
        k += 1
    mt = pd.DataFrame([{"protein_id": pid, "uniprot": p.at[pid, "uniprot"], "gene": p.at[pid, "gene"],
                        "n_candidates": len(order[pid]), "n_tried": tried.get(pid, 0),
                        "brandes_entry": match.get(pid, (None, None))[0], "match_type": match.get(pid, (None, "no_exact_match"))[1]}
                       for pid in todo])
    mt.to_csv(EXT_DOC / "esm1b_matches.tsv", sep="\t", index=False)
    rep = rep[rep["protein_id"].isin(match)].copy()
    step(log, "protein has a Brandes et al. entry with exactly the benchmark sequence", df=rep,
         note=f"match types: {mt['match_type'].value_counts().to_dict()}")

    llr_vals, agree = [], []
    for pid, g in rep.groupby("protein_id"):
        llr, _ = esm.load(match[pid][0])
        for i, w, pos, m in zip(g.index, g["wt"], g["position"], g["mt"]):
            llr_vals.append((i, float(llr.at[m, f"{w} {pos}"])))
        b = bench[bench["protein_id"] == pid]
        for w, pos, m, s in zip(b["variant"].str[0], b["position"], b["variant"].str[-1], b["ESM1b"]):
            agree.append((pid, float(s), -float(llr.at[m, f"{w} {pos}"])))
    rep["esm1b_llr"] = pd.Series(dict(llr_vals))
    rep["esm1b_oriented"] = -rep["esm1b_llr"]
    rep["length"] = rep["protein_id"].map(p["length"]).astype(int)
    rep["over_1022"] = rep["length"] > 1022
    rep["brandes_entry"] = rep["protein_id"].map({k: v[0] for k, v in match.items()})
    rep["uniprot"] = rep["protein_id"].map(p["uniprot"])
    out = rep.rename(columns={"VariationID": "variation_id", "ReviewStatus": "review_status", "ClinicalSignificance": "clinsig",
                              "GeneSymbol": "gene", "LastEvaluated": "last_evaluated"})
    cols = ["protein_id", "uniprot", "gene", "mutant", "position", "label", "esm1b_oriented", "esm1b_llr", "variation_id",
            "variation_ids", "n_variation_ids", "review_status", "clinsig", "last_evaluated", "length", "over_1022",
            "np_accession", "np_version_match", "brandes_entry", "in_2023_07_release", "aa_change_in_2023_06_release"]
    out = out[cols].sort_values(["protein_id", "position", "mutant"]).reset_index(drop=True)
    assert out["esm1b_oriented"].notna().all()
    assert not set(zip(out["protein_id"], out["mutant"])) & bench_pairs
    out.to_parquet(processed_dir(cfg) / "external_temporal.parquet", index=False)

    # Orientation check against the benchmark ESM1b column (both oriented),
    # overall and split by protein length (ESM-1b's window is 1,022 residues).
    ag = pd.DataFrame(agree, columns=["protein_id", "bench_esm1b", "brandes_oriented"])
    ag["over_1022"] = ag["protein_id"].map(p["length"]) > 1022

    def agreement_stats(a):
        d = (a["bench_esm1b"] - a["brandes_oriented"]).abs()
        return {"n_proteins": int(a["protein_id"].nunique()), "n_benchmark_variants": int(len(a)),
                "max_abs_diff": float(d.max()), "median_abs_diff": float(d.median()),
                "n_abs_diff_gt_1e-3": int((d > 1e-3).sum()), "n_abs_diff_gt_0.1": int((d > 0.1).sum()),
                "pearson_r": float(pearsonr(a["bench_esm1b"], a["brandes_oriented"])[0]),
                "spearman_rho": float(spearmanr(a["bench_esm1b"], a["brandes_oriented"])[0])}
    agreement = {"all": agreement_stats(ag), "length_le_1022": agreement_stats(ag[~ag["over_1022"]]),
                 "length_gt_1022": agreement_stats(ag[ag["over_1022"]])}

    # Predictor sanity check (no conformal method is evaluated here).
    def auc_stats(df, score):
        a = per_protein_auc(df, score)
        return {"per_protein_mean_auc": float(a.mean()), "n_proteins_with_both_classes": int(len(a)),
                "overall_auc": float(roc_auc_score(df["label"], df[score])), "n_variants": int(len(df))}
    bench_m = bench[bench["protein_id"].isin(out["protein_id"].unique())]
    bench_m = bench_m.assign(over_1022=bench_m["protein_id"].map(p["length"]) > 1022)
    sanity = {
        "external": auc_stats(out, "esm1b_oriented"),
        "external_length_le_1022": auc_stats(out[~out["over_1022"]], "esm1b_oriented"),
        "benchmark_same_proteins": auc_stats(bench_m, "ESM1b"),
        "benchmark_same_proteins_length_le_1022": auc_stats(bench_m[~bench_m["over_1022"]], "ESM1b"),
    }
    auc_ext = per_protein_auc(out, "esm1b_oriented")

    # Where does the benchmark's own ClinVar snapshot fall? Map benchmark
    # variants to current VariationIDs and ask whether any existed on 2023-06-01.
    bm = res_all.merge(bench[["protein_id", "variant"]].rename(columns={"variant": "mutant"}), on=["protein_id", "mutant"])
    bm["in_06"], bm["in_07"] = bm["VariationID"].isin(old_ids), bm["VariationID"].isin(next_ids)
    bg = bm.groupby(["protein_id", "mutant"])[["in_06", "in_07"]].any()
    lab_old = snapshot_labeled[snapshot_labeled["VariationID"].isin(old_ids)]
    snapshot_diag = {
        "benchmark_variants": int(len(bench)),
        "benchmark_variants_mapped_to_a_current_variation_id": int(len(bg)),
        "mapped_with_no_variation_id_in_2023_06_release": int((~bg["in_06"]).sum()),
        "mapped_with_no_variation_id_in_2023_07_release": int((~bg["in_07"]).sum()),
        "current_labeled_1star_snv_variation_ids_also_in_2023_06_release": int(len(lab_old)),
    }

    summary = {
        "inputs": {
            "snapshot_archive": {"file": "pub/clinvar/tab_delimited/archive/2023/" + SNAPSHOT_FILE.name, "release_date": "2023-06-01",
                                 "sha256": sha256(SNAPSHOT_FILE), "n_variation_ids": len(old_ids)},
            "next_release_archive": {"file": "pub/clinvar/tab_delimited/archive/2023/" + NEXT_RELEASE_FILE.name, "release_date": "2023-07-06",
                                     "sha256": sha256(NEXT_RELEASE_FILE), "n_variation_ids": len(next_ids)},
            "current_variant_summary": {"file": "pub/clinvar/tab_delimited/variant_summary.txt.gz",
                                        "last_modified": (CURRENT_VS.parent / (CURRENT_VS.name + ".last_modified")).read_text().strip(),
                                        "sha256": sha256(CURRENT_VS)},
            "current_hgvs4variation": {"file": "pub/clinvar/tab_delimited/hgvs4variation.txt.gz",
                                       "last_modified": (CURRENT_HGVS.parent / (CURRENT_HGVS.name + ".last_modified")).read_text().strip(),
                                       "sha256": sha256(CURRENT_HGVS), "n_rows": n_hgvs_rows,
                                       "n_rows_on_benchmark_accessions": int(len(hg))},
            "esm1b": {"space": HF_SPACE, "revision": HF_REVISION, "file": ZIP_NAME, "size": ZIP_SIZE,
                      "lfs_sha256": ZIP_LFS_SHA256, "n_entries": len(esm.entries),
                      "central_directory_sha256": sha256(EXT_RAW / "esm1b" / "central_directory.bin")},
        },
        "filter_steps": log,
        "excluded_clinsig_examples_at_label_step": other_clinsig,
        "final": {
            "n_variants": int(len(out)), "n_proteins": int(out["protein_id"].nunique()),
            "n_pathogenic": int(out["label"].sum()), "n_benign": int((1 - out["label"]).sum()),
            "n_proteins_with_both_classes": int(len(auc_ext)),
            "n_proteins_over_1022": int(out.loc[out["over_1022"], "protein_id"].nunique()),
            "n_variants_in_proteins_over_1022": int(out["over_1022"].sum()),
            "n_rows_np_version_differs": int((~out["np_version_match"]).sum()),
            "n_rows_in_2023_07_release": int(out["in_2023_07_release"].sum()),
            "n_rows_aa_change_in_2023_06_release": int(out["aa_change_in_2023_06_release"].sum()),
            "review_status": out["review_status"].value_counts().to_dict(),
            "clinsig": out["clinsig"].value_counts().to_dict(),
        },
        "esm1b_agreement_with_benchmark": agreement,
        "esm1b_sanity_auc": sanity,
        "benchmark_snapshot_diagnostic": snapshot_diag,
    }
    g = out.groupby("protein_id")["label"].agg(["min", "max", "size"])
    le = pd.to_datetime(out["last_evaluated"], format="%b %d, %Y", errors="coerce")
    shared = out[out["variation_id"].duplicated(keep=False)]
    nm = mt[mt["match_type"] == "no_exact_match"]
    summary["final"].update({
        "n_proteins_only_pathogenic": int((g["min"] == 1).sum()), "n_proteins_only_benign": int((g["max"] == 0).sum()),
        "variants_per_protein_median": float(g["size"].median()), "variants_per_protein_max": int(g["size"].max()),
        "n_proteins_ge10_variants": int((g["size"] >= 10).sum()),
        "labels_in_proteins_over_1022": {"pathogenic": int(out.loc[out["over_1022"], "label"].sum()),
                                         "benign": int((1 - out.loc[out["over_1022"], "label"]).sum())},
        "n_rows_merging_gt1_variation_id": int((out["n_variation_ids"] > 1).sum()),
        "n_rows_variation_id_shared_across_proteins": int(len(shared)),
        "n_variation_ids_shared_across_proteins": int(shared["variation_id"].nunique()),
        "n_rows_last_evaluated_before_2023_06_01": int((le < "2023-06-01").sum()),
        "n_rows_last_evaluated_missing": int(le.isna().sum()),
    })
    summary["esm1b_no_exact_match"] = {
        "n_proteins": int(len(nm)), "n_without_uniprot_map": int(nm["uniprot"].isna().sum()),
        "n_without_any_candidate_entry": int((nm["n_candidates"] == 0).sum()),
        "n_variants_lost": int(summary["filter_steps"][-2]["n_rows"] - summary["filter_steps"][-1]["n_rows"]),
    }
    with open(EXT_DOC / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ["final", "esm1b_agreement_with_benchmark", "esm1b_sanity_auc", "benchmark_snapshot_diagnostic"]}, indent=2))


if __name__ == "__main__":
    main()
