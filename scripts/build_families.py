"""Step 1a: protein families and the all-vs-all homology table.

Writes
  data/processed/families.parquet   protein_id plus one family column per definition
  data/processed/homology.parquet   all-vs-all MMseqs2 hits (self hits removed)
  results/families_summary.csv

Family definitions
  fam_mm030  MMseqs2 cluster, --min-seq-id 0.3 -c 0.8 (main)
  fam_mm050  MMseqs2 cluster, --min-seq-id 0.5 -c 0.8 (sensitivity)
  fam_hc     single-linkage components of significant homology (E <= 1e-3,
             identity >= 0.3, both sequences covered >= 0.5); stricter
             sensitivity check that merges partial-coverage homologs

Usage: python scripts/build_families.py [--config configs/default.yaml]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from protcover.config import load_config, processed_dir, results_dir
from protcover.data import load_proteins, load_variants
from protcover.families import all_vs_all, cluster, orphan_flags

THREADS = os.cpu_count() or 4


def homology_components(proteins, hits, evalue, min_id, min_cov):
    idx = {k: i for i, k in enumerate(proteins["protein_id"])}
    h = hits[(hits["evalue"] <= evalue) & (hits["fident"] >= min_id) & (np.minimum(hits["qcov"], hits["tcov"]) >= min_cov)]
    i, j = h["query"].map(idx).to_numpy(), h["target"].map(idx).to_numpy()
    a = coo_matrix((np.ones(len(i)), (i, j)), shape=(len(idx), len(idx)))
    _, lab = connected_components(a, directed=False)
    # Name each component after its lexicographically smallest member.
    s = pd.Series(proteins["protein_id"].to_numpy(), index=lab)
    rep = s.groupby(level=0).min()
    return pd.DataFrame({"protein_id": proteins["protein_id"], "fam_hc": rep.loc[lab].to_numpy()})


def summarize(fam, variants, hits, cfg):
    hom = cfg["families"]["homology"]
    nvar = variants.groupby("protein_id").size()
    rows = []
    for col in [c for c in fam.columns if c.startswith("fam_")]:
        size = fam.groupby(col).size()
        fv = fam.assign(n=fam["protein_id"].map(nvar)).groupby(col)["n"].sum()
        orph = orphan_flags(fam, col, hits, hom["evalue"], hom["min_id"])
        rows.append({
            "definition": col, "n_families": len(size), "n_singleton_families": int((size == 1).sum()),
            "max_family_size": int(size.max()), "mean_family_size": size.mean(),
            "n_families_ge10_variants": int((fv >= 10).sum()), "variants_per_family_median": fv.median(),
            "variants_per_family_max": int(fv.max()), "n_orphan_families": int(orph.sum()),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    fc = cfg["families"]
    proteins, variants = load_proteins(cfg), load_variants(cfg)

    fam = proteins[["protein_id"]].copy()
    for t in [fc["min_seq_id"]] + fc["sensitivity_min_seq_id"]:
        c = cluster(proteins, t, fc["coverage"], fc["cov_mode"], threads=THREADS)
        fam = fam.merge(c.rename(columns={"family_id": f"fam_mm{int(round(t * 100)):03d}"}), on="protein_id")

    hits = all_vs_all(proteins, fc["search_sensitivity"], fc["search_evalue"], threads=THREADS)
    hom = fc["homology"]
    fam = fam.merge(homology_components(proteins, hits, hom["evalue"], hom["min_id"], hom["min_cov"]), on="protein_id")

    out = processed_dir(cfg)
    fam.to_parquet(out / "families.parquet", index=False)
    hits.to_parquet(out / "homology.parquet", index=False)
    summ = summarize(fam, variants, hits, cfg)
    summ.to_csv(results_dir() / "families_summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
