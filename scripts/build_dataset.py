"""Step 0: build data/processed/{variants,proteins}.parquet and results/data_summary.csv.

Also writes results/sign_check.csv, which reproduces ProteinGym's published
per-gene average AUC for each predictor with our score orientation. This is a
check of the sign convention only; nothing downstream reads it.

Usage: python scripts/build_dataset.py [--config configs/default.yaml]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from protcover.config import load_config, predictor_names, processed_dir, raw_dir, results_dir
from protcover.data import build_proteins, build_variants

PUBLISHED_NAMES = {"ESM1b": "ESM-1b", "EVE": "EVE", "GEMME": "GEMME", "TranceptEVE_L": "TranceptEVE L", "PoET": "PoET (200M)"}


def summarize(variants, proteins):
    n = variants.groupby("protein_id").size()
    q = n.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    rows = [
        ("n_proteins", variants["protein_id"].nunique()),
        ("n_variants", len(variants)),
        ("n_pathogenic", int(variants["label"].sum())),
        ("n_benign", int((1 - variants["label"]).sum())),
        ("frac_pathogenic", variants["label"].mean()),
        ("n_proteins_with_uniprot", int(proteins["uniprot"].notna().sum())),
        ("variants_per_protein_mean", n.mean()),
        ("variants_per_protein_sd", n.std()),
        ("variants_per_protein_min", n.min()),
        ("variants_per_protein_q10", q[0.1]),
        ("variants_per_protein_q25", q[0.25]),
        ("variants_per_protein_median", q[0.5]),
        ("variants_per_protein_q75", q[0.75]),
        ("variants_per_protein_q90", q[0.9]),
        ("variants_per_protein_max", n.max()),
        ("n_proteins_ge10_variants", int((n >= 10).sum())),
        ("frac_pathogenic_per_protein_median", variants.groupby("protein_id")["label"].mean().median()),
        ("protein_length_median", proteins["length"].median()),
        ("msa_neff_median", proteins["msa_neff"].median()),
    ]
    return pd.DataFrame(rows, columns=["statistic", "value"])


def sign_check(cfg, variants):
    pub = pd.read_csv(raw_dir(cfg) / "published_clinical_AUC_summary.csv")
    pub = pub.set_index("Model_name")["Average_AUC"]
    rows = []
    for p in predictor_names(cfg):
        aucs = variants.groupby("protein_id").apply(lambda g: roc_auc_score(g["label"], g[p]), include_groups=False)
        rows.append({"predictor": p, "per_protein_mean_auc_ours": round(float(np.mean(aucs)), 3),
                     "published_average_auc": pub.get(PUBLISHED_NAMES[p], np.nan)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    variants = build_variants(cfg)
    proteins = build_proteins(cfg, variants)
    out = processed_dir(cfg)
    variants.to_parquet(out / "variants.parquet", index=False)
    proteins.to_parquet(out / "proteins.parquet", index=False)
    summary = summarize(variants, proteins)
    summary.to_csv(results_dir() / "data_summary.csv", index=False)
    sc = sign_check(cfg, variants)
    sc.to_csv(results_dir() / "sign_check.csv", index=False)
    print(summary.to_string(index=False))
    print(sc.to_string(index=False))


if __name__ == "__main__":
    main()
