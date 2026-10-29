"""Evaluation metrics for binary-label conformal sets, with families as units."""
import numpy as np
import pandas as pd
from scipy.stats import binom


def per_family(in_set, label, family):
    """Per-family coverage table (all test families)."""
    cov = in_set[np.arange(len(label)), label]
    df = pd.DataFrame({"family": family, "covered": cov, "label": label})
    return df.groupby("family").agg(n=("covered", "size"), coverage=("covered", "mean"))


def per_family_counts(in_set, label, family):
    """Per-family sufficient counts, so any coverage or size metric can be
    re-aggregated (and bootstrapped over families) without the variant table."""
    in_set, label = np.asarray(in_set), np.asarray(label)
    cov = in_set[np.arange(len(label)), label]
    size = in_set.sum(1)
    df = pd.DataFrame({"family": family, "n": 1, "n_path": label, "k": cov, "k_path": cov & (label == 1),
                       "k_benign": cov & (label == 0), "size": size, "singleton": size == 1, "full": size == 2})
    out = df.groupby("family").sum()
    out["n_benign"] = out["n"] - out["n_path"]
    return out.astype(int)


def null_frac_below(n, level, thresh):
    """Expected fraction of families with empirical coverage < thresh if every
    family had true coverage exactly `level` (pure binomial noise)."""
    n = np.asarray(n)
    k_max = np.ceil(thresh * n) - 1  # coverage < thresh  <=>  covered count <= k_max
    return float(np.mean(binom.cdf(k_max, n, level)))


def evaluate(in_set, label, family, alpha, min_variants=10):
    in_set, label, family = np.asarray(in_set), np.asarray(label), np.asarray(family)
    cov = in_set[np.arange(len(label)), label]
    size = in_set.sum(1)
    pf = per_family(in_set, label, family)
    big = pf[pf["n"] >= min_variants]
    thresh = 1 - alpha - 0.05
    out = {
        "cov_family": pf["coverage"].mean(),
        "cov_variant": cov.mean(),
        "set_size": size.mean(),
        "set_size_family": pd.Series(size).groupby(family).mean().mean(),
        "singleton_rate": (size == 1).mean(),
        "full_set_rate": (size == 2).mean(),
        "empty_rate": (size == 0).mean(),
        "n_test_families": len(pf),
        "n_test_variants": len(label),
        "n_fam_ge_min": len(big),
        "frac_fam_below": (big["coverage"] < thresh).mean() if len(big) else np.nan,
        "null_frac_fam_below": null_frac_below(big["n"], 1 - alpha, thresh) if len(big) else np.nan,
        "fam_cov_sd": big["coverage"].std(ddof=0) if len(big) else np.nan,
        "fam_cov_q10": big["coverage"].quantile(0.1) if len(big) else np.nan,
    }
    if len(big):
        cbar = big["coverage"].mean()
        out["fam_cov_excess_var"] = big["coverage"].var(ddof=0) - np.mean(cbar * (1 - cbar) / big["n"])
    for y, name in ((1, "path"), (0, "benign")):
        k = label == y
        out[f"cov_{name}_variant"] = cov[k].mean()
        out[f"cov_{name}_family"] = pd.Series(cov[k]).groupby(family[k]).mean().mean()
    out["path_miss_variant"] = 1 - out["cov_path_variant"]
    out["path_miss_family"] = 1 - out["cov_path_family"]
    return out
