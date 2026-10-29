"""Aggregate raw experiment output: seed mean/sd and family-level bootstrap CIs.

Bootstrap: in each replicate, test families are resampled with replacement
within every seed, the metric is recomputed per seed and averaged over seeds.
The same resampling weights are used for every method, so comparisons are
paired. Writes
  results/<tag>/summary.csv          seed mean, seed sd, bootstrap 95% CI
  results/<tag>/by_support.csv       coverage by support tercile of test families

Usage: python scripts/summarize.py --tag checkpoint1
"""
import argparse

import numpy as np
import pandas as pd

from protcover.config import load_config, predictor_names, processed_dir, results_dir
from protcover.data import load_proteins, load_variants
from protcover.descriptors import family_descriptors

KEYS = ["scenario", "beta", "predictor", "alpha", "method", "cc"]
N_BOOT = 1000


def family_metrics(c, w):
    """Metrics from per-family counts c (DataFrame) under family weights w (B x K)."""
    n, npth, nbn = c["n"].to_numpy(), c["n_path"].to_numpy(), c["n_benign"].to_numpy()
    hp, hb = (npth > 0).astype(float), (nbn > 0).astype(float)
    cp = np.divide(c["k_path"], npth, out=np.zeros(len(n)), where=npth > 0)
    cb = np.divide(c["k_benign"], nbn, out=np.zeros(len(n)), where=nbn > 0)
    return {
        "cov_family": w @ (c["k"].to_numpy() / n) / w.sum(1),
        "cov_variant": (w @ c["k"].to_numpy()) / (w @ n),
        "cov_path_family": (w @ cp) / (w @ hp),
        "cov_benign_family": (w @ cb) / (w @ hb),
        "cov_path_variant": (w @ c["k_path"].to_numpy()) / (w @ npth),
        "set_size": (w @ c["size"].to_numpy()) / (w @ n),
        "singleton_rate": (w @ c["singleton"].to_numpy()) / (w @ n),
    }


def bootstrap(pf):
    rows = []
    for (scen, beta), g in pf.groupby(["scenario", "beta"], dropna=False):
        seeds = sorted(g["seed"].unique())
        weights = {}
        for s in seeds:
            fams = np.sort(g.loc[g["seed"] == s, "family"].unique())
            rng = np.random.default_rng([s, 7])
            weights[s] = (fams, np.vstack([np.ones(len(fams)), rng.multinomial(len(fams), np.full(len(fams), 1 / len(fams)), N_BOOT)]))
        for key, h in g.groupby(["predictor", "alpha", "method", "cc"]):
            per_seed = []
            for s in seeds:
                fams, w = weights[s]
                c = h[h["seed"] == s].set_index("family").loc[fams]
                per_seed.append(family_metrics(c, w))
            for metric in per_seed[0]:
                vals = np.mean([m[metric] for m in per_seed], axis=0)  # row 0 is the point estimate
                seed_vals = np.array([m[metric][0] for m in per_seed])
                rows.append(dict(zip(KEYS, (scen, beta) + key), metric=metric, mean=vals[0],
                                 sd_seeds=seed_vals.std(ddof=1) if len(seeds) > 1 else np.nan,
                                 ci_lo=np.quantile(vals[1:], 0.025), ci_hi=np.quantile(vals[1:], 0.975),
                                 n_seeds=len(seeds)))
    return pd.DataFrame(rows)


def by_support(cfg, pf, fam_col):
    """Coverage of test families split into terciles of max identity to train."""
    variants, proteins = load_variants(cfg), load_proteins(cfg)
    fam = pd.read_parquet(processed_dir(cfg) / "families.parquet")[["protein_id", fam_col]]
    preds = predictor_names(cfg)
    out = []
    for (scen, seed), g in pf[pf["scenario"].isin(["S1", "S2"])].groupby(["scenario", "seed"]):
        mid = pd.read_parquet(processed_dir(cfg) / "cache" / f"maxid_{fam_col}_S12_seed{seed}.parquet")
        desc = family_descriptors(variants, proteins, fam, fam_col, preds, mid)
        out.append(g.merge(desc[["max_id_to_train"]], left_on="family", right_index=True))
    # Terciles of support among the test families of each split.
    d = pd.concat(out)
    d["support_bin"] = d.groupby(["scenario", "beta", "seed", "predictor", "alpha", "method", "cc"])["max_id_to_train"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=["low", "mid", "high"]))
    d["cov"] = d["k"] / d["n"]
    d["cov_path"] = np.where(d["n_path"] > 0, d["k_path"] / d["n_path"].where(d["n_path"] > 0, 1), np.nan)
    agg = d.groupby(["scenario", "beta", "predictor", "alpha", "method", "cc", "support_bin", "seed"], observed=True).agg(
        cov_family=("cov", "mean"), cov_path_family=("cov_path", "mean"), mean_max_id=("max_id_to_train", "mean"))
    res = agg.groupby(level=[0, 1, 2, 3, 4, 5, 6], observed=True).agg(["mean", "std"])
    res.columns = [f"{a}_{b}" for a, b in res.columns]
    return res.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw = results_dir() / "raw" / args.tag
    out = results_dir() / args.tag
    out.mkdir(parents=True, exist_ok=True)
    pf = pd.read_parquet(raw / "perfamily.parquet")
    pf["beta"] = pf["beta"].fillna(-1.0)  # S3 has no beta
    summ = bootstrap(pf)
    summ["beta"] = summ["beta"].replace(-1.0, np.nan)
    summ.to_csv(out / "summary.csv", index=False)
    # Seed mean/sd of the dispersion metrics, which need per-variant data and so come from metrics.csv.
    m = pd.read_csv(raw / "metrics.csv")
    disp = ["frac_fam_below", "null_frac_fam_below", "fam_cov_sd", "fam_cov_q10", "fam_cov_excess_var", "full_set_rate", "empty_rate"]
    d = m.groupby(KEYS, dropna=False)[disp].agg(["mean", "std"])
    d.columns = [f"{a}_{b}" for a, b in d.columns]
    d.reset_index().to_csv(out / "dispersion.csv", index=False)
    pf["beta"] = pf["beta"].replace(-1.0, np.nan)
    by_support(cfg, pf, cfg["experiment"]["family_def"]).to_csv(out / "by_support.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
