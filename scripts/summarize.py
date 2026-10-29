"""Aggregate raw experiment output: seed mean/sd and family-level bootstrap CIs.

Bootstrap: in each replicate, test families are resampled with replacement
within every seed, the metric is recomputed per seed and averaged over seeds.
The same resampling weights are used for every method, so comparisons are
paired. Writes, under results/<tag>/:
  summary.csv         per-family-count metrics: seed mean, seed sd, bootstrap 95% CI
  paired.csv          paired bootstrap differences between selected methods
  seed_summary.csv    seed mean and sd of every metric in metrics.csv
  dispersion.csv      the same, without ablation variants
  ablations.csv       ablation variants (with B3_pool and P_w for reference)
  risk_summary.csv    risk-coverage curves and AURC, seed mean and sd
  risk_vs_random.csv  paired AURC differences against random abstention
  by_support.csv      coverage by support tercile of test families
  splits.csv, gate_params.csv  concatenated over seeds

Usage: python scripts/summarize.py --tag main
"""
import argparse

import numpy as np
import pandas as pd

from protcover.config import load_config, predictor_names, processed_dir, results_dir
from protcover.data import load_proteins, load_variants
from protcover.descriptors import family_descriptors

KEYS = ["scenario", "beta", "predictor", "alpha", "method", "cc"]
N_BOOT = 1000
# (method A, cc A) minus (method B, cc B)
PAIRS = [(("P_w", True), ("B3_pool", True)), (("P_w", True), ("B4_gwcp", True)), (("B3_pool", True), ("B1_split", True)),
         (("P_w", False), ("B3_pool", False)), (("P_w", False), ("B4_gwcp", False)), (("B3_pool", False), ("B1_split", False))]


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
        **({"singleton_error": (w @ c["singleton_err"].to_numpy()) / (w @ c["singleton"].to_numpy())}
           if "singleton_err" in c else {}),
    }


def bootstrap(pf):
    rows, reps = [], {}
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
                reps[(scen, beta) + key + (metric,)] = vals
                seed_vals = np.array([m[metric][0] for m in per_seed])
                rows.append(dict(zip(KEYS, (scen, beta) + key), metric=metric, mean=vals[0],
                                 sd_seeds=seed_vals.std(ddof=1) if len(seeds) > 1 else np.nan,
                                 ci_lo=np.quantile(vals[1:], 0.025), ci_hi=np.quantile(vals[1:], 0.975),
                                 n_seeds=len(seeds)))
    return pd.DataFrame(rows), reps


def paired(reps):
    rows = []
    for (scen, beta, pred, alpha, method, cc, metric), va in reps.items():
        for (ma, ca), (mb, cb) in PAIRS:
            if (method, cc) != (ma, ca):
                continue
            vb = reps.get((scen, beta, pred, alpha, mb, cb, metric))
            if vb is None:
                continue
            d = va - vb
            rows.append({"scenario": scen, "beta": beta, "predictor": pred, "alpha": alpha, "metric": metric,
                         "a": f"{ma}{'_cc' if ca else ''}", "b": f"{mb}{'_cc' if cb else ''}", "diff": d[0],
                         "ci_lo": np.quantile(d[1:], 0.025), "ci_hi": np.quantile(d[1:], 0.975)})
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
    ap.add_argument("--skip-support", action="store_true", help="skip the support-tercile table (main family definition only)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw = results_dir() / "raw" / args.tag
    out = results_dir() / args.tag
    out.mkdir(parents=True, exist_ok=True)
    seed_dirs = sorted(p for p in raw.glob("seed*") if p.is_dir())
    read = lambda name, f: pd.concat([f(d / name) for d in seed_dirs if (d / name).exists()], ignore_index=True)
    pf = read("perfamily.parquet", pd.read_parquet)
    pf["beta"] = pf["beta"].fillna(-1.0)  # S3 has no beta
    summ, reps = bootstrap(pf)
    summ["beta"] = summ["beta"].replace(-1.0, np.nan)
    summ.to_csv(out / "summary.csv", index=False)
    pr = paired(reps)
    pr["beta"] = pr["beta"].replace(-1.0, np.nan)
    pr.to_csv(out / "paired.csv", index=False)

    # Seed mean/sd of every metric in metrics.csv (dispersion, abstention and
    # retained metrics need per-variant data; ablation variants live only here).
    m = read("metrics.csv", pd.read_csv)
    num = [c for c in m.columns if c not in KEYS + ["seed"]]
    d = m.groupby(KEYS, dropna=False)[num].agg(["mean", "std"])
    d.columns = [f"{a}_{b}" for a, b in d.columns]
    d = d.reset_index()
    d.to_csv(out / "seed_summary.csv", index=False)
    d[~d["method"].str.contains(r"\[")].to_csv(out / "dispersion.csv", index=False)
    d[d["method"].str.contains(r"\[") | d["method"].isin(["B3_pool", "P_w"])].to_csv(out / "ablations.csv", index=False)

    # Risk-coverage: seed mean curve per score, and paired AURC differences against random abstention.
    rc = read("risk_coverage.csv", pd.read_csv)
    rkeys = ["scenario", "beta", "predictor", "alpha", "cc", "score"]
    rcols = [c for c in rc.columns if c.startswith("r") and c[1:].replace(".", "").isdigit()] + ["aurc"]
    rs = rc.groupby(rkeys, dropna=False)[rcols].agg(["mean", "std"])
    rs.columns = [f"{a}_{b}" for a, b in rs.columns]
    rs.reset_index().to_csv(out / "risk_summary.csv", index=False)
    base = rc[rc["score"] == "random"].set_index(rkeys[:-1] + ["seed"])["aurc"]
    diff = rc[rc["score"] != "random"].copy()
    diff["d_aurc_vs_random"] = diff["aurc"].to_numpy() - base.loc[pd.MultiIndex.from_frame(diff[rkeys[:-1] + ["seed"]])].to_numpy()
    dd = diff.groupby(rkeys, dropna=False)["d_aurc_vs_random"].agg(
        mean="mean", sd="std", n_seeds="size", frac_seeds_better=lambda x: float((x < 0).mean()))
    dd.reset_index().to_csv(out / "risk_vs_random.csv", index=False)

    for name in ("splits.csv", "gate_params.csv"):
        read(name, pd.read_csv).to_csv(out / name, index=False)
    pf["beta"] = pf["beta"].replace(-1.0, np.nan)
    if not args.skip_support:
        by_support(cfg, pf, cfg["experiment"]["family_def"]).to_csv(out / "by_support.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
