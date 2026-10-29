"""Score the pre-registered endpoints and hypotheses of the external evaluation
(docs/preregistration_external.md, with Amendment 1). Reads results/<tag>/ written by
scripts/summarize.py and writes, under the same folder:
  hypotheses.csv   H1, H2, H3a, H3b, H4: the quantity, its value and the verdict
  endpoints.csv    Cov_1 and Cov_0 per setting (class-conditional, every alpha)
  shortfall.csv    worse-class shortfall over the shifted settings, per method and alpha
  matched.csv      mean set size at matched pathogenic coverage 0.90

Usage: python scripts/score_external.py --tags external external_main
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from protcover.config import results_dir

CC = ["B1_split", "B3_pool", "B3_sub1", "B3_subR", "B4_gwcp", "P_w", "P_full", "C_supp", "C_ratio"]
NO_DIFF = 0.01  # pre-registered: a shortfall difference below this is no difference


def shifted(d):
    return d["scenario"].isin(["S3", "S4"]) | ((d["scenario"] == "S2") & (d["beta"].fillna(0) > 0))


def endpoints(summ):
    s = summ[(summ["cc"] == True) & summ["method"].isin(CC) & summ["metric"].isin(["cov_path_family", "cov_benign_family"])]
    s = s.assign(metric=s["metric"].map({"cov_path_family": "cov1", "cov_benign_family": "cov0"}))
    w = s.pivot_table(index=["scenario", "beta", "alpha", "method"], columns="metric",
                      values=["mean", "ci_lo", "ci_hi", "sd_seeds"], dropna=False)
    w.columns = [f"{m}_{v}" for v, m in w.columns]
    return w.dropna(how="all").reset_index()


def shortfall(seed_summ):
    s = seed_summ[(seed_summ["cc"] == True) & seed_summ["method"].isin(CC) & shifted(seed_summ)].copy()
    s["worst"] = s[["cov_path_family_mean", "cov_benign_family_mean"]].min(axis=1)
    s["short"] = (1 - s["alpha"] - s["worst"]).clip(lower=0)
    agg = s.groupby(["alpha", "method"]).agg(
        short_mean=("short", "mean"), short_max=("short", "max"), worst_min=("worst", "min"), n_cells=("short", "size"),
        set_size=("set_size_mean", "mean"), singleton_error=("ret_singleton_error_mean", "mean"),
        abstain_rate=("abstain_rate_family_mean", "mean"))
    return s, agg.reset_index()


def matched(seed_summ, methods=("B3_pool", "B4_gwcp", "P_w", "C_supp", "C_ratio")):
    """Set size at pathogenic coverage 0.90, interpolated along alpha, over the shifted settings
    where every method reaches 0.90."""
    s = seed_summ[(seed_summ["cc"] == True) & shifted(seed_summ)].assign(beta=lambda d: d["beta"].fillna(-1))
    sizes = {}
    for m in methods:
        sizes[m] = {}
        for cell, g in s[s["method"] == m].groupby(["scenario", "beta"]):
            g = g.sort_values("cov_path_family_mean")
            cov, size = g["cov_path_family_mean"].to_numpy(), g["set_size_mean"].to_numpy()
            if cov[0] <= 0.9 <= cov[-1]:
                sizes[m][cell] = np.interp(0.9, cov, size)
    common = sorted(set.intersection(*(set(v) for v in sizes.values())))
    return pd.DataFrame([{"method": m, "size_matched": np.mean([sizes[m][c] for c in common]), "n_cells": len(common)}
                         for m in methods])


def hypotheses(ep, agg, metrics):
    s1 = ep[(ep["scenario"] == "S1") & np.isclose(ep["alpha"], 0.1)].set_index("method")
    rows = [{"hypothesis": "H1", "quantity": "S1 B2 Cov_1", "value": s1.loc["B1_split", "cov1_mean"],
             "ci_lo": s1.loc["B1_split", "cov1_ci_lo"], "ci_hi": s1.loc["B1_split", "cov1_ci_hi"],
             "supported": s1.loc["B1_split", "cov1_mean"] < 0.9}]
    for c in ("cov1", "cov0"):
        rows.append({"hypothesis": "H2", "quantity": f"S1 HCP {c.replace('cov', 'Cov_')}", "value": s1.loc["B3_pool", f"{c}_mean"],
                     "ci_lo": s1.loc["B3_pool", f"{c}_ci_lo"], "ci_hi": s1.loc["B3_pool", f"{c}_ci_hi"],
                     "supported": s1.loc["B3_pool", "cov1_mean"] >= 0.89 and s1.loc["B3_pool", "cov0_mean"] >= 0.89})
    h = agg[np.isclose(agg["alpha"], 0.1)].set_index("method")["short_mean"]
    for c, nm in (("C_supp", "H3a"), ("C_ratio", "H3b")):
        ok = h[c] - h["B3_pool"] < NO_DIFF and h[c] - h["B4_gwcp"] < NO_DIFF
        for b in ("B3_pool", "B4_gwcp"):
            rows.append({"hypothesis": nm, "quantity": f"mean shortfall {c} minus {b}", "value": h[c] - h[b],
                         "supported": ok})
    # H4 has no numeric rule in the pre-registration. Rule fixed after the run: abstention reduces the
    # error if, in some setting, the seed-paired 95% t-interval of P_full minus B5_random lies below 0.
    m = metrics[(metrics["cc"] == True) & np.isclose(metrics["alpha"], 0.1)].assign(beta=lambda d: d["beta"].fillna(-1))
    err = m.pivot_table(index=["scenario", "beta", "seed"], columns="method", values="ret_singleton_error")
    d = (err["P_full"] - err["B5_random"]).groupby(level=[0, 1]).agg(["mean", "std", "size"])
    half = stats.t.ppf(0.975, d["size"] - 1) * d["std"].fillna(0) / np.sqrt(d["size"])
    hi = d["mean"] + half
    worst = d["mean"].idxmin()
    rows.append({"hypothesis": "H4", "quantity": "retained singleton error P_full minus B5_random (setting with the most negative mean)",
                 "value": d.loc[worst, "mean"], "ci_lo": d.loc[worst, "mean"] - half[worst], "ci_hi": hi[worst],
                 "supported": bool((hi >= 0).all())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["external", "external_main"])
    args = ap.parse_args()
    pd.set_option("display.width", 200, "display.max_columns", 20)
    for tag in args.tags:
        out = results_dir() / tag
        summ, seed_summ = pd.read_csv(out / "summary.csv"), pd.read_csv(out / "seed_summary.csv")
        ep = endpoints(summ)
        _, agg = shortfall(seed_summ)
        mt = matched(seed_summ)
        metrics = pd.concat([pd.read_csv(p) for p in sorted((results_dir() / "raw" / tag).glob("seed*/metrics.csv"))])
        hy = hypotheses(ep, agg, metrics)
        ep.to_csv(out / "endpoints.csv", index=False)
        agg.to_csv(out / "shortfall.csv", index=False)
        mt.to_csv(out / "matched.csv", index=False)
        hy.to_csv(out / "hypotheses.csv", index=False)
        print(f"\n== {tag}\n{hy.round(4).to_string(index=False)}")
        print(agg[np.isclose(agg["alpha"], 0.1)].round(4).to_string(index=False))
        print(mt.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
