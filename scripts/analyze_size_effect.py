"""Why variant-level class-conditional calibration misses pathogenic variants per family.

From the per-family counts of S1 (B2 = class-conditional variant-level split
conformal, alpha = 0.1), for each predictor: the share of pathogenic variants
held by the 5% of test families with the most pathogenic variants, and the
mean per-family pathogenic coverage for families with few (< 5) and many
(>= 20) pathogenic variants. Writes results/<tag>/size_effect.csv.

Usage: python scripts/analyze_size_effect.py --tag main
"""
import argparse

import numpy as np
import pandas as pd

from protcover.config import results_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    tag = ap.parse_args().tag
    raw = results_dir() / "raw" / tag
    pf = pd.concat([pd.read_parquet(p / "perfamily.parquet") for p in sorted(raw.glob("seed*")) if p.is_dir()])
    pf = pf[(pf["scenario"] == "S1") & np.isclose(pf["alpha"], 0.1) & (pf["method"] == "B1_split") & pf["cc"] & (pf["n_path"] > 0)]
    rows = []
    for (pred, seed), g in pf.groupby(["predictor", "seed"]):
        cov = g["k_path"] / g["n_path"]
        top = g["n_path"].sort_values(ascending=False)
        k = max(1, int(round(0.05 * len(g))))
        rows.append({"predictor": pred, "seed": seed, "share_path_top5pct": top.iloc[:k].sum() / top.sum(),
                     "cov_path_few": cov[g["n_path"] < 5].mean(), "cov_path_many": cov[g["n_path"] >= 20].mean()})
    out = pd.DataFrame(rows).groupby("predictor").mean(numeric_only=True).drop(columns="seed")
    out.to_csv(results_dir() / tag / "size_effect.csv")
    print(out.round(3))


if __name__ == "__main__":
    main()
