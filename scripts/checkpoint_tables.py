"""Compact checkpoint tables read from results/<tag>/{summary,dispersion,by_support}.csv.

Writes results/<tag>/key_table.csv and prints the tables.
Usage: python scripts/checkpoint_tables.py --tag checkpoint1 [--alpha 0.1]
"""
import argparse

import numpy as np
import pandas as pd

from protcover.config import results_dir

METRICS = ["cov_family", "cov_variant", "cov_path_family", "cov_benign_family", "set_size", "singleton_rate"]


def fmt(r):
    return f"{r['mean']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()
    d = results_dir() / args.tag
    s = pd.read_csv(d / "summary.csv")
    s = s[np.isclose(s["alpha"], args.alpha)]
    s["setting"] = s["scenario"] + np.where(s["beta"].notna(), " b=" + s["beta"].astype(str), "")
    s["meth"] = s["method"] + np.where(s["cc"], "_cc", "")
    s["cell"] = s.apply(fmt, axis=1)
    key = s.pivot_table(index=["setting", "predictor", "meth"], columns="metric", values="cell", aggfunc="first")[METRICS]
    key.to_csv(d / "key_table.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 40)
    print(key.to_string())

    disp = pd.read_csv(d / "dispersion.csv")
    disp = disp[np.isclose(disp["alpha"], args.alpha)]
    cols = ["scenario", "beta", "predictor", "method", "cc", "frac_fam_below_mean", "null_frac_fam_below_mean",
            "fam_cov_sd_mean", "fam_cov_excess_var_mean"]
    print(disp[cols].round(3).to_string(index=False))

    sup = pd.read_csv(d / "by_support.csv")
    sup = sup[np.isclose(sup["alpha"], args.alpha)]
    print(sup.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
