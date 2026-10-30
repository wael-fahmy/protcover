"""Score the pre-registered endpoints and hypotheses of the evaluation on proteins outside the
benchmark (docs/preregistration_unseen.md). Reads results/<tag>/ written by scripts/summarize.py
and writes, under the same folder:
  hypotheses.csv   U1 to U4 per population (U primary, Ustrict secondary): quantity, value, verdict
  endpoints.csv    Cov_1 and Cov_0 per population (class-conditional, every alpha)
  shortfall.csv    worse-class shortfall, set size and singleton error per population, method and alpha
  matched.csv      mean set size at pathogenic coverage 0.90, interpolated along alpha
and results/unseen_data.csv (size of the test set; the predictor's per-protein AUC, a descriptive
statistic that was not pre-registered).

Usage: python scripts/score_unseen.py --tag unseen
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from protcover.config import load_config, processed_dir, results_dir

CC = ["B1_split", "B3_pool", "B3_sub1", "B3_subR", "B4_gwcp", "P_w", "P_full", "C_supp", "C_ratio"]
POPS = ["U", "Ustrict", "Ulen"]  # Ulen is exploratory and has no hypothesis
NO_DIFF = 0.01  # pre-registered: a shortfall difference below this is no difference


def endpoints(summ):
    s = summ[(summ["cc"] == True) & summ["method"].isin(CC) & summ["metric"].isin(["cov_path_family", "cov_benign_family"])]
    s = s.assign(metric=s["metric"].map({"cov_path_family": "cov1", "cov_benign_family": "cov0"}))
    w = s.pivot_table(index=["scenario", "alpha", "method"], columns="metric", values=["mean", "ci_lo", "ci_hi", "sd_seeds"])
    w.columns = [f"{m}_{v}" for v, m in w.columns]
    return w.reset_index()


def shortfall(seed_summ):
    s = seed_summ[(seed_summ["cc"] == True) & seed_summ["method"].isin(CC)].copy()
    s["worst"] = s[["cov_path_family_mean", "cov_benign_family_mean"]].min(axis=1)
    s["short"] = (1 - s["alpha"] - s["worst"]).clip(lower=0)
    cols = {"short": "short", "worst": "worst", "set_size_mean": "set_size", "ret_singleton_error_mean": "singleton_error",
            "abstain_rate_family_mean": "abstain_rate", "full_set_rate_mean": "full_set_rate"}
    return s[["scenario", "alpha", "method", *cols]].rename(columns=cols)


def matched(seed_summ, methods=("B3_pool", "B4_gwcp", "P_w", "C_supp", "C_ratio")):
    """Set size at pathogenic coverage 0.90, interpolated along alpha in {0.2, 0.1, 0.05}; NaN where a
    method's pathogenic coverage does not bracket 0.90."""
    s = seed_summ[(seed_summ["cc"] == True)]
    rows = []
    for (scen, m), g in s[s["method"].isin(methods)].groupby(["scenario", "method"]):
        g = g.sort_values("cov_path_family_mean")
        cov, size = g["cov_path_family_mean"].to_numpy(), g["set_size_mean"].to_numpy()
        rows.append({"scenario": scen, "method": m,
                     "size_matched": np.interp(0.9, cov, size) if cov[0] <= 0.9 <= cov[-1] else np.nan})
    return pd.DataFrame(rows)


def hypotheses(ep, sf):
    rows = []
    for pop in ("U", "Ustrict"):
        e = ep[(ep["scenario"] == pop) & np.isclose(ep["alpha"], 0.1)].set_index("method")
        h = sf[(sf["scenario"] == pop) & np.isclose(sf["alpha"], 0.1)].set_index("method")
        b2 = e.loc["B1_split"]
        rows.append({"population": pop, "hypothesis": "U1", "quantity": "B2 Cov_1", "value": b2["cov1_mean"],
                     "ci_lo": b2["cov1_ci_lo"], "ci_hi": b2["cov1_ci_hi"], "supported": b2["cov1_ci_hi"] < 0.9})
        rows.append({"population": pop, "hypothesis": "U2", "quantity": "ProtCOVER min(Cov_0, Cov_1)",
                     "value": h.loc["C_ratio", "worst"], "supported": h.loc["C_ratio", "worst"] >= 0.89})
        ok = all(h.loc["C_ratio", "short"] - h.loc[b, "short"] < NO_DIFF for b in ("B3_pool", "B4_gwcp"))
        for b in ("B3_pool", "B4_gwcp"):
            rows.append({"population": pop, "hypothesis": "U3", "quantity": f"shortfall ProtCOVER minus {b}",
                         "value": h.loc["C_ratio", "short"] - h.loc[b, "short"], "supported": ok})
        d = h.loc["B4_gwcp", "short"] - h.loc["C_ratio", "short"]
        rows.append({"population": pop, "hypothesis": "U4", "quantity": "shortfall GWCP minus ProtCOVER",
                     "value": d, "supported": d > NO_DIFF})
    return pd.DataFrame(rows)


def data_summary(seed_dirs):
    pdir = processed_dir(load_config(None))
    uv = pd.read_parquet(pdir / "unseen_variants.parquet")
    fams = pd.read_parquet(pdir / "unseen_families.parquet")
    strict = ~fams["has_benchmark_hit_e1e-3"].astype(bool)
    bv = pd.read_parquet(pdir / "variants.parquet")[["protein_id", "variant", "label"]].merge(
        pd.read_parquet(pdir / "unseen_benchmark_esm1b.parquet")[["protein_id", "variant", "scored", "esm1b_brandes_oriented"]],
        on=["protein_id", "variant"])
    bv = bv[bv["scored"].astype(bool)]

    def per_protein_auc(df, score):
        aucs = [roc_auc_score(g["label"], g[score]) for _, g in df.groupby("protein_id") if g["label"].nunique() == 2]
        return float(np.mean(aucs)), len(aucs)

    auc_u, n_u = per_protein_auc(uv, "esm1b_oriented")
    auc_b, n_b = per_protein_auc(bv, "esm1b_brandes_oriented")
    sp = pd.concat([pd.read_csv(d / "splits.csv") for d in seed_dirs])
    rows = {"n_variants": len(uv), "n_pathogenic": int(uv["label"].sum()), "frac_pathogenic": uv["label"].mean(),
            "n_proteins": uv["protein_id"].nunique(), "n_families": len(fams), "n_families_path": int((fams["n_path"] > 0).sum()),
            "n_families_strict": int(strict.sum()), "n_families_strict_path": int((strict & (fams["n_path"] > 0)).sum()),
            "n_variants_strict": int(fams.loc[strict, "n_variants"].sum()),
            "n_train_fam": sp["n_train_fam"].mean(), "n_cal_fam": sp["n_cal_fam"].mean(),
            "n_bench_var_dropped": sp["n_bench_var_dropped"].iloc[0], "w_cal_ess_frac": sp["w_cal_ess_frac"].mean(),
            "test_mean_max_id": sp["test_mean_max_id_to_train"].mean(), "cal_mean_max_id": sp["cal_mean_max_id_to_train"].mean(),
            "test_mean_emb_dist": sp["test_mean_emb_dist_to_train"].mean(), "cal_mean_emb_dist": sp["cal_mean_emb_dist_to_train"].mean(),
            "per_protein_mean_auc": auc_u, "n_proteins_auc": n_u,
            "per_protein_mean_auc_benchmark": auc_b, "n_proteins_auc_benchmark": n_b}
    pd.DataFrame({"statistic": list(rows), "value": list(rows.values())}).to_csv(results_dir() / "unseen_data.csv", index=False)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="unseen")
    args = ap.parse_args()
    pd.set_option("display.width", 200, "display.max_columns", 20)
    out = results_dir() / args.tag
    seed_dirs = sorted(p for p in (results_dir() / "raw" / args.tag).glob("seed*") if p.is_dir())
    dry = args.tag.endswith("_dryrun")  # code-path test on permuted labels: no statistic of the real labels
    if not dry:
        print(data_summary(seed_dirs))
    summ, seed_summ = pd.read_csv(out / "summary.csv"), pd.read_csv(out / "seed_summary.csv")
    ep, sf, mt = endpoints(summ), shortfall(seed_summ), matched(seed_summ)
    hy = hypotheses(ep, sf)
    ep.to_csv(out / "endpoints.csv", index=False)
    sf.to_csv(out / "shortfall.csv", index=False)
    mt.to_csv(out / "matched.csv", index=False)
    hy.to_csv(out / "hypotheses.csv", index=False)
    print(hy.round(4).to_string(index=False))
    for pop in POPS:
        print(f"\n== {pop}")
        print(sf[(sf["scenario"] == pop) & np.isclose(sf["alpha"], 0.1)].round(4).to_string(index=False))
    print(mt.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
