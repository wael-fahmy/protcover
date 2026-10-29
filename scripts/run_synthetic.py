"""Synthetic validation with known truth (paper Section "Synthetic validation").

A1 (exchangeable families): random family splits, with and without a link
between family size and difficulty. A2 (exact family-level covariate shift):
calibration descriptors N(0, 1), test descriptors N(-beta, 1); and a shift whose
density ratio is linear in the descriptor (conditional calibration on it is exact).

Writes results/synthetic.csv: mean and Monte Carlo standard error over
replicates of each metric, per setting and method.

Usage: python scripts/run_synthetic.py
"""
import numpy as np
import pandas as pd

from protcover.base_model import nonconformity
from protcover.conditional import conditional_sets
from protcover.conformal import conformal_sets, subsample_sets
from protcover.config import results_dir
from protcover.density_ratio import cross_fit_weights
from protcover.metrics import evaluate
from protcover.synthetic import simulate_families, simulate_shift

ALPHA = 0.1
N_REP = 300
METRICS = ["cov_family", "cov_variant", "cov_path_family", "cov_benign_family", "set_size"]


def exchangeable(size_d, rng, k_cal=150, k_test=150):
    sim = simulate_families(k_cal + k_test, rng, size_d=size_d, slope_d=1.0)
    perm = rng.permutation(k_cal + k_test)
    fam, y = sim["family"], sim["label"]
    a = nonconformity(sim["p1"])
    c, t = np.isin(fam, perm[:k_cal]), np.isin(fam, perm[k_cal:])
    args = (a[c, y[c]], y[c], fam[c], a[t])
    sets = {
        "B1_split": conformal_sets(*args, ALPHA, "variant"),
        "B1_split_cc": conformal_sets(*args, ALPHA, "variant", True),
        "B3_pool": conformal_sets(*args, ALPHA, "family"),
        "B3_pool_cc": conformal_sets(*args, ALPHA, "family", True),
        "B3_sub1": subsample_sets(*args, ALPHA, 1, rng),
        "B3_subR": subsample_sets(*args, ALPHA, 50, rng),
    }
    return {m: evaluate(s, y[t], fam[t], ALPHA) for m, s in sets.items()}


def shifted(beta, rng, k_cal=300, k_test=150):
    cal, test, w = simulate_shift(k_cal, k_test, beta, rng, slope_d=1.0)
    a_cal_all, a_test = nonconformity(cal["p1"]), nonconformity(test["p1"])
    y, f, yt, ft = cal["label"], cal["family"], test["label"], test["family"]
    a_cal = a_cal_all[np.arange(len(y)), y]
    out = {}
    n_fit = k_cal // 3
    fold = rng.permutation(np.arange(k_test) % 5)
    wc, wt = cross_fit_weights(cal["d"][:n_fit, None], cal["d"][n_fit:, None], test["d"][:, None], fold, 20.0)
    keep = f >= n_fit
    for cc in (False, True):
        sfx = "_cc" if cc else ""
        out["B3_pool" + sfx] = conformal_sets(a_cal, y, f, a_test, ALPHA, "family", cc)
        out["P_w_oracle" + sfx] = conformal_sets(a_cal, y, f, a_test, ALPHA, "family", cc,
                                                 w_cal=w(cal["d"])[f], w_test=w(test["d"])[ft])
        est = np.zeros(a_test.shape, bool)
        for j in range(5):
            sel = fold[ft] == j
            est[sel] = conformal_sets(a_cal[keep], y[keep], f[keep], a_test[sel], ALPHA, "family", cc,
                                      w_cal=wc[j][f[keep] - n_fit], w_test=wt[ft[sel]])
        out["P_w" + sfx] = est
    out.update(conditional(a_cal, y, f, cal["d"], a_test, ft, test["d"], w, keep, n_fit, wc, wt, fold))
    return {m: evaluate(s, yt, ft, ALPHA) for m, s in out.items()}


def conditional(a_cal, y, f, d_cal, a_test, ft, d_test, w, keep, n_fit, wc, wt, fold):
    """Class-conditional family-level conditional calibration with features (1, true ratio),
    (1, cross-fitted estimated ratio) and (1, descriptor)."""
    phi = lambda v: np.column_stack([np.ones(len(v)), v])
    out = {"C_oracle_cc": conditional_sets(a_cal, y, f, phi(w(d_cal))[f], a_test, ft, phi(w(d_test)), ALPHA, True),
           "C_supp_cc": conditional_sets(a_cal, y, f, phi(d_cal)[f], a_test, ft, phi(d_test), ALPHA, True)}
    est = np.zeros(a_test.shape, bool)
    for j in range(5):
        sel = fold[ft] == j
        est[sel] = conditional_sets(a_cal[keep], y[keep], f[keep], phi(wc[j])[f[keep] - n_fit], a_test[sel], ft[sel],
                                    phi(wt), ALPHA, True)
    out["C_ratio_cc"] = est
    return out


def linear_shift(rng, k_cal=300, k_test=150):
    """Family-level covariate shift whose density ratio is linear in the descriptor:
    calibration d ~ U(-2, 2), test d with density (2 - d) / 8 on [-2, 2], so w(d) = (2 - d) / 2
    lies in the span of (1, d) and conditional calibration on d is exact (Theorem 1)."""
    cal = simulate_families(k_cal, rng, slope_d=1.0, d=rng.uniform(-2.0, 2.0, k_cal))
    test = simulate_families(k_test, rng, slope_d=1.0, d=2.0 - 4.0 * np.sqrt(rng.random(k_test)))
    w = lambda d: (2.0 - np.asarray(d)) / 2.0
    a_cal_all, a_test = nonconformity(cal["p1"]), nonconformity(test["p1"])
    y, f, yt, ft = cal["label"], cal["family"], test["label"], test["family"]
    a_cal = a_cal_all[np.arange(len(y)), y]
    n_fit = k_cal // 3
    fold = rng.permutation(np.arange(k_test) % 5)
    wc, wt = cross_fit_weights(cal["d"][:n_fit, None], cal["d"][n_fit:, None], test["d"][:, None], fold, 20.0)
    keep = f >= n_fit
    out = {"B3_pool_cc": conformal_sets(a_cal, y, f, a_test, ALPHA, "family", True),
           "P_w_oracle_cc": conformal_sets(a_cal, y, f, a_test, ALPHA, "family", True,
                                           w_cal=w(cal["d"])[f], w_test=w(test["d"])[ft])}
    out.update(conditional(a_cal, y, f, cal["d"], a_test, ft, test["d"], w, keep, n_fit, wc, wt, fold))
    return {m: evaluate(s, yt, ft, ALPHA) for m, s in out.items()}


def summarize(setting, param, reps):
    rows = []
    for m in reps[0]:
        for k in METRICS:
            v = np.array([r[m][k] for r in reps])
            rows.append({"setting": setting, "param": param, "method": m, "metric": k,
                         "mean": v.mean(), "mc_se": v.std(ddof=1) / np.sqrt(len(v)), "n_rep": len(v)})
    return rows


def main():
    rows = []
    for size_d in (0.0, 1.0):
        rng = np.random.default_rng([1, int(size_d * 10)])
        rows += summarize("A1_exchangeable", size_d, [exchangeable(size_d, rng) for _ in range(N_REP)])
    for beta in (0.5, 1.0, 2.0):
        rng = np.random.default_rng([2, int(beta * 10)])
        rows += summarize("A2_shift", beta, [shifted(beta, rng) for _ in range(N_REP)])
    rng = np.random.default_rng([3])
    rows += summarize("A2_linear", 1.0, [linear_shift(rng) for _ in range(N_REP)])
    df = pd.DataFrame(rows)
    df.to_csv(results_dir() / "synthetic.csv", index=False)
    print(df[df["metric"].isin(["cov_family", "cov_path_family"])].pivot_table(
        index=["setting", "param", "method"], columns="metric", values="mean").round(4).to_string())


if __name__ == "__main__":
    main()
