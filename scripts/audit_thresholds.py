"""Numerical audit of the conditional-calibration threshold under solver tolerances.

For a random sample of test families, compares the production threshold (dual program, HiGHS
warm start, test weight tau + 1e-4; protcover/conditional.py) with a reference supremum computed
independently, in two stages, on the primal program with the exact test weight tau:
  1. L* = min_b L(b); the reference is +inf if L is unbounded below;
  2. reference = max {Phi_t'b : L(b) <= L* + delta}, +inf if unbounded.
The slack delta can only raise the reference, so a production threshold that is at least the
reference is conservative. For each audited family it also counts the test variants whose
membership would differ between the two thresholds.

Writes results/audit/thresholds.csv (one row per audited threshold) and summary.csv.
Usage: python scripts/audit_thresholds.py --seeds 0 --n-fam 8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import SCENARIO_CODE, load_context, make_splits  # noqa: E402

from protcover.base_model import fit_predict, nonconformity  # noqa: E402
from protcover.conditional import THRESHOLD_TOL, FamilyQuantileLP  # noqa: E402
from protcover.config import load_config, results_dir  # noqa: E402
from protcover.pipeline import Split, conditional_programs  # noqa: E402

ALPHA = 0.1
OPTS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}
DELTA = 1e-8  # relative slack on the optimal value in stage 2


def reference_sup(a_k, f_k, phi_k, phi_t, tau):
    n, p = phi_k.shape
    _, codes, counts = np.unique(f_k, return_inverse=True, return_counts=True)
    w = 1.0 / counts[codes]
    a_eq = sparse.hstack([sparse.csr_matrix(phi_k), sparse.identity(n), -sparse.identity(n)]).tocsr()
    bounds = [(None, None)] * p + [(0, None)] * (2 * n)
    c1 = np.r_[-tau * phi_t, tau * w, (1 - tau) * w]
    r1 = linprog(c1, A_eq=a_eq, b_eq=a_k, bounds=bounds, method="highs", options=OPTS)
    if r1.status == 3:
        return np.inf
    if r1.status != 0:
        raise RuntimeError(r1.message)
    c2 = np.r_[-phi_t, np.zeros(2 * n)]
    r2 = linprog(c2, A_ub=c1[None, :], b_ub=[r1.fun + DELTA * max(1.0, abs(r1.fun))], A_eq=a_eq, b_eq=a_k,
                 bounds=bounds, method="highs", options=OPTS)
    if r2.status == 3:
        return np.inf
    if r2.status != 0:
        raise RuntimeError(r2.message)
    return -r2.fun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--n-fam", type=int, default=8,
                    help="test families audited per fold and class (all folds share one program for C_supp)")
    args = ap.parse_args()
    cfg = load_config()
    ex = cfg["experiment"]
    ctx = load_context(cfg, ex["family_def"])
    vfam, label = ctx.vfam, ctx.label
    rows = []
    for seed in args.seeds:
        for scen, beta, train, cal, test, desc, tag in make_splits(ctx, ["S1", "S2", "S3", "S4"], seed):
            b100 = 0 if beta != beta else int(round(beta * 100))
            split = Split(desc, train, cal, test, ex, np.random.default_rng([seed, SCENARIO_CODE[scen], b100, 5]))
            w_cal, w_test = split.weights("main", ex["clip"])
            tr = np.isin(vfam, train)
            for pred in ctx.preds:
                a = nonconformity(fit_predict(ctx.variants, pred, tr, within=True))
                rng = np.random.default_rng([seed, SCENARIO_CODE[scen], b100, ctx.preds.index(pred), 77])
                for method, kw in [("C_ratio", {"w_cal": w_cal, "w_test": w_test}),
                                   ("C_supp", {"cols": ex["conditional"]["primary"]})]:
                    for test_idx, phi_test, y, a_k, f_k, phi_k in conditional_programs(split, a, label, vfam, True, **kw):
                        lp = FamilyQuantileLP(a_k, f_k, phi_k, ALPHA)
                        size = args.n_fam * (split.n_folds if method == "C_supp" else 1)
                        pick = rng.choice(len(test_idx), size=min(size, len(test_idx)), replace=False)
                        for i in pick:
                            prod = lp.threshold(phi_test[i])
                            ref = reference_sup(a_k, f_k, phi_k, phi_test[i], lp.tau)
                            s = a[vfam == split.test[test_idx[i]], y]  # candidate label y for every variant
                            fin = np.isfinite(prod) and np.isfinite(ref)
                            rows.append({"seed": seed, "scenario": scen, "beta": beta, "predictor": pred,
                                         "method": method, "label": y, "production": prod, "reference": ref,
                                         "diff": prod - ref if fin else (0.0 if prod == ref else np.inf if prod > ref else -np.inf),
                                         "n_added": int(((s > ref) & (s <= prod + THRESHOLD_TOL)).sum()),
                                         "n_lost": int(((s <= ref + THRESHOLD_TOL) & (s > prod + THRESHOLD_TOL)).sum()),
                                         "n_at_tol": int((np.abs(s - prod) <= THRESHOLD_TOL).sum()) if np.isfinite(prod) else 0})
            print(f"seed {seed} {scen} {beta} done ({len(rows)} thresholds)", flush=True)
    df = pd.DataFrame(rows)
    out = results_dir() / "audit"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "thresholds.csv", index=False)
    fin = np.isfinite(df["production"]) & np.isfinite(df["reference"])
    summ = pd.DataFrame([{
        "n_thresholds": len(df), "n_both_infinite": int((~np.isfinite(df["production"]) & ~np.isfinite(df["reference"])).sum()),
        "n_inf_disagree": int((np.isfinite(df["production"]) != np.isfinite(df["reference"])).sum()),
        "n_prod_inf_ref_finite": int((~np.isfinite(df["production"]) & np.isfinite(df["reference"])).sum()),
        "min_diff": df.loc[fin, "diff"].min(), "max_diff": df.loc[fin, "diff"].max(),
        "n_diff_above_1e-6": int((df.loc[fin, "diff"] > 1e-6).sum()),
        "n_diff_below_minus_1e-9": int((df.loc[fin, "diff"] < -1e-9).sum()),
        "n_added": int(df["n_added"].sum()), "n_lost": int(df["n_lost"].sum()),
        "n_at_tol": int(df["n_at_tol"].sum())}])
    summ.to_csv(out / "summary.csv", index=False)
    print(summ.T.to_string())


if __name__ == "__main__":
    main()
