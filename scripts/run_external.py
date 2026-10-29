"""External temporal evaluation, exactly as pre-registered in docs/preregistration_external.md.

For each seed and benchmark setting, the benchmark split defines training,
calibration and test families. The base model is fitted on the benchmark variants
of the training families and calibration uses the benchmark variants of the
calibration families; the test set is the external (post-snapshot) ClinVar variants
of the test families' proteins. Only ESM-1b. Every method of run_methods is
evaluated. Writes results/raw/<tag>/seed<k>/ with the same files as
run_experiments.py, so scripts/summarize.py works unchanged.

Usage:
  python scripts/run_external.py --base realistic --tag external   (primary)
  python scripts/run_external.py --base main --tag external_main    (secondary)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from protcover.base_model import clip_bounds, nonconformity, within_protein_z
from protcover.config import load_config, processed_dir, results_dir
from protcover.metrics import evaluate, per_family_counts, random_risk_coverage, risk_coverage
from protcover.pipeline import RISK_GRID, Split, risk_scores, run_methods

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import SCENARIO_CODE, inner_tuning, load_context, make_splits, weight_diagnostics  # noqa: E402

PRED = "ESM1b"


def load_external(ctx):
    """Pre-registered inclusion: proteins of at most 1,022 residues; drop rows that may
    predate the benchmark snapshot or whose change was in ClinVar before it."""
    e = pd.read_parquet(processed_dir(ctx.cfg) / "external_temporal.parquet")
    e = e[~e["over_1022"] & ~e["in_2023_07_release"] & ~e["aa_change_in_2023_06_release"]]
    fam = ctx.fam.set_index("protein_id")[ctx.fam_col]
    missing = ~e["protein_id"].isin(fam.index)
    if missing.any():
        raise ValueError(f"{missing.sum()} external variants in proteins without a family")
    return pd.DataFrame({"protein_id": e["protein_id"].to_numpy(), "label": e["label"].to_numpy().astype(int),
                         PRED: e["esm1b_oriented"].to_numpy(float),
                         "family": fam.loc[e["protein_id"]].to_numpy()})


def base_probabilities(ctx, train_fams, ext_test, within):
    """p_hat(pathogenic) for benchmark variants and external test variants.

    Fitted on the benchmark variants of the training families. With `within`, the second
    feature standardizes a score within its protein: benchmark variants over their
    protein's benchmark scores (as in the benchmark runs), external variants over the
    same protein's benchmark scores (pre-registered secondary analysis)."""
    v = ctx.variants
    tr = np.isin(ctx.vfam, train_fams)
    bounds = clip_bounds(v[PRED].to_numpy(), tr)
    s_b = np.clip(v[PRED].to_numpy(float), *bounds)
    s_e = np.clip(ext_test[PRED].to_numpy(float), *bounds)
    if within:
        z_b = within_protein_z(v["protein_id"].to_numpy(), s_b)
        g = pd.Series(s_b).groupby(v["protein_id"].to_numpy())
        mu, sd = g.mean(), g.std()
        mu_e, sd_e = mu.loc[ext_test["protein_id"]].to_numpy(), sd.loc[ext_test["protein_id"]].fillna(0).to_numpy()
        z_e = np.where(sd_e > 0, (s_e - mu_e) / np.where(sd_e > 0, sd_e, 1.0), 0.0)
        x_b, x_e = np.column_stack([s_b, z_b]), np.column_stack([s_e, z_e])
    else:
        x_b, x_e = s_b[:, None], s_e[:, None]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    model.fit(x_b[tr], v["label"].to_numpy()[tr])
    return model.predict_proba(x_b)[:, 1], model.predict_proba(x_e)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base", choices=["realistic", "main"], required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--scenarios", nargs="+", default=["S1", "S2", "S3", "S4"])
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ex = cfg["experiment"]
    if args.base == "realistic":
        ex["realistic"], ex["drop_descriptors"] = True, ["log_n_variants"]
    ex["skip_conditional"] = False
    ctx = load_context(cfg, ex["family_def"])
    ctx.preds = [PRED]
    ext = load_external(ctx)
    seeds = args.seeds if args.seeds is not None else ex["seeds"]

    t0 = time.time()
    for seed in seeds:
        rows, pf_rows, rc_rows, split_rows, tune_rows, param_rows = [], [], [], [], [], []
        tuned = {}
        for scen, beta, train, cal, test, desc, tag in make_splits(ctx, args.scenarios, seed):
            if tag not in tuned:
                tuned[tag], tab = inner_tuning(ctx, train, seed, tag)
                tune_rows.append(tab.assign(seed=seed))
                for (pred, alpha), p in tuned[tag].items():
                    for (method, cc), (m, tau, sc) in p.items():
                        param_rows.append({"train_tag": tag, "seed": seed, "predictor": pred, "alpha": alpha,
                                           "method": method, "cc": cc, "m": m, "tau": tau, "scale": sc})
            # Benchmark variants of training and calibration families, external variants of test families.
            keep = np.isin(ctx.vfam, np.r_[train, cal])
            ext_t = ext[np.isin(ext["family"].to_numpy(), test)]
            p_b, p_e = base_probabilities(ctx, train, ext_t, within=args.base == "main")
            vfam = np.r_[ctx.vfam[keep], ext_t["family"].to_numpy()]
            label = np.r_[ctx.label[keep], ext_t["label"].to_numpy()]
            a = nonconformity(np.r_[p_b[keep], p_e])
            b100 = 0 if beta != beta else int(round(beta * 100))
            split = Split(desc, train, cal, test, ex, np.random.default_rng([seed, SCENARIO_CODE[scen], b100, 5]))
            te = np.isin(vfam, test)
            split_rows.append({"scenario": scen, "beta": beta, "seed": seed, "n_test_fam": len(test),
                               "n_test_fam_with_external": int(len(np.unique(vfam[te]))), "n_test_var": int(te.sum()),
                               "test_frac_path": float(label[te].mean()), **weight_diagnostics(split, ex)})
            for alpha in ex["alphas"]:
                rng = np.random.default_rng([seed, SCENARIO_CODE[scen], b100, int(alpha * 1000), 0])
                results = run_methods(split, a, label, vfam, alpha, ex, tuned[tag][(PRED, alpha)], PRED, rng,
                                      ablations=False)
                ids = {"scenario": scen, "beta": beta, "seed": seed, "predictor": PRED, "alpha": alpha}
                for name, cc, sets, ab in results:
                    m = evaluate(sets, label[te], vfam[te], alpha, ex["min_family_variants"], ab)
                    rows.append({**ids, "method": name, "cc": cc, **m})
                    pf = per_family_counts(sets, label[te], vfam[te]).reset_index()
                    pf["family"] = ctx.family_names[pf["family"].to_numpy()]
                    pf_rows.append(pf.assign(**ids, method=name, cc=cc))
                    if name == "P_w":
                        scale = tuned[tag][(PRED, alpha)][("P_full", cc)][2]
                        scores = {k: s.loc[np.unique(vfam[te])] for k, s in risk_scores(split, ex, scale).items()}
                        curves = {k: risk_coverage(sets, label[te], vfam[te], s, RISK_GRID) for k, s in scores.items()}
                        rrng = np.random.default_rng([seed, SCENARIO_CODE[scen], b100, int(alpha * 1000), 0, int(cc), 99])
                        curves["random"] = random_risk_coverage(sets, label[te], vfam[te], RISK_GRID, rrng,
                                                                ex["n_random_orders"])
                        for sname, (curve, aurc) in curves.items():
                            rc_rows.append({**ids, "cc": cc, "score": sname, "aurc": aurc,
                                            **{f"r{g:.2f}": v for g, v in zip(RISK_GRID, curve)}})
        out = results_dir() / "raw" / args.tag / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
        pd.concat(pf_rows).to_parquet(out / "perfamily.parquet", index=False)
        pd.DataFrame(rc_rows).to_csv(out / "risk_coverage.csv", index=False)
        pd.DataFrame(split_rows).to_csv(out / "splits.csv", index=False)
        pd.concat(tune_rows).to_csv(out / "gate_tuning.csv", index=False)
        pd.DataFrame(param_rows).to_csv(out / "gate_params.csv", index=False)
        print(f"seed {seed} done ({time.time() - t0:.0f}s), wrote {out}", flush=True)


if __name__ == "__main__":
    main()
