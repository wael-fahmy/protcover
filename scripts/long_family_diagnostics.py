"""Descriptive diagnostics of the new families with a protein longer than any benchmark protein
(exploratory; no method is rerun): how the density ratio and the support gate treat them.

From the label-free descriptors, recomputes each seed's cross-fitted ratio exactly as
scripts/run_unseen.py does, and reads the gate's abstentions from that run's per-family outputs.

Writes results/unseen/long_families.csv (one row per seed, class-conditional versions at alpha 0.1).

Usage: python scripts/long_family_diagnostics.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import split_descriptors  # noqa: E402
from run_unseen import load_unseen_context  # noqa: E402

from protcover.config import load_config, results_dir  # noqa: E402
from protcover.density_ratio import _neg_log_loss  # noqa: E402
from protcover.pipeline import MSA_COLS, Split, descriptor_columns  # noqa: E402
from protcover.splits import draw_train  # noqa: E402


def abstained(pf):
    """Families on which P_gate abstains: every set is {0, 1}, and not already so without the gate."""
    gate = pf[pf["method"] == "P_gate"].set_index("family")
    base = pf[pf["method"] == "B3_pool"].set_index("family").reindex(gate.index)
    return gate.index[(gate["full"] == gate["n"]) & (base["full"] < base["n"])]


def ratio_coefficients(desc, split, cols):
    """Standardized coefficients of the ratio's logistic fit (all target families, the pipeline's
    penalty selection): positive means more typical of the target families."""
    x = np.vstack([desc.loc[split.cal_fit, cols].to_numpy(), desc.loc[split.test, cols].to_numpy()])
    y = np.r_[np.zeros(len(split.cal_fit)), np.ones(len(split.test))]
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, cv=5, scoring=_neg_log_loss, max_iter=2000,
                                                               random_state=0)).fit(x, y)
    return dict(zip(cols, clf[-1].coef_[0]))


def main():
    cfg = load_config()
    ex = cfg["experiment"]
    ex["realistic"], ex["no_msa"] = True, True
    ex["drop_descriptors"] = ["log_n_variants", *MSA_COLS]
    ctx = load_unseen_context(cfg)
    names = ctx.family_names
    rows = []
    for seed in ex["seeds"]:
        rng = np.random.default_rng([seed, 12])  # as scripts/run_unseen.py
        train = draw_train(ctx.bench, ex["frac_train"], rng)
        cal = np.setdiff1d(ctx.bench, train)
        desc = split_descriptors(ctx, train, f"unseen_seed{seed}")
        split = Split(desc, train, cal, ctx.test, ex, np.random.default_rng([seed, 21, 0, 5]))
        _, w_test = split.weights("main", ex["clip"])
        w = pd.Series(w_test, index=split.test)
        long_ = np.setdiff1d(split.test, ctx.in_len)
        rest = ctx.in_len
        rank = w.rank(pct=True)
        pf = pd.read_parquet(results_dir() / "raw" / "unseen" / f"seed{seed}" / "perfamily.parquet")
        pf = pf[(pf["scenario"] == "U") & np.isclose(pf["alpha"], 0.1) & (pf["cc"] == True)]
        ab = set(abstained(pf))
        ab_long = np.isin(names[long_], list(ab))
        ab_rest = np.isin(names[rest], list(ab))
        # The gate's own statistic: the kernel ESS at the seed's tuned scale; low means flagged first.
        gt = pd.read_csv(results_dir() / "raw" / "unseen" / f"seed{seed}" / "gate_tuning.csv")
        gt = gt[(gt["method"] == "P_gate") & (gt["cc"] == True) & np.isclose(gt["alpha"], 0.1)]
        sc = float(gt.sort_values("aurc").iloc[0]["scale"])
        e_rank = split.ess_full[sc].rank(pct=True)
        cols = [c for c in descriptor_columns("main", None) if c not in split.drop]
        coef = ratio_coefficients(desc, split, cols)
        rows.append({"seed": seed, "n_long": len(long_), "n_rest": len(rest),
                     "gate_scale": sc, "log_length_cal_mean": desc.loc[cal, "log_length"].mean(),
                     "log_length_test_mean": desc.loc[split.test, "log_length"].mean(),
                     "log_length_long_mean": desc.loc[long_, "log_length"].mean(),
                     **{f"ratio_coef_{c}": v for c, v in coef.items()},
                     "ess_long_rank_median": e_rank.loc[long_].median(),
                     "frac_long_lowest_decile_ess": (e_rank.loc[long_] <= 0.1).mean(),
                     "frac_long_at_lower_clip": (w.loc[long_] <= 1.0 / ex["clip"] + 1e-9).mean(),
                     "max_log_length_cal": desc.loc[cal, "log_length"].max(),
                     "w_long_mean": w.loc[long_].mean(), "w_rest_mean": w.loc[rest].mean(),
                     "w_long_median": w.loc[long_].median(), "w_rest_median": w.loc[rest].median(),
                     "w_long_rank_median": rank.loc[long_].median(),
                     "frac_long_clipped": (w.loc[long_] >= ex["clip"] - 1e-9).mean(),
                     "frac_long_above_rest_q90": (w.loc[long_] > w.loc[rest].quantile(0.9)).mean(),
                     "gate_abstain_long": ab_long.mean(), "gate_abstain_rest": ab_rest.mean()})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(results_dir() / "unseen" / "long_families.csv", index=False)
    print(out.mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
