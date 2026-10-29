"""Base-model sanity check on the S1 splits: test AUC, Brier score, coefficients.

Writes results/base_model.csv (one row per seed and predictor).
Usage: python scripts/base_model_diagnostics.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from protcover.base_model import features, fit
from protcover.config import load_config, predictor_names, results_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import load_context, make_splits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    ex = cfg["experiment"]
    ctx = load_context(cfg, ex["family_def"])
    variants, vfam, y = ctx.variants, ctx.vfam, ctx.label
    rows = []
    for seed in ex["seeds"]:
        for scen, beta, train, cal, test, _, _ in make_splits(ctx, ["S1"], seed):
            tr, te = np.isin(vfam, train), np.isin(vfam, test)
            for pred in predictor_names(cfg):
                model, bounds = fit(variants, pred, tr)
                p = model.predict_proba(features(variants, pred, bounds)[te])[:, 1]
                d = pd.DataFrame({"f": vfam[te], "y": y[te], "p": p})
                fam_auc = d.groupby("f").filter(lambda g: g["y"].nunique() == 2).groupby("f").apply(
                    lambda g: roc_auc_score(g["y"], g["p"]), include_groups=False)
                coef = model[-1].coef_[0]
                rows.append({"seed": seed, "predictor": pred, "test_auc": roc_auc_score(y[te], p),
                             "test_auc_raw_score": roc_auc_score(y[te], variants[pred].to_numpy()[te]),
                             "test_family_mean_auc": fam_auc.mean(), "test_brier": brier_score_loss(y[te], p),
                             "coef_raw": coef[0], "coef_within_z": coef[1]})
    out = pd.DataFrame(rows)
    out.to_csv(results_dir() / "base_model.csv", index=False)
    print(out.groupby("predictor").mean(numeric_only=True).drop(columns="seed").round(3))


if __name__ == "__main__":
    main()
