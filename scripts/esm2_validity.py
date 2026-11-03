"""Pre-registered validity check of the ESM2 scores (docs/preregistration_unseen_esm2.md), run
before the ESM2 evaluation: the per-protein mean AUC of the oriented ESM2 scores on the benchmark
variants that train and calibrate the new-protein evaluation (development data). A value at or
below 0.5 would mean an orientation or position bug. ESM-1b (Brandes et al.) on the same variants
is printed for reference.

Writes results/unseen_esm2/validity.json.

Usage: python scripts/esm2_validity.py
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from protcover.config import load_config, results_dir
from run_unseen import load_unseen_context


def mean_auc(v, col):
    aucs = [roc_auc_score(g["label"], g[col]) for _, g in v.groupby("protein_id") if g["label"].nunique() == 2]
    return float(np.mean(aucs)), len(aucs)


def main():
    cfg = load_config()
    out = {}
    for pred in ("ESM2_150M", "ESM1b"):
        ctx = load_unseen_context(cfg, pred=pred)
        v = ctx.variants[np.isin(ctx.vfam, ctx.bench)]
        out[pred] = dict(zip(("mean_auc", "n_proteins"), mean_auc(v, pred)), n_variants=len(v))
        print(pred, out[pred])
    out["passed"] = out["ESM2_150M"]["mean_auc"] > 0.5
    d = results_dir() / "unseen_esm2"
    d.mkdir(parents=True, exist_ok=True)
    (d / "validity.json").write_text(json.dumps(out, indent=2) + "\n")
    if not out["passed"]:
        raise SystemExit("validity check not passed (mean AUC at most 0.5): fix the scores before the evaluation")


if __name__ == "__main__":
    main()
