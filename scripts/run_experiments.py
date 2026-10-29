"""Run conformal methods on family-level splits of the clinical benchmark.

Writes, under results/raw/<tag>/:
  metrics.csv     one row per (scenario, beta, seed, predictor, alpha, method, cc)
  perfamily.parquet  per-test-family counts (coverage, set size) for every row of metrics.csv
  splits.csv      sizes and support descriptors of each split

Usage:
  python scripts/run_experiments.py --scenarios S1 S2 --tag checkpoint1
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from protcover.base_model import fit_predict, nonconformity
from protcover.conformal import conformal_sets, subsample_sets
from protcover.config import load_config, predictor_names, processed_dir, results_dir
from protcover.data import load_proteins, load_variants
from protcover.descriptors import family_descriptors, support_z
from protcover.families import max_identity, orphan_flags
from protcover.metrics import evaluate, per_family_counts
from protcover.splits import draw_train, orphan_split, shifted_cal_test, split_sizes

THREADS = os.cpu_count() or 4
SCENARIO_CODE = {"S1": 1, "S2": 2, "S3": 3}


def cached_max_id(cfg, proteins, fam, fam_col, train, tag):
    path = processed_dir(cfg) / "cache" / f"maxid_{fam_col}_{tag}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_parquet(path)
    in_train = fam[fam_col].isin(train).to_numpy()
    q = proteins[~proteins["protein_id"].isin(fam.loc[in_train, "protein_id"])]
    t = proteins[proteins["protein_id"].isin(fam.loc[in_train, "protein_id"])]
    hom = cfg["families"]["homology"]
    m = max_identity(q, t, cfg["families"]["search_sensitivity"], hom["evalue"], THREADS)
    m.to_parquet(path, index=False)
    return m


def methods(ex):
    """(name, cc, callable(a_cal, y_cal, fam_cal, a_test, alpha, rng))."""
    out = []
    for cc in (False, True):
        out += [
            ("B1_split", cc, lambda a, y, f, t, al, r, cc=cc: conformal_sets(a, y, f, t, al, "variant", cc)),
            ("B3_pool", cc, lambda a, y, f, t, al, r, cc=cc: conformal_sets(a, y, f, t, al, "family", cc)),
            ("B3_sub1", cc, lambda a, y, f, t, al, r, cc=cc: subsample_sets(a, y, f, t, al, 1, r, cc)),
            ("B3_subR", cc, lambda a, y, f, t, al, r, cc=cc: subsample_sets(a, y, f, t, al, ex["n_subsample"], r, cc)),
        ]
    return out


def make_splits(cfg, scenarios, seed, proteins, variants, fam, fam_col, families, orphan):
    """Yield (scenario, beta, train, cal, test, desc) for one seed."""
    ex = cfg["experiment"]
    preds = predictor_names(cfg)
    n_train, n_cal, n_test = split_sizes(len(families), ex["frac_train"], ex["frac_cal"])
    if {"S1", "S2"} & set(scenarios):
        rng = np.random.default_rng([seed, 12])
        train = draw_train(families, ex["frac_train"], rng)
        mid = cached_max_id(cfg, proteins, fam, fam_col, train, f"S12_seed{seed}")
        desc = family_descriptors(variants, proteins, fam, fam_col, preds, mid)
        nontrain = np.setdiff1d(families, train)
        z = support_z(desc.loc[nontrain], ex["support_cols"]).to_numpy()
        u = rng.random(len(nontrain))
        betas = ([0.0] if "S1" in scenarios else []) + ([b for b in ex["betas"] if b != 0] if "S2" in scenarios else [])
        for beta in betas:
            cal, test = shifted_cal_test(nontrain, z, n_test, u, beta)
            yield ("S1" if beta == 0 else "S2"), beta, train, cal, test, desc
    if "S3" in scenarios:
        rng = np.random.default_rng([seed, 3])
        train, cal, test = orphan_split(families, orphan, ex["frac_train"], ex["frac_cal"], rng)
        mid = cached_max_id(cfg, proteins, fam, fam_col, train, f"S3_seed{seed}")
        desc = family_descriptors(variants, proteins, fam, fam_col, preds, mid)
        yield "S3", np.nan, train, cal, test, desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--scenarios", nargs="+", default=["S1", "S2", "S3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--family-def", default=None)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ex = cfg["experiment"]
    fam_col = args.family_def or ex["family_def"]
    seeds = args.seeds if args.seeds is not None else ex["seeds"]
    preds = predictor_names(cfg)

    variants, proteins = load_variants(cfg), load_proteins(cfg)
    fam = pd.read_parquet(processed_dir(cfg) / "families.parquet")[["protein_id", fam_col]]
    variants = variants.merge(fam, on="protein_id")
    families = np.sort(fam[fam_col].unique())
    hom = cfg["families"]["homology"]
    hits = pd.read_parquet(processed_dir(cfg) / "homology.parquet")
    orphan = orphan_flags(fam, fam_col, hits, hom["evalue"], hom["min_id"])
    vfam = variants[fam_col].to_numpy()
    label = variants["label"].to_numpy()

    rows, pf_rows, split_rows = [], [], []
    t0 = time.time()
    for seed in seeds:
        fits = {}
        for scen, beta, train, cal, test, desc in make_splits(cfg, args.scenarios, seed, proteins, variants, fam, fam_col, families, orphan):
            tr, ca, te = np.isin(vfam, train), np.isin(vfam, cal), np.isin(vfam, test)
            split_rows.append({"scenario": scen, "beta": beta, "seed": seed,
                               "n_train_fam": len(train), "n_cal_fam": len(cal), "n_test_fam": len(test),
                               "n_train_var": tr.sum(), "n_cal_var": ca.sum(), "n_test_var": te.sum(),
                               "cal_frac_path": label[ca].mean(), "test_frac_path": label[te].mean(),
                               **{f"cal_mean_{c}": desc.loc[cal, c].mean() for c in ex["support_cols"] + ["log_n_variants"]},
                               **{f"test_mean_{c}": desc.loc[test, c].mean() for c in ex["support_cols"] + ["log_n_variants"]}})
            for pred in preds:
                key = (pred, tuple(train))
                if key not in fits:
                    fits[key] = nonconformity(fit_predict(variants, pred, tr))
                a = fits[key]
                a_cal = a[ca, label[ca]]
                for alpha in ex["alphas"]:
                    for mi, (name, cc, fn) in enumerate(methods(ex)):
                        rng = np.random.default_rng([seed, SCENARIO_CODE[scen], int(round((beta if beta == beta else 0) * 100)), mi, int(alpha * 1000)])
                        sets = fn(a_cal, label[ca], vfam[ca], a[te], alpha, rng)
                        m = evaluate(sets, label[te], vfam[te], alpha, ex["min_family_variants"])
                        rows.append({"scenario": scen, "beta": beta, "seed": seed, "predictor": pred,
                                     "alpha": alpha, "method": name, "cc": cc, **m})
                        p = per_family_counts(sets, label[te], vfam[te]).reset_index()
                        pf_rows.append(p.assign(scenario=scen, beta=beta, seed=seed, predictor=pred,
                                                alpha=alpha, method=name, cc=cc))
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    out = results_dir() / "raw" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.concat(pf_rows).to_parquet(out / "perfamily.parquet", index=False)
    pd.DataFrame(split_rows).to_csv(out / "splits.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
