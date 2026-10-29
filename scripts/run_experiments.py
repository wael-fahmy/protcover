"""Run every method on family-level splits of the clinical benchmark.

Writes, under results/raw/<tag>/seed<k>/:
  metrics.csv         one row per (scenario, beta, seed, predictor, alpha, method, cc)
  perfamily.parquet   per-test-family counts for every non-ablation row of metrics.csv
  risk_coverage.csv   risk-coverage curves of abstention rules on ProtCOVER (i)+(ii) sets
  splits.csv          split sizes, support descriptors and density-ratio diagnostics
  gate_tuning.csv     inner-split grid used to choose the gate parameters
  gate_params.csv     chosen gate parameters

Usage:
  python scripts/run_experiments.py --scenarios S1 S2 S3 S4 --tag main
"""
import argparse
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from protcover.base_model import fit_predict, nonconformity
from protcover.config import ROOT, load_config, predictor_names, processed_dir, results_dir
from protcover.data import load_proteins, load_variants
from protcover.density_ratio import ess
from protcover.descriptors import emb_dist_to_ref, embedding_regions, family_descriptors, load_embeddings, support_z
from protcover.families import max_identity, orphan_flags
from protcover.metrics import evaluate, per_family_counts, random_risk_coverage, risk_coverage
from protcover.pipeline import RISK_GRID, Split, risk_scores, run_methods, tune_gate
from protcover.splits import draw_train, orphan_split, region_split, shifted_cal_test, split_sizes

THREADS = os.cpu_count() or 4
SCENARIO_CODE = {"S1": 1, "S2": 2, "S3": 3, "S4": 4}


def load_context(cfg, fam_col):
    variants, proteins = load_variants(cfg), load_proteins(cfg)
    fam = pd.read_parquet(processed_dir(cfg) / "families.parquet")[["protein_id", fam_col]]
    # Integer family codes keep set operations fast; names are restored on output.
    names = np.sort(fam[fam_col].unique())
    fam[fam_col] = np.searchsorted(names, fam[fam_col].to_numpy())
    variants = variants.merge(fam, on="protein_id")
    hom = cfg["families"]["homology"]
    hits = pd.read_parquet(processed_dir(cfg) / "homology.parquet")
    emb = load_embeddings(ROOT / cfg["experiment"]["embeddings"])
    return SimpleNamespace(
        cfg=cfg, variants=variants, proteins=proteins, fam=fam, fam_col=fam_col, family_names=names,
        families=np.sort(fam[fam_col].unique()), preds=predictor_names(cfg),
        orphan=orphan_flags(fam, fam_col, hits, hom["evalue"], hom["min_id"]),
        emb=emb, region=embedding_regions(emb, fam, fam_col, **cfg["experiment"]["s4"]),
        vfam=variants[fam_col].to_numpy(), label=variants["label"].to_numpy())


def cached_max_id(ctx, train, tag):
    cfg = ctx.cfg
    path = processed_dir(cfg) / "cache" / f"maxid_{ctx.fam_col}_{tag}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_parquet(path)
    in_train = ctx.proteins["protein_id"].isin(ctx.fam.loc[ctx.fam[ctx.fam_col].isin(train), "protein_id"])
    hom = cfg["families"]["homology"]
    m = max_identity(ctx.proteins[~in_train], ctx.proteins[in_train], cfg["families"]["search_sensitivity"], hom["evalue"], THREADS)
    m.to_parquet(path, index=False)
    return m


def split_descriptors(ctx, train, tag):
    """Family descriptors with support measured relative to `train` families."""
    in_train = ctx.fam[ctx.fam_col].isin(train)
    ed = emb_dist_to_ref(ctx.emb, ctx.fam.loc[~in_train, "protein_id"], ctx.fam.loc[in_train, "protein_id"])
    return family_descriptors(ctx.variants, ctx.proteins, ctx.fam, ctx.fam_col, ctx.preds,
                              cached_max_id(ctx, train, tag), ed)


def make_splits(ctx, scenarios, seed):
    """Yield (scenario, beta, train, cal, test, desc, train_tag) for one seed."""
    ex = ctx.cfg["experiment"]
    n_train, n_cal, n_test = split_sizes(len(ctx.families), ex["frac_train"], ex["frac_cal"])
    if {"S1", "S2"} & set(scenarios):
        rng = np.random.default_rng([seed, 12])
        train = draw_train(ctx.families, ex["frac_train"], rng)
        tag = f"S12_seed{seed}"
        desc = split_descriptors(ctx, train, tag)
        nontrain = np.setdiff1d(ctx.families, train)
        z = support_z(desc.loc[nontrain], ex["support_cols"]).to_numpy()
        u = rng.random(len(nontrain))
        betas = ([0.0] if "S1" in scenarios else []) + ([b for b in ex["betas"] if b != 0] if "S2" in scenarios else [])
        for beta in betas:
            cal, test = shifted_cal_test(nontrain, z, n_test, u, beta)
            yield ("S1" if beta == 0 else "S2"), beta, train, cal, test, desc, tag
    if "S3" in scenarios:
        rng = np.random.default_rng([seed, 3])
        train, cal, test = orphan_split(ctx.families, ctx.orphan, ex["frac_train"], ex["frac_cal"], rng)
        tag = f"S3_seed{seed}"
        yield "S3", np.nan, train, cal, test, split_descriptors(ctx, train, tag), tag
    if "S4" in scenarios:
        rng = np.random.default_rng([seed, 4])
        train, cal, test = region_split(ctx.families, ctx.region, ex["frac_train"], ex["frac_cal"], rng)
        tag = f"S4_seed{seed}"
        yield "S4", np.nan, train, cal, test, split_descriptors(ctx, train, tag), tag


def inner_tuning(ctx, train, seed, tag):
    """Gate parameters per (predictor, alpha) from a shifted inner split of the train families."""
    ex = ctx.cfg["experiment"]
    inn = ex["inner"]
    rng = np.random.default_rng([seed, 99, len(tag)])
    itrain = draw_train(train, inn["frac_train"], rng)
    desc = split_descriptors(ctx, itrain, f"inner_{tag}")
    rest = np.setdiff1d(train, itrain)
    z = support_z(desc.loc[rest], ex["support_cols"]).to_numpy()
    n_test = int(round(inn["frac_test"] * len(train)))
    ical, itest = shifted_cal_test(rest, z, n_test, rng.random(len(rest)), inn["beta"])
    split = Split(desc, itrain, ical, itest, ex, rng)
    params, tables = {}, []
    tr = np.isin(ctx.vfam, itrain)
    for pred in ctx.preds:
        a = nonconformity(fit_predict(ctx.variants, pred, tr))
        for alpha in ex["alphas"]:
            p, tab = tune_gate(split, a, ctx.label, ctx.vfam, alpha, ex)
            params[(pred, alpha)] = p
            tables.append(tab.assign(train_tag=tag, predictor=pred, alpha=alpha))
    return params, pd.concat(tables)


def weight_diagnostics(split, ex):
    w_cal, w_test = split.weights("main", ex["clip"])
    pi = split.test_mass(w_cal, w_test)
    return {"w_cal_ess_frac": float(np.mean(ess(w_cal)) / w_cal.shape[1]), "w_cal_max": float(w_cal.max()),
            "w_test_mean": float(w_test.mean()), "w_test_max": float(w_test.max()),
            "frac_w_clipped": float(np.mean(np.r_[w_cal.ravel(), w_test] >= ex["clip"] - 1e-9)),
            "test_mass_mean": float(pi.mean()), "test_mass_max": float(pi.max()),
            **{f"ess_conf_median_s{sc:g}": float(e.median()) for sc, e in split.ess_conf.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--scenarios", nargs="+", default=["S1", "S2", "S3", "S4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--family-def", default=None)
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ex = cfg["experiment"]
    ctx = load_context(cfg, args.family_def or ex["family_def"])
    seeds = args.seeds if args.seeds is not None else ex["seeds"]
    vfam, label = ctx.vfam, ctx.label

    t0 = time.time()
    for seed in seeds:
        rows, pf_rows, rc_rows, split_rows, tune_rows, param_rows = [], [], [], [], [], []
        fits, tuned = {}, {}
        for scen, beta, train, cal, test, desc, tag in make_splits(ctx, args.scenarios, seed):
            if tag not in tuned:
                tuned[tag], tab = inner_tuning(ctx, train, seed, tag)
                tune_rows.append(tab.assign(seed=seed))
                for (pred, alpha), p in tuned[tag].items():
                    for (method, cc), (m, tau, sc) in p.items():
                        param_rows.append({"train_tag": tag, "seed": seed, "predictor": pred, "alpha": alpha,
                                           "method": method, "cc": cc, "m": m, "tau": tau, "scale": sc})
            b100 = 0 if beta != beta else int(round(beta * 100))
            split = Split(desc, train, cal, test, ex, np.random.default_rng([seed, SCENARIO_CODE[scen], b100, 5]))
            tr, ca, te = np.isin(vfam, train), np.isin(vfam, cal), np.isin(vfam, test)
            split_rows.append({"scenario": scen, "beta": beta, "seed": seed,
                               "n_train_fam": len(train), "n_cal_fam": len(cal), "n_cal_conf_fam": len(split.cal_conf),
                               "n_test_fam": len(test), "n_train_var": tr.sum(), "n_cal_var": ca.sum(), "n_test_var": te.sum(),
                               "cal_frac_path": label[ca].mean(), "test_frac_path": label[te].mean(),
                               **{f"cal_mean_{c}": desc.loc[cal, c].mean() for c in ["max_id_to_train", "log_neff", "emb_dist_to_train", "log_n_variants"]},
                               **{f"test_mean_{c}": desc.loc[test, c].mean() for c in ["max_id_to_train", "log_neff", "emb_dist_to_train", "log_n_variants"]},
                               **weight_diagnostics(split, ex)})
            for pred in ctx.preds:
                key = (pred, tag)
                if key not in fits:
                    fits[key] = nonconformity(fit_predict(ctx.variants, pred, tr))
                a = fits[key]
                for alpha in ex["alphas"]:
                    rng = np.random.default_rng([seed, SCENARIO_CODE[scen], b100, int(alpha * 1000), ctx.preds.index(pred)])
                    results = run_methods(split, a, label, vfam, alpha, ex, tuned[tag][(pred, alpha)], pred, rng,
                                          ablations=not args.no_ablations)
                    ids = {"scenario": scen, "beta": beta, "seed": seed, "predictor": pred, "alpha": alpha}
                    for name, cc, sets, ab in results:
                        m = evaluate(sets, label[te], vfam[te], alpha, ex["min_family_variants"], ab)
                        rows.append({**ids, "method": name, "cc": cc, **m})
                        if "[" not in name:
                            pf = per_family_counts(sets, label[te], vfam[te]).reset_index()
                            pf["family"] = ctx.family_names[pf["family"].to_numpy()]
                            pf_rows.append(pf.assign(**ids, method=name, cc=cc))
                        if name == "P_w":
                            scale = tuned[tag][(pred, alpha)][("P_full", cc)][2]
                            curves = {sname: risk_coverage(sets, label[te], vfam[te], s, RISK_GRID)
                                      for sname, s in risk_scores(split, ex, scale).items()}
                            rrng = np.random.default_rng([seed, SCENARIO_CODE[scen], b100, int(alpha * 1000),
                                                          ctx.preds.index(pred), int(cc), 99])
                            curves["random"] = random_risk_coverage(sets, label[te], vfam[te], RISK_GRID, rrng,
                                                                    ex["n_random_orders"])
                            for sname, (curve, aurc) in curves.items():
                                rc_rows.append({**ids, "cc": cc, "score": sname, "aurc": aurc,
                                                **{f"r{g:.2f}": v for g, v in zip(RISK_GRID, curve)}})
        # One directory per seed, so seeds can run as parallel processes.
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
