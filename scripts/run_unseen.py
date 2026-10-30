"""Pre-registered evaluation on proteins outside the benchmark (docs/preregistration_unseen.md).

Calibrates on the ProteinGym clinical benchmark and tests on ClinVar variants in human proteins
that are not in it (scripts/build_unseen.py). For each seed, the training families are the S1
training draw of the benchmark, every other benchmark family is a calibration family, and the test
families are all families of new proteins. Every protein is scored by the ESM-1b scores of Brandes
et al. (benchmark variants without a Brandes score are dropped), the base model uses the raw score
only, and the descriptors are alignment-free (new proteins have no alignment depth).

Writes, under results/raw/<tag>/seed<k>/, the same files as run_experiments.py (no risk-coverage
curves), with scenario "U" (all test families), "Ustrict" (test families without any MMseqs2
hit at E <= 1e-3 to a benchmark protein) and "Ulen" (exploratory: test families whose longest
protein is no longer than the longest benchmark protein); the same sets, evaluated on those
families only.

--dry-run tests the code path before the pre-registration is frozen: 300 random test families,
labels randomly permuted, one seed, alpha 0.1; it prints only whether the run completed.

Usage: python scripts/run_unseen.py --seeds 0 1 --tag unseen
"""
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import inner_tuning, split_descriptors, weight_diagnostics  # noqa: E402

from protcover.base_model import fit_predict, nonconformity  # noqa: E402
from protcover.config import ROOT, load_config, processed_dir, results_dir  # noqa: E402
from protcover.descriptors import load_embeddings  # noqa: E402
from protcover.metrics import evaluate, per_family_counts  # noqa: E402
from protcover.pipeline import MSA_COLS, Split, run_methods  # noqa: E402
from protcover.splits import draw_train  # noqa: E402

FAM_COL = "fam_unseen"
PRED = "ESM1b"


def load_unseen_context(cfg, dry_run=False):
    """Benchmark and new proteins in one table, families coded as integers like load_context."""
    pdir = processed_dir(cfg)
    ex = cfg["experiment"]
    # Benchmark variants, rescored with Brandes et al.'s ESM-1b (same score function as the test set).
    bv = pd.read_parquet(pdir / "variants.parquet")
    be = pd.read_parquet(pdir / "unseen_benchmark_esm1b.parquet")
    bv = bv.merge(be[["protein_id", "variant", "scored", "esm1b_brandes_oriented"]], on=["protein_id", "variant"], how="left")
    n_bench = len(bv)
    bv = bv[bv["scored"].fillna(False).astype(bool)]
    bv = bv.assign(**{PRED: bv["esm1b_brandes_oriented"].to_numpy()})[["protein_id", "label", PRED]]
    uv = pd.read_parquet(pdir / "unseen_variants.parquet")
    uv = uv.assign(**{PRED: uv["esm1b_oriented"].to_numpy()})[["protein_id", "label", PRED]]
    # Proteins: benchmark (with alignment depth, unused here) and new (none).
    bp = pd.read_parquet(pdir / "proteins.parquet")[["protein_id", "sequence", "length", "msa_neff", "msa_neff_L"]]
    up = pd.read_parquet(pdir / "unseen_proteins.parquet")
    assert not set(bp["protein_id"]) & set(up["protein_id"]), "protein ids overlap"
    proteins = pd.concat([bp, up[["protein_id", "sequence", "length"]].assign(msa_neff=np.nan, msa_neff_L=np.nan)],
                         ignore_index=True)
    bf = pd.read_parquet(pdir / "families.parquet")[["protein_id", ex["family_def"]]].rename(columns={ex["family_def"]: FAM_COL})
    bf[FAM_COL] = "B:" + bf[FAM_COL].astype(str)
    uf = up[["protein_id", "family"]].rename(columns={"family": FAM_COL})
    uf[FAM_COL] = "U:" + uf[FAM_COL].astype(str)
    fams = pd.read_parquet(pdir / "unseen_families.parquet")
    strict_names = set("U:" + fams.loc[~fams["has_benchmark_hit_e1e-3"].astype(bool), "family"].astype(str))
    if dry_run:
        rng = np.random.default_rng(12345)
        keep = set(rng.choice(np.sort(uf[FAM_COL].unique()), 300, replace=False))
        uf = uf[uf[FAM_COL].isin(keep)]
        uv = uv[uv["protein_id"].isin(uf["protein_id"])].copy()
        uv["label"] = rng.permutation(uv["label"].to_numpy())  # no real label is used
        proteins = proteins[proteins["protein_id"].isin(set(bp["protein_id"]) | set(uf["protein_id"]))]
    fam = pd.concat([bf, uf], ignore_index=True)
    variants = pd.concat([bv, uv], ignore_index=True).merge(fam, on="protein_id")
    names = np.sort(fam[FAM_COL].unique())
    fam[FAM_COL] = np.searchsorted(names, fam[FAM_COL].to_numpy())
    variants[FAM_COL] = np.searchsorted(names, variants[FAM_COL].to_numpy())
    emb = pd.concat([load_embeddings(ROOT / ex["embeddings"]), load_embeddings(ROOT / "data" / "esm2_t12_35M_meanpool_unseen.npz")])
    bench = np.sort(fam.loc[fam["protein_id"].isin(bp["protein_id"]), FAM_COL].unique())
    test = np.sort(fam.loc[fam["protein_id"].isin(up["protein_id"]), FAM_COL].unique())
    strict = np.sort(np.searchsorted(names, np.array(sorted(strict_names & set(names)))))
    # Exploratory: test families whose longest protein is within the benchmark's length range.
    long_names = set("U:" + fams.loc[fams["max_length"] > bp["length"].max(), "family"].astype(str))
    in_len = np.setdiff1d(test, np.searchsorted(names, np.array(sorted(long_names & set(names)), dtype=object)))
    return SimpleNamespace(
        cfg=cfg, variants=variants, proteins=proteins, fam=fam, fam_col=FAM_COL, family_names=names,
        families=np.sort(fam[FAM_COL].unique()), preds=[PRED], emb=emb, bench=bench, test=test, strict=strict,
        in_len=in_len,
        vfam=variants[FAM_COL].to_numpy(), label=variants["label"].to_numpy(),
        n_bench_dropped=n_bench - len(bv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--tag", default="unseen")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ex = cfg["experiment"]
    # Deployment-realistic, alignment-free configuration (as --realistic --no-msa in run_experiments.py).
    ex["realistic"], ex["no_msa"] = True, True
    ex["drop_descriptors"] = ["log_n_variants", *MSA_COLS]
    ex["conditional"]["primary"] = [c for c in ex["conditional"]["primary"] if c not in MSA_COLS]
    ex["review_only"] = False
    if args.dry_run:
        ex["alphas"], args.tag = [0.1], f"{args.tag}_dryrun"
    ctx = load_unseen_context(cfg, args.dry_run)
    seeds = [0] if args.dry_run else (args.seeds if args.seeds is not None else ex["seeds"])
    vfam, label = ctx.vfam, ctx.label
    t0 = time.time()
    for seed in seeds:
        rng = np.random.default_rng([seed, 12])  # the benchmark's S1 training draw
        train = draw_train(ctx.bench, ex["frac_train"], rng)
        cal = np.setdiff1d(ctx.bench, train)
        test = ctx.test
        tag = f"unseen_seed{seed}" + ("_dry" if args.dry_run else "")
        desc = split_descriptors(ctx, train, tag)
        tuned, tune_tab = inner_tuning(ctx, train, seed, tag)
        split = Split(desc, train, cal, test, ex, np.random.default_rng([seed, 21, 0, 5]))
        tr, te = np.isin(vfam, train), np.isin(vfam, test)
        a = nonconformity(fit_predict(ctx.variants, PRED, tr, within=False))
        pops = (("U", np.ones(te.sum(), bool)), ("Ustrict", np.isin(vfam[te], ctx.strict)),
                ("Ulen", np.isin(vfam[te], ctx.in_len)))
        rows, pf_rows = [], []
        for alpha in ex["alphas"]:
            mrng = np.random.default_rng([seed, 21, 0, int(alpha * 1000), 0])
            for name, cc, sets, ab in run_methods(split, a, label, vfam, alpha, ex, tuned[(PRED, alpha)], PRED, mrng,
                                                  ablations=False):
                for scen, k in pops:
                    ids = {"scenario": scen, "beta": np.nan, "seed": seed, "predictor": PRED, "alpha": alpha}
                    m = evaluate(sets[k], label[te][k], vfam[te][k], alpha, ex["min_family_variants"],
                                 None if ab is None else ab[k])
                    rows.append({**ids, "method": name, "cc": cc, **m})
                    pf = per_family_counts(sets[k], label[te][k], vfam[te][k]).reset_index()
                    pf["family"] = ctx.family_names[pf["family"].to_numpy()]
                    pf_rows.append(pf.assign(**ids, method=name, cc=cc))
        out = results_dir() / "raw" / args.tag / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
        pd.concat(pf_rows).to_parquet(out / "perfamily.parquet", index=False)
        tr_v, te_v = tr.sum(), te.sum()
        pd.DataFrame([{"scenario": "U", "beta": np.nan, "seed": seed, "n_train_fam": len(train), "n_cal_fam": len(cal),
                       "n_cal_conf_fam": len(split.cal_conf), "n_test_fam": len(test), "n_test_fam_strict": len(ctx.strict),
                       "n_test_fam_in_len": len(ctx.in_len),
                       "n_train_var": tr_v, "n_test_var": te_v, "n_bench_var_dropped": ctx.n_bench_dropped,
                       **{f"cal_mean_{c}": desc.loc[cal, c].mean() for c in ["max_id_to_train", "emb_dist_to_train", "log_length"]},
                       **{f"test_mean_{c}": desc.loc[test, c].mean() for c in ["max_id_to_train", "emb_dist_to_train", "log_length"]},
                       **weight_diagnostics(split, ex)}]).to_csv(out / "splits.csv", index=False)
        tune_tab.assign(seed=seed).to_csv(out / "gate_tuning.csv", index=False)
        print(f"seed {seed} done ({time.time() - t0:.0f}s)" + (" [dry run: permuted labels, no metrics shown]" if args.dry_run else ""),
              flush=True)


if __name__ == "__main__":
    main()
