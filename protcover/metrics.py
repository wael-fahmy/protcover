"""Evaluation metrics for binary-label conformal sets, with families as units.

All per-family aggregation uses integer family codes and np.bincount.
"""
import numpy as np
import pandas as pd
from scipy.stats import binom


def _codes(family):
    names, code = np.unique(np.asarray(family), return_inverse=True)
    return names, code


def _fam_mean(values, code, n_fam, mask=None):
    """Per-family mean of `values` over rows in `mask`; NaN for families with no such rows."""
    if mask is None:
        mask = np.ones(len(code), bool)
    num = np.bincount(code[mask], weights=np.asarray(values, float)[mask], minlength=n_fam)
    den = np.bincount(code[mask], minlength=n_fam)
    return np.divide(num, den, out=np.full(n_fam, np.nan), where=den > 0)


def per_family(in_set, label, family):
    """Per-family coverage table (all test families)."""
    names, code = _codes(family)
    cov = np.asarray(in_set)[np.arange(len(label)), label]
    return pd.DataFrame({"n": np.bincount(code), "coverage": _fam_mean(cov, code, len(names))},
                        index=pd.Index(names, name="family"))


def per_family_counts(in_set, label, family):
    """Per-family sufficient counts, so any coverage or size metric can be
    re-aggregated (and bootstrapped over families) without the variant table."""
    in_set, label = np.asarray(in_set), np.asarray(label)
    names, code = _codes(family)
    cov = in_set[np.arange(len(label)), label]
    size = in_set.sum(1)
    b = lambda w: np.bincount(code, weights=w, minlength=len(names)).astype(int)
    out = pd.DataFrame({"n": b(None), "n_path": b(label), "k": b(cov), "k_path": b(cov & (label == 1)),
                        "k_benign": b(cov & (label == 0)), "size": b(size), "singleton": b(size == 1),
                        "full": b(size == 2)}, index=pd.Index(names, name="family"))
    out["n_benign"] = out["n"] - out["n_path"]
    return out


def null_frac_below(n, level, thresh):
    """Expected fraction of families with empirical coverage < thresh if every
    family had true coverage exactly `level` (pure binomial noise)."""
    n = np.asarray(n)
    k_max = np.ceil(thresh * n) - 1  # coverage < thresh  <=>  covered count <= k_max
    return float(np.mean(binom.cdf(k_max, n, level)))


def retained_metrics(in_set, label, code, n_fam, abstain):
    """Metrics restricted to queries the gate did not abstain on (no guarantee
    applies to this selected subset; reported as an empirical diagnostic)."""
    keep = ~np.asarray(abstain, bool)
    fam_ab = np.bincount(code[~keep], minlength=n_fam) > 0
    out = {"abstain_rate_family": fam_ab.mean(), "abstain_rate_variant": 1 - keep.mean()}
    if not keep.any():
        return out
    cov = in_set[np.arange(len(label)), label]
    single = in_set.sum(1) == 1
    out.update({
        "ret_cov_family": np.nanmean(_fam_mean(cov, code, n_fam, keep)),
        "ret_cov_path_family": np.nanmean(_fam_mean(cov, code, n_fam, keep & (label == 1))),
        "ret_cov_benign_family": np.nanmean(_fam_mean(cov, code, n_fam, keep & (label == 0))),
        "ret_singleton_rate": single[keep].mean(),
        "ret_singleton_error": (~cov[keep & single]).mean() if (keep & single).any() else np.nan,
    })
    return out


def risk_coverage(in_set, label, family, score, grid):
    """Abstain whole families in increasing order of `score` (a Series indexed
    by family; least supported first). Returns the error on retained singleton
    sets at each retained fraction of variants in `grid`, and the area under
    the curve (AURC) normalized by the grid range."""
    in_set, label = np.asarray(in_set), np.asarray(label)
    names, code = _codes(family)
    cov = in_set[np.arange(len(label)), label]
    single = in_set.sum(1) == 1
    n = np.bincount(code, minlength=len(names))
    s = np.bincount(code, weights=single, minlength=len(names))
    e = np.bincount(code, weights=single & ~cov, minlength=len(names))
    order = np.argsort(-score.loc[names].to_numpy(), kind="stable")  # most supported retained first
    n, s, e = np.cumsum(n[order]), np.cumsum(s[order]), np.cumsum(e[order])
    frac = n / n[-1]
    err = np.divide(e, s, out=np.zeros(len(s)), where=s > 0)
    curve = np.interp(grid, frac, err, left=np.nan)
    ok = ~np.isnan(curve)
    aurc = np.trapezoid(curve[ok], grid[ok]) / (grid[ok][-1] - grid[ok][0]) if ok.sum() > 1 else np.nan
    return curve, aurc


def random_risk_coverage(in_set, label, family, grid, rng, n_perm=100):
    """Risk-coverage of abstaining whole families in random order, averaged over n_perm orders."""
    names = np.unique(np.asarray(family))
    res = [risk_coverage(in_set, label, family, pd.Series(rng.random(len(names)), index=names), grid)
           for _ in range(n_perm)]
    return np.nanmean([r[0] for r in res], 0), float(np.nanmean([r[1] for r in res]))


def evaluate(in_set, label, family, alpha, min_variants=10, abstain=None):
    in_set, label = np.asarray(in_set), np.asarray(label)
    names, code = _codes(family)
    n_fam = len(names)
    cov = in_set[np.arange(len(label)), label]
    size = in_set.sum(1)
    n = np.bincount(code, minlength=n_fam)
    c = _fam_mean(cov, code, n_fam)
    big = n >= min_variants
    thresh = 1 - alpha - 0.05
    out = {
        "cov_family": c.mean(),
        "cov_variant": cov.mean(),
        "set_size": size.mean(),
        "set_size_family": _fam_mean(size, code, n_fam).mean(),
        "singleton_rate": (size == 1).mean(),
        "full_set_rate": (size == 2).mean(),
        "empty_rate": (size == 0).mean(),
        "n_test_families": n_fam,
        "n_test_variants": len(label),
        "n_fam_ge_min": int(big.sum()),
        "frac_fam_below": (c[big] < thresh).mean() if big.any() else np.nan,
        "null_frac_fam_below": null_frac_below(n[big], 1 - alpha, thresh) if big.any() else np.nan,
        "fam_cov_sd": c[big].std() if big.any() else np.nan,
        "fam_cov_q10": np.quantile(c[big], 0.1) if big.any() else np.nan,
    }
    if big.any():
        cbar = c[big].mean()
        out["fam_cov_excess_var"] = c[big].var() - np.mean(cbar * (1 - cbar) / n[big])
    for y, name in ((1, "path"), (0, "benign")):
        k = label == y
        out[f"cov_{name}_variant"] = cov[k].mean()
        out[f"cov_{name}_family"] = np.nanmean(_fam_mean(cov, code, n_fam, k))
    out["path_miss_variant"] = 1 - out["cov_path_variant"]
    out["path_miss_family"] = 1 - out["cov_path_family"]
    out.update(retained_metrics(in_set, label, code, n_fam, np.zeros(len(label), bool) if abstain is None else abstain))
    return out
