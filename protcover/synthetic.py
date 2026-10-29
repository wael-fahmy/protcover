"""Synthetic hierarchical data with known truth.

Family k has a label-free descriptor d_k ~ N(0, 1), a size N_k, a random
offset b_k and a random slope c_k. Variant i of family k has a predictor score
s_ki ~ N(0, 1) and label y_ki ~ Bernoulli(sigmoid(b_k + c_k * s_ki)).

Difficulty depends on the descriptor: the slope mean is slope_mean + slope_d * d_k,
so families with low d are harder (flatter). Family size can also depend on d
through size_d, which makes large families easier when size_d > 0; that is the
regime in which variant-level conformal over-weights easy families.

The base model is a fixed p_hat(1 | s) = sigmoid(s), shared by every family.
"""
import numpy as np


def simulate_families(n_families, rng, mean_size=12.0, size_d=0.0, offset_sd=1.0,
                      slope_mean=1.5, slope_sd=0.5, slope_d=0.0, d=None):
    if d is None:
        d = rng.standard_normal(n_families)
    lam = mean_size * np.exp(size_d * d - 0.5 * size_d**2)
    sizes = 1 + rng.poisson(np.maximum(lam - 1, 0.0))
    b = offset_sd * rng.standard_normal(n_families)
    c = slope_mean + slope_d * d + slope_sd * rng.standard_normal(n_families)
    fam = np.repeat(np.arange(n_families), sizes)
    s = rng.standard_normal(fam.size)
    logit = b[fam] + c[fam] * s
    y = (rng.random(fam.size) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    p1 = 1.0 / (1.0 + np.exp(-s))
    return {"family": fam, "d": d, "sizes": sizes, "score": s, "label": y, "p1": p1}


def tilted_split(d, n_cal, n_test, beta, rng):
    """Covariate shift at the family level: calibration families are sampled
    uniformly, test families with probability proportional to exp(-beta * d)
    (from a separate pool), so the density ratio is w(d) proportional to exp(-beta * d)."""
    n = len(d)
    cal = rng.choice(n, size=n_cal, replace=False)
    rest = np.setdiff1d(np.arange(n), cal)
    p = np.exp(-beta * d[rest])
    test = rng.choice(rest, size=n_test, replace=False, p=p / p.sum())
    return cal, test
