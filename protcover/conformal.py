"""Conformal prediction sets over the label space {0 = benign, 1 = pathogenic}.

Every quantile-based method below is an instance of one primitive: the
(1 - alpha) quantile of the weighted measure

    sum_i w_i delta_{a_i} + w_test delta_{+inf}

over calibration scores a_i = a(x_i, y_i). The choice of weights gives
  variant-level split conformal (B1)     w_i = 1,                   w_test = 1
  family-level pooling / HCP (B3, (i))   w_i = 1 / N_k(i),          w_test = 1
  family-level weighted ((ii))           w_i = w(d_k(i)) / N_k(i),  w_test = w(d_test)
where N_k is the number of calibration variants of family k (within the label
class for Mondrian versions). Sets are {y : a(x, y) <= threshold}.
"""
from fractions import Fraction
from itertools import accumulate

import numpy as np

U = 2.0 ** -53  # unit roundoff of float64


def weighted_threshold(a_cal, w_cal, w_test, alpha, den=None):
    """Q_{1-alpha}(sum_i w_i delta_{a_i} + w_test delta_inf) for each entry of w_test, with
    w_i = w_cal_i / den_i (den: optional positive integers, e.g. family sizes).

    Q_tau(P) = inf{t : P(T <= t) >= tau}. Returns +inf where the finite part
    never reaches the level. Exact for weights w_cal_i / den_i (doubles over integers) and
    tau = 1 - alpha with alpha read as a decimal: where rounding of the cumulative weights
    could decide the comparison with the level wrongly (in either direction), it is redone
    in exact rational arithmetic.
    """
    a_cal = np.asarray(a_cal, float)
    w_test = np.atleast_1d(np.asarray(w_test, float))
    if len(a_cal) == 0:
        return np.full(w_test.shape, np.inf)
    num = np.asarray(w_cal, float)
    den = np.ones(len(a_cal), np.int64) if den is None else np.asarray(den, np.int64)
    order = np.argsort(a_cal, kind="stable")
    s, num, den = a_cal[order], num[order], den[order]
    w = num / den
    c = np.cumsum(w)
    n = len(c)
    target = (1.0 - alpha) * (c[-1] + w_test)
    # Rigorous bound on the rounding error of every prefix sum and of the level (weights >= 0).
    err = 4.0 * (n + 4) * U * (c[-1] + np.abs(w_test))
    lo = np.searchsorted(c, target - err, side="left")
    hi = np.searchsorted(c, target + err, side="left")
    idx = lo.copy()
    amb = np.flatnonzero(lo != hi)
    if len(amb):
        cum = list(accumulate(Fraction(float(v)) / int(d) for v, d in zip(num, den)))
        tau = 1 - Fraction(repr(float(alpha)))
        for t in amb:
            level = tau * (cum[-1] + Fraction(float(w_test[t])))
            i = lo[t]
            while i < hi[t] and cum[i] < level:
                i += 1
            idx[t] = i
    return np.where(idx < n, s[np.minimum(idx, n - 1)], np.inf)


def _codes(groups):
    _, codes, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return codes, counts


def base_denominators(fam_cal, scheme):
    """Integer N such that each calibration score's base weight is 1 / N."""
    if scheme == "variant":
        return np.ones(len(fam_cal), np.int64)
    if scheme == "family":
        codes, counts = _codes(fam_cal)
        return counts[codes].astype(np.int64)
    raise ValueError(scheme)


def base_weights(fam_cal, scheme):
    return 1.0 / base_denominators(fam_cal, scheme)


def conformal_sets(a_cal, y_cal, fam_cal, a_test, alpha, scheme="variant", mondrian=False,
                   w_cal=None, w_test=None):
    """Quantile-based conformal sets.

    a_cal: (n,) calibration scores of the true label; y_cal, fam_cal: (n,)
    a_test: (m, 2) scores a(x, y) for y = 0, 1
    w_cal: optional (n,) density-ratio weight of each calibration point's family
    w_test: optional (m,) density-ratio weight of each test point's family
    Returns (m, 2) boolean membership.
    """
    a_cal, y_cal, fam_cal = np.asarray(a_cal), np.asarray(y_cal), np.asarray(fam_cal)
    m = a_test.shape[0]
    w_cal = np.ones(len(a_cal)) if w_cal is None else np.asarray(w_cal, float)
    w_test = np.ones(m) if w_test is None else np.asarray(w_test, float)
    if not mondrian:
        q = weighted_threshold(a_cal, w_cal, w_test, alpha, base_denominators(fam_cal, scheme))
        return a_test <= q[:, None]
    out = np.zeros(a_test.shape, bool)
    for y in (0, 1):
        k = y_cal == y
        q = weighted_threshold(a_cal[k], w_cal[k], w_test, alpha, base_denominators(fam_cal[k], scheme))
        out[:, y] = a_test[:, y] <= q
    return out


def gwcp_sets(a_cal, y_cal, fam_cal, bin_cal, a_test, alpha, q, mondrian=False):
    """Group-weighted conformal prediction (Bhattacharyya and Barber) with
    descriptor bins as groups, pooled at the family level.

    q: target proportion of each bin (array indexed by bin). The threshold is
    the (1 - alpha) quantile of sum_b q_b P_b, where P_b averages the score
    ECDFs of the calibration families in bin b (each family weight 1); bins
    with no calibration family contribute a point mass at +inf. There is no
    test-point mass, as in GWCP.
    """
    a_cal, y_cal, fam_cal, bin_cal = map(np.asarray, (a_cal, y_cal, fam_cal, bin_cal))
    q = np.asarray(q, float)
    out = np.zeros(a_test.shape, bool)
    for y in (0, 1):
        k = (y_cal == y) if mondrian else np.ones(len(a_cal), bool)
        fam_n = base_denominators(fam_cal[k], "family")  # N_k within the class subset
        bins_present = np.unique(bin_cal[k])
        n_fam_bin = {b: len(np.unique(fam_cal[k][bin_cal[k] == b])) for b in bins_present}
        num = np.array([q[b] for b in bin_cal[k]])
        den = np.array([n_fam_bin[b] for b in bin_cal[k]], np.int64) * fam_n  # weight q_b / (n_b N_k)
        w_inf = q[np.setdiff1d(np.arange(len(q)), bins_present)].sum()
        thr = weighted_threshold(a_cal[k], num, w_inf, alpha, den)[0]
        out[:, y] = a_test[:, y] <= thr
    return out


def _subsample_pvalues(a_cal, fam_cal, a_query, n_rep, rng):
    """Average over n_rep subsamples (one calibration point per family) of the
    conformal p-value (1 + #{k : a_k >= a}) / (K + 1)."""
    codes, _ = _codes(fam_cal)
    n_fam = codes.max() + 1
    p = np.zeros(len(a_query))
    for _ in range(n_rep):
        order = np.lexsort((rng.random(len(codes)), codes))
        last = np.r_[np.flatnonzero(np.diff(codes[order])), len(order) - 1]
        s = np.sort(a_cal[order[last]])
        n_ge = n_fam - np.searchsorted(s, a_query, side="left")
        p += (1.0 + n_ge) / (n_fam + 1.0)
    return p / n_rep


def subsample_sets(a_cal, y_cal, fam_cal, a_test, alpha, n_rep, rng, mondrian=False):
    """Dunn, Wasserman and Ramdas subsampling: n_rep = 1 is "subsampling once"
    (coverage >= 1 - alpha); n_rep > 1 is "repeated subsampling" (guaranteed
    >= 1 - 2 alpha). y is included when the averaged p-value exceeds alpha."""
    a_cal, y_cal, fam_cal = np.asarray(a_cal), np.asarray(y_cal), np.asarray(fam_cal)
    out = np.zeros(a_test.shape, bool)
    for y in (0, 1):
        k = (y_cal == y) if mondrian else np.ones(len(a_cal), bool)
        out[:, y] = _subsample_pvalues(a_cal[k], fam_cal[k], a_test[:, y], n_rep, rng) > alpha
    return out
