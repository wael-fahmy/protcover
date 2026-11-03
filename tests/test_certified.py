"""Certified thresholds: the computed threshold is never below the exact supremum of program (3),
including the degenerate ties where a solver can stop at the wrong end of the tie (constructions of
the adversarial proof review), and the weighted quantile is exact at ties of the cumulative weight."""
from fractions import Fraction as Fr

import numpy as np

from protcover.conditional import FamilyQuantileLP, round_up
from protcover.conformal import weighted_threshold


def exact_sup(S, x, fam, tau, xt):
    """Exact supremum of the query's fitted value over argmin of program (3) with features (1, x),
    by vertex enumeration in rational arithmetic (+inf if unbounded); small instances only."""
    S, x, xt = [Fr(float(v)) for v in S], [Fr(float(v)) for v in x], Fr(float(xt))
    cnt = {}
    for f in fam:
        cnt[f] = cnt.get(f, 0) + 1
    w = [Fr(1, cnt[f]) for f in fam]
    rho = lambda u: tau * u if u >= 0 else (tau - 1) * u
    L = lambda b0, b1: sum(wj * rho(sj - b0 - b1 * xj) for sj, xj, wj in zip(S, x, w)) - tau * (b0 + b1 * xt)
    Linf = lambda d0, d1: sum(wj * rho(-(d0 + d1 * xj)) for xj, wj in zip(x, w)) - tau * (d0 + d1 * xt)
    rays = [r for v in sorted(set(x)) for r in ((-v, Fr(1)), (v, Fr(-1)))]
    if any(Linf(*d) < 0 for d in rays) or any(Linf(*d) == 0 and d[0] + d[1] * xt > 0 for d in rays):
        return float("inf")
    verts = []
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if x[i] != x[j]:
                b1 = (S[i] - S[j]) / (x[i] - x[j])
                b0 = S[i] - b1 * x[i]
                verts.append((L(b0, b1), b0 + b1 * xt))
    best = min(v for v, _ in verts)
    return max(fv for v, fv in verts if v == best)


TIES = {  # delta = 2^-delta_exp = 8 / big_n, so that sum_k F_k(1/2) = tau (8 - delta) exactly
    "narrow": (12, 2 ** 15, [0.75, 0.8, 0.8125, 0.875, 0.9, 0.9375, 0.96875], 0.984375),
    "wide": (10, 2 ** 13, [0.96875, 0.97265625, 0.9765625, 0.98046875, 0.984375, 0.98828125, 0.9921875], 0.99609375),
}


def designed_tie(delta_exp, big_n, highs, last):
    """A tie in (3) whose wrong-end basis violates the tau + 1e-4 program by only 1e-4 * delta,
    below HiGHS's feasibility tolerance (tau = 7/8, everything dyadic). Block G0 at x = 0 has the
    tie c0 in [1/2, highs[0]]; block G1 at x = 1 fixes c1 = 57/64; the query sits at x = 1 + delta."""
    rng = np.random.default_rng(0)
    g0 = [list(rng.integers(0, 129, 4) / 256) for _ in range(6)]
    big = list(rng.integers(0, 128, big_n - 8) / 256) + [0.5] + highs
    g0 += [big, [last]]
    g1 = [[v] for v in (np.arange(1, 21) * 3 / 64)]
    S, fam, x = [], [], []
    for k, s in enumerate(g0 + g1):
        S += s
        fam += [k] * len(s)
        x += [0.0 if k < len(g0) else 1.0] * len(s)
    delta = Fr(1, 2 ** delta_exp)
    # Exact supremum (derived in the construction): c1 = 57/64, lower end of the c0 tie = 1/2.
    sup = (1 + delta) * Fr(57, 64) - delta * Fr(1, 2)
    return np.array(S), np.array(fam), np.column_stack([np.ones(len(S)), np.array(x)]), delta, sup


def test_designed_ties_return_the_supremum_cold_and_warm():
    for args in TIES.values():
        S, fam, phi, delta, sup = designed_tie(*args)
        xt = float(1 + delta)
        cold = FamilyQuantileLP(S, fam, phi, 1 / 8)
        assert Fr(cold.threshold(np.array([1.0, xt]))) >= sup
        warm = FamilyQuantileLP(S, fam, phi, 1 / 8)
        warm.threshold(np.array([1.0, float(1 + delta - Fr(1, 2 ** 24))]))  # previous query: the other end
        q = warm.threshold(np.array([1.0, xt]))
        assert Fr(q) >= sup and q == round_up(sup)
        assert warm.stats["uncertified_set_to_inf"] == 0


def test_certified_threshold_is_at_least_the_exact_supremum():
    rng = np.random.default_rng(3)
    n_finite = 0
    for _ in range(25):
        K = int(rng.integers(8, 16))
        sizes = rng.integers(1, 4, K)
        fam = np.repeat(np.arange(K), sizes)
        x = (rng.integers(0, 33, K) / 8)[fam]  # dyadic features: ties are likely
        S = rng.integers(0, 65, len(fam)) / 64
        lp = FamilyQuantileLP(S, fam, np.column_stack([np.ones(len(S)), x]), 0.1)
        for xt in rng.integers(-4, 37, 4) / 8:
            q = lp.threshold(np.array([1.0, xt]))
            ex = exact_sup(S, x, fam, Fr(9, 10), xt)
            if ex == float("inf"):
                assert np.isinf(q)
                continue
            n_finite += 1
            assert np.isinf(q) or Fr(q) >= ex
        assert lp.stats["uncertified_set_to_inf"] == 0
    assert n_finite > 20


def test_weighted_quantile_is_exact_at_ties_of_the_cumulative_weight():
    # Ten variants of weight 1 and a test point: the level 0.9 * 11 = 9.9 is not attained, 0.9 * 10 = 9 is.
    a = np.arange(1.0, 11.0)
    assert weighted_threshold(a[:9], np.ones(9), 1.0, 0.1)[0] == 9.0
    assert weighted_threshold(a, np.ones(10), 1.0, 0.1)[0] == 10.0
    # Nine families of three variants (weights exactly 1/3): the level 0.9 * (9 + 1) = 9 is reached
    # exactly at the last score, although 27 rounded thirds sum to less than 9.
    s = np.arange(27.0)
    assert weighted_threshold(s, np.ones(27), 1.0, 0.1, np.full(27, 3))[0] == 26.0
    # Cumulative weight below the level by about one part in 10^15: never reached, so +inf (a
    # relative tolerance window would have returned the last score).
    w = np.r_[np.ones(8), 1 - 1e-14]
    assert np.isinf(weighted_threshold(a[:9], w, 1.0, 0.1)[0])
