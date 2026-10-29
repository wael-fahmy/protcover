"""Family-level conditional calibration: exact special cases and the properties its proof uses."""
import numpy as np
from scipy.optimize import linprog

from protcover.conditional import FamilyQuantileLP
from protcover.conformal import base_weights, weighted_threshold
from protcover.synthetic import simulate_families


def _cal(rng, n=150):
    cal = simulate_families(n, rng)
    a = 1 - np.where(cal["label"] == 1, cal["p1"], 1 - cal["p1"])
    return a, cal["family"], cal["d"]


def test_constant_class_is_hcp():
    rng = np.random.default_rng(0)
    for _ in range(10):
        a, f, _ = _cal(rng)
        for alpha in (0.05, 0.1, 0.2):
            hcp = weighted_threshold(a, base_weights(f, "family"), 1.0, alpha)[0]
            lp = FamilyQuantileLP(a, f, np.ones((len(a), 1)), alpha).threshold(np.ones(1))
            assert np.isclose(hcp, lp)


def test_dual_equals_primal():
    rng = np.random.default_rng(1)
    a, f, d = _cal(rng)
    phi = np.column_stack([np.ones(len(a)), d[f], d[f] ** 2])
    lp = FamilyQuantileLP(a, f, phi, 0.1)
    for _ in range(5):
        pt = np.array([1.0, rng.normal(), rng.normal() ** 2])
        assert np.isclose(lp.threshold(pt), lp.primal_threshold(pt))


def test_warm_started_solves_match_cold_solves_across_an_infeasible_family():
    # A descriptor far outside the calibration range lets the linear class extrapolate
    # g(D_t) upward for free (threshold +inf); the solver must recover for the next family.
    rng = np.random.default_rng(4)
    a, f, d = _cal(rng)
    x = np.clip(d, -1, 1)
    lp = FamilyQuantileLP(a, f, np.column_stack([np.ones(len(a)), x[f]]), 0.1)
    for xt in (0.3, -0.5, 25.0, 0.1, -40.0, 0.8):
        warm, cold = lp.threshold(np.array([1.0, xt])), lp.threshold_scipy(np.array([1.0, xt]))
        assert (np.isinf(warm) and np.isinf(cold)) or np.isclose(warm, cold)
    assert np.isinf(lp.threshold(np.array([1.0, 25.0])))


def test_unbounded_when_class_can_isolate_the_test_family():
    # An indicator of the test family's descriptor is zero on every calibration family,
    # so the class can raise g(D_t) at no cost: the set must be the full label set.
    rng = np.random.default_rng(2)
    a, f, d = _cal(rng)
    phi = np.column_stack([np.ones(len(a)), np.zeros(len(a))])
    assert np.isinf(FamilyQuantileLP(a, f, phi, 0.1).threshold(np.array([1.0, 1.0])))


def _fit_value_with_test_scores(a, w, phi, phi_t, s_test, tau):
    """Family-weighted quantile regression with the test family's scores s_test included
    (weight 1 / len(s_test) each); returns the fitted value at the test family."""
    n_t = len(s_test)
    phi_all = np.vstack([phi, np.tile(phi_t, (n_t, 1))])
    y = np.r_[a, s_test]
    ww = np.r_[w, np.full(n_t, 1.0 / n_t)]
    res = linprog(-y, A_eq=phi_all.T, b_eq=np.zeros(phi_all.shape[1]),
                  bounds=np.column_stack([(tau - 1) * ww, tau * ww]), method="highs")
    return float(-phi_t @ res.eqlin.marginals)


def test_fitted_value_monotone_in_test_scores_and_bounded_by_plus_infinity():
    rng = np.random.default_rng(3)
    a, f, d = _cal(rng, 80)
    _, codes, counts = np.unique(f, return_inverse=True, return_counts=True)
    w = 1.0 / counts[codes]
    phi = np.column_stack([np.ones(len(a)), d[f]])
    tau = 0.9
    lp = FamilyQuantileLP(a, f, phi, 1 - tau)
    for _ in range(5):
        phi_t = np.array([1.0, rng.normal()])
        s = np.sort(rng.random(6))
        low = _fit_value_with_test_scores(a, w, phi, phi_t, s, tau)
        high = _fit_value_with_test_scores(a, w, phi, phi_t, s + 0.3, tau)
        assert high >= low - 1e-9
        assert lp.threshold(phi_t) >= high - 1e-9


def test_tie_returns_largest_optimal_threshold():
    # Nine single-variant families with scores 1..9 and alpha = 0.1: the pooled distribution
    # (test family at +inf) reaches 0.9 exactly at score 9, so every value in [9, +inf) is
    # optimal for the intercept-only program and the guarantee needs the largest one.
    # With four families, it reaches 0.6 exactly at score 3: optimal set [3, 4] at alpha = 0.4.
    a, f, phi = np.arange(1.0, 10.0), np.arange(9), np.ones((9, 1))
    assert np.isinf(FamilyQuantileLP(a, f, phi, 0.1).threshold(np.ones(1)))
    a, f, phi = np.array([1.0, 2.0, 3.0, 4.0]), np.arange(4), np.ones((4, 1))
    lp = FamilyQuantileLP(a, f, phi, 0.4)
    assert np.isclose(lp.threshold(np.ones(1)), 4.0)
    assert np.isclose(lp.threshold_scipy(np.ones(1)), 4.0)
    assert np.isclose(lp.primal_threshold(np.ones(1)), 4.0)
    assert np.isclose(weighted_threshold(a, base_weights(f, "family"), 1.0, 0.4)[0], 3.0)  # HCP: lower end
