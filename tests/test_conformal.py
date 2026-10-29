"""Coverage tests on exchangeable synthetic data with known truth."""
import numpy as np
import pytest

from protcover.base_model import nonconformity
from protcover.conformal import conformal_sets, subsample_sets, weighted_threshold
from protcover.metrics import evaluate
from protcover.synthetic import simulate_families

ALPHA = 0.1
N_REP = 300
K_CAL, K_TEST = 150, 150


def test_unit_weights_match_split_conformal_quantile():
    rng = np.random.default_rng(0)
    a = rng.random(99)
    for alpha in (0.05, 0.1, 0.2):
        k = int(np.ceil((len(a) + 1) * (1 - alpha)))
        assert weighted_threshold(a, np.ones(len(a)), 1.0, alpha)[0] == np.sort(a)[k - 1]


def test_threshold_is_infinite_when_level_unreachable():
    assert np.isinf(weighted_threshold(np.array([0.1, 0.2]), np.ones(2), 1.0, 0.1)[0])


def _replicate(method, size_d, seed=0):
    """Mean family-weighted and variant-weighted coverage over random family splits."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_REP):
        sim = simulate_families(K_CAL + K_TEST, rng, size_d=size_d, slope_d=1.0)
        perm = rng.permutation(K_CAL + K_TEST)
        cal_f, test_f = perm[:K_CAL], perm[K_CAL:]
        fam, y = sim["family"], sim["label"]
        a = nonconformity(sim["p1"])
        c, t = np.isin(fam, cal_f), np.isin(fam, test_f)
        a_cal = a[c, y[c]]
        sets = method(a_cal, y[c], fam[c], a[t], rng)
        out.append(evaluate(sets, y[t], fam[t], ALPHA))
    return {k: np.mean([o[k] for o in out]) for k in out[0]}


def hcp(a_cal, y_cal, fam_cal, a_test, rng):
    return conformal_sets(a_cal, y_cal, fam_cal, a_test, ALPHA, scheme="family")


def hcp_mondrian(a_cal, y_cal, fam_cal, a_test, rng):
    return conformal_sets(a_cal, y_cal, fam_cal, a_test, ALPHA, scheme="family", mondrian=True)


def variant_level(a_cal, y_cal, fam_cal, a_test, rng):
    return conformal_sets(a_cal, y_cal, fam_cal, a_test, ALPHA, scheme="variant")


def subsample_once(a_cal, y_cal, fam_cal, a_test, rng):
    return subsample_sets(a_cal, y_cal, fam_cal, a_test, ALPHA, 1, rng)


# Monte Carlo standard error of a mean of N_REP coverages is about 0.002 here.
MC_TOL = 0.01


@pytest.mark.parametrize("size_d", [0.0, 1.0])
def test_hcp_family_weighted_coverage(size_d):
    r = _replicate(hcp, size_d)
    upper = 1 - ALPHA + 2 / (K_CAL + 1)  # Lee, Barber and Willett, Theorem 1 (distinct scores)
    assert 1 - ALPHA - MC_TOL <= r["cov_family"] <= upper + MC_TOL


def test_hcp_mondrian_class_conditional_coverage():
    r = _replicate(hcp_mondrian, 1.0)
    assert r["cov_path_family"] >= 1 - ALPHA - MC_TOL
    assert r["cov_benign_family"] >= 1 - ALPHA - MC_TOL


def test_subsample_once_family_weighted_coverage():
    r = _replicate(subsample_once, 1.0)
    assert r["cov_family"] >= 1 - ALPHA - MC_TOL


def test_variant_level_undercovers_families_when_large_families_are_easy():
    # Not a guarantee: documents the failure mode that motivates family weighting.
    r = _replicate(variant_level, 1.0)
    assert r["cov_family"] < 1 - ALPHA - MC_TOL
    assert r["cov_variant"] >= 1 - ALPHA - MC_TOL
