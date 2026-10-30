"""Coverage of family-level conditional calibration where Theorem 1 applies exactly: a family-level
shift whose density ratio lies in the span of the features."""
import numpy as np

from protcover.base_model import nonconformity
from protcover.conditional import conditional_sets
from protcover.conformal import conformal_sets
from protcover.metrics import evaluate
from protcover.synthetic import simulate_families, simulate_shift

ALPHA = 0.1
K_CAL, K_TEST, N_REP = 300, 150, 100
MC_TOL = 0.01


def _phi(v):
    return np.column_stack([np.ones(len(v)), v])


def _scores(cal, test):
    a_cal_all, a_test = nonconformity(cal["p1"]), nonconformity(test["p1"])
    y, f = cal["label"], cal["family"]
    return a_cal_all[np.arange(len(y)), y], y, f, a_test, test["label"], test["family"]


def _run(seed=3):
    rng = np.random.default_rng(seed)
    res = {k: [] for k in ("hcp_lin", "desc_lin", "hcp_gauss", "ratio_gauss")}
    for _ in range(N_REP):
        # Linear ratio: calibration d ~ U(-2, 2), test density (2 - d) / 8, so w(d) = (2 - d) / 2.
        cal = simulate_families(K_CAL, rng, slope_d=1.0, d=rng.uniform(-2.0, 2.0, K_CAL))
        test = simulate_families(K_TEST, rng, slope_d=1.0, d=2.0 - 4.0 * np.sqrt(rng.random(K_TEST)))
        a_cal, y, f, a_test, yt, ft = _scores(cal, test)
        ev = lambda s: evaluate(s, yt, ft, ALPHA)
        res["hcp_lin"].append(ev(conformal_sets(a_cal, y, f, a_test, ALPHA, "family", True)))
        res["desc_lin"].append(ev(conditional_sets(a_cal, y, f, _phi(cal["d"])[f], a_test, ft, _phi(test["d"]),
                                                   ALPHA, True)))
        # Gaussian shift: the true ratio as a feature.
        cal, test, w = simulate_shift(K_CAL, K_TEST, 1.0, rng, slope_d=1.0)
        a_cal, y, f, a_test, yt, ft = _scores(cal, test)
        ev = lambda s: evaluate(s, yt, ft, ALPHA)
        res["hcp_gauss"].append(ev(conformal_sets(a_cal, y, f, a_test, ALPHA, "family", True)))
        res["ratio_gauss"].append(ev(conditional_sets(a_cal, y, f, _phi(w(cal["d"]))[f], a_test, ft,
                                                      _phi(w(test["d"])), ALPHA, True)))
    return {k: {m: np.mean([r[m] for r in v]) for m in v[0]} for k, v in res.items()}


R = _run()


def _worse(r):
    return min(r["cov_path_family"], r["cov_benign_family"])


def test_hcp_undercovers_under_both_shifts():
    assert _worse(R["hcp_lin"]) < 1 - ALPHA - 3 * MC_TOL
    assert _worse(R["hcp_gauss"]) < 1 - ALPHA - 3 * MC_TOL


def test_descriptor_features_cover_when_the_ratio_is_affine_in_the_descriptor():
    assert _worse(R["desc_lin"]) >= 1 - ALPHA - MC_TOL


def test_ratio_feature_covers_under_the_shift_it_describes():
    assert _worse(R["ratio_gauss"]) >= 1 - ALPHA - MC_TOL
