"""Family-level weighted conformal under exact family-level covariate shift."""
import numpy as np

from protcover.base_model import nonconformity
from protcover.conformal import conformal_sets
from protcover.density_ratio import cross_fit_weights
from protcover.gate import gate
from protcover.metrics import evaluate
from protcover.synthetic import simulate_shift

ALPHA, BETA = 0.1, 1.0
K_CAL, K_TEST, N_REP = 300, 150, 100
MC_TOL = 0.01


def _run(seed=1):
    rng = np.random.default_rng(seed)
    res = {k: [] for k in ("hcp", "oracle", "estimated", "oracle_cc", "gated")}
    for _ in range(N_REP):
        cal, test, w = simulate_shift(K_CAL, K_TEST, BETA, rng, slope_d=1.0)
        a_cal_all, a_test = nonconformity(cal["p1"]), nonconformity(test["p1"])
        y, f = cal["label"], cal["family"]
        a_cal = a_cal_all[np.arange(len(y)), y]
        yt, ft = test["label"], test["family"]
        ev = lambda s: evaluate(s, yt, ft, ALPHA)
        res["hcp"].append(ev(conformal_sets(a_cal, y, f, a_test, ALPHA, "family")))
        wc_o, wt_o = w(cal["d"])[f], w(test["d"])[ft]
        oracle = conformal_sets(a_cal, y, f, a_test, ALPHA, "family", w_cal=wc_o, w_test=wt_o)
        res["oracle"].append(ev(oracle))
        res["oracle_cc"].append(ev(conformal_sets(a_cal, y, f, a_test, ALPHA, "family", True, w_cal=wc_o, w_test=wt_o)))
        # Gate: abstain on a random third of test families; sets can only grow.
        fam_ab = rng.random(K_TEST) < 1 / 3
        gated = oracle.copy()
        gated[gate(np.where(fam_ab, 0, 10), np.zeros(K_TEST), 1, 1.0)[ft]] = True
        res["gated"].append(ev(gated))
        # Cross-fitted estimated ratio: a third of calibration families fit it.
        n_fit = K_CAL // 3
        fold = rng.permutation(np.arange(K_TEST) % 5)
        wc, wt = cross_fit_weights(cal["d"][:n_fit, None], cal["d"][n_fit:, None], test["d"][:, None], fold, 20.0)
        keep = f >= n_fit
        est = np.zeros(a_test.shape, bool)
        for j in range(5):
            sel = fold[ft] == j
            est[sel] = conformal_sets(a_cal[keep], y[keep], f[keep], a_test[sel], ALPHA, "family",
                                      w_cal=wc[j][f[keep] - n_fit], w_test=wt[ft[sel]])
        res["estimated"].append(ev(est))
    return {k: {m: np.mean([r[m] for r in v]) for m in v[0]} for k, v in res.items()}


R = _run()


def test_unweighted_hcp_undercovers_under_family_shift():
    assert R["hcp"]["cov_family"] < 1 - ALPHA - 3 * MC_TOL


def test_oracle_weighted_hcp_restores_family_coverage():
    assert R["oracle"]["cov_family"] >= 1 - ALPHA - MC_TOL


def test_oracle_weighted_hcp_class_conditional():
    assert R["oracle_cc"]["cov_path_family"] >= 1 - ALPHA - MC_TOL
    assert R["oracle_cc"]["cov_benign_family"] >= 1 - ALPHA - MC_TOL


def test_cross_fitted_weights_close_to_target():
    assert R["estimated"]["cov_family"] >= 1 - ALPHA - 2 * MC_TOL


def test_gate_never_lowers_coverage():
    assert R["gated"]["cov_family"] >= R["oracle"]["cov_family"]


def test_weights_finite_when_descriptors_separate_perfectly():
    rng = np.random.default_rng(0)
    xc, xt = rng.random((300, 1)), 5 + rng.random((150, 1))
    fold = rng.permutation(np.arange(150) % 5)
    for clip in (20.0, None):
        wc, wt = cross_fit_weights(xc[:100], xc[100:], xt, fold, clip)
        assert np.isfinite(wc).all() and np.isfinite(wt).all()
        assert np.allclose(wc.mean(1), 1.0) or clip is not None
