"""Family-level density ratios w(d) = dP_target / dP_cal from label-free descriptors.

A logistic classifier separates calibration families (class 0) from target
families (class 1); its odds, corrected for the class sizes, estimate the
density ratio. Regularization is chosen by internal cross-validation on the
log loss of this membership task (no variant labels are used).

Cross-fitting keeps the weights independent of the families they are applied
to: calibration families are split into a fitting part and a conformal part,
and target families into folds. The ratio applied to target fold j and to the
conformal calibration families is fitted on the fitting calibration families
against the target families outside fold j.
"""
import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _neg_log_loss(est, x, y):
    """Mean log likelihood (same ordering as sklearn's neg_log_loss, without its input validation)."""
    p = np.clip(est.predict_proba(x)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def fit_ratio(x_src, x_tgt, seed=0):
    x = np.vstack([x_src, x_tgt])
    y = np.r_[np.zeros(len(x_src)), np.ones(len(x_tgt))]
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, cv=5, scoring=_neg_log_loss, max_iter=2000,
                                                               random_state=seed))
    clf.fit(x, y)
    log_prior = np.log(len(x_src) / len(x_tgt))
    return lambda z: log_prior + clf.decision_function(z)  # log density ratio


def cross_fit_weights(x_cal_fit, x_cal_conf, x_test, fold, clip, seed=0):
    """Returns (cal weights per fold (n_folds x n_conf), test weights).

    fold: fold index of each test family. Within each fold, weights are
    rescaled to mean 1 over the conformal calibration families and then
    clipped to [1 / clip, clip] (clip=None: no clipping).
    """
    n_folds = int(fold.max()) + 1
    w_cal = np.zeros((n_folds, len(x_cal_conf)))
    w_test = np.zeros(len(x_test))
    for j in range(n_folds):
        log_ratio = fit_ratio(x_cal_fit, x_test[fold != j], seed=seed + j)
        lc, lt = log_ratio(x_cal_conf), log_ratio(x_test[fold == j])
        # Normalize in log space (mean 1 over the conformal calibration
        # families) so near-separable descriptors cannot overflow.
        log_scale = logsumexp(lc) - np.log(len(lc))
        wc = np.exp(np.clip(lc - log_scale, -700, 700))
        wt = np.exp(np.clip(lt - log_scale, -700, 700))
        if clip is not None:
            wc, wt = np.clip(wc, 1.0 / clip, clip), np.clip(wt, 1.0 / clip, clip)
        w_cal[j], w_test[fold == j] = wc, wt
    return w_cal, w_test


def ess(w):
    """Effective sample size (sum w)^2 / sum w^2 along the last axis; 0 when all weights are 0."""
    w = np.asarray(w, float)
    s2 = (w ** 2).sum(-1)
    return np.divide(w.sum(-1) ** 2, s2, out=np.zeros(np.shape(s2)), where=s2 > 0)
