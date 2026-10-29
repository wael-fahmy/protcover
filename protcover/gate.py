"""Support gate: per-query effective calibration support and test-point mass.

For query family t, the localized calibration weights are a Gaussian kernel in
standardized descriptor space, k_tk = exp(-||x_k - x_t||^2 / (2 h^2)), and

    ESS_t = (sum_k k_tk)^2 / sum_k k_tk^2,

the effective number of calibration families that resemble the query. The
test-point mass is pi_t = w_t / (sum_k w_k + w_t) under the family-level
density-ratio weights. The gate returns the full label set when ESS_t < m or
pi_t > tau. Replacing a set by a superset cannot lower coverage, so the gate
preserves every coverage guarantee of the sets it acts on (Lemma in the paper).
"""
import numpy as np
from scipy.spatial.distance import cdist, pdist

from .density_ratio import ess


def median_bandwidth(x):
    """Median pairwise Euclidean distance (label-free bandwidth rule)."""
    return float(np.median(pdist(x)))


def kernel_ess(x_cal, x_query, h):
    d2 = cdist(x_query, x_cal, "sqeuclidean")
    return ess(np.exp(-d2 / (2 * h * h)))


def test_mass(w_cal_sum, w_test):
    return np.asarray(w_test) / (np.asarray(w_cal_sum) + np.asarray(w_test))


def gate(ess_t, pi_t, m, tau):
    return (np.asarray(ess_t) < m) | (np.asarray(pi_t) > tau)
