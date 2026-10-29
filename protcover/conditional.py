"""Family-level conditional calibration.

Conformal prediction with conditional guarantees (Gibbs, Cherian and Candes)
lifted from exchangeable points to exchangeable families. For a feature map Phi
of label-free family descriptors whose span F contains the constants, the
threshold for a test family t is g(D_t), where g = Phi(.)^T beta minimizes

    sum_k (1 / N_k) sum_i rho_tau(S_ki - g(D_k))  -  tau * g(D_t),   tau = 1 - alpha,

over the calibration families k: family-weighted quantile regression with the
test family's scores placed at +inf (each family carries total weight one).
Sets are {y : a(x, y) <= g(D_t)}. If F holds only the constants this is the
HCP threshold; in general, family-weighted coverage holds under every
family-level covariate shift whose density ratio is a nonnegative function in F.
An unbounded program (F can raise g(D_t) at no cost) gives threshold +inf.

The guarantee needs the largest optimal g(D_t) when the minimizer is not unique.
Raising the weight of the test term from tau to tau + TEST_EPS selects it (and can
only raise the threshold, so the guarantee is kept whatever the solver returns).
"""
import highspy
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

# Extra weight of the test term (see the module docstring); well above the solver's 1e-7 feasibility
# tolerance, so a tied basis cannot pass as feasible at the lower end.
TEST_EPS = 1e-4
# Set membership tolerance: the threshold interpolates calibration scores, and rounding must not
# exclude a test score equal to it.
THRESHOLD_TOL = 1e-9


class FamilyQuantileLP:
    """The calibration part of the program, built once and solved for many test families.

    a_cal: (n,) calibration scores; fam_cal: (n,) family of each score;
    phi_cal: (n, p) features of each score's family (rows equal within a family).

    Solved through the dual, which has one bounded variable per calibration score but
    only p equality constraints: maximize sum_j a_j S_j subject to
    sum_j a_j Phi_j = -tau_t Phi_t and (tau - 1) w_j <= a_j <= tau w_j, with w_j = 1 / N_k(j)
    and tau_t = tau + test_eps the weight of the test term.
    The primal coefficients are minus the multipliers of the equality constraints; an
    infeasible dual means an unbounded primal, i.e. threshold +inf. Test families change
    only the right-hand side, so HiGHS re-solves each one from the previous basis.
    """

    def __init__(self, a_cal, fam_cal, phi_cal, alpha, test_eps=TEST_EPS):
        self.a = np.asarray(a_cal, float)
        _, codes, counts = np.unique(np.asarray(fam_cal), return_inverse=True, return_counts=True)
        w = 1.0 / counts[codes]
        self.tau = 1.0 - alpha
        self.tau_t = self.tau + test_eps
        self.phi_t = np.asarray(phi_cal, float).T
        self.bounds = np.column_stack([(self.tau - 1.0) * w, self.tau * w])
        p, n = self.phi_t.shape
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = n, p
        lp.col_cost_ = -self.a
        lp.col_lower_, lp.col_upper_ = self.bounds[:, 0], self.bounds[:, 1]
        lp.row_lower_, lp.row_upper_ = np.zeros(p), np.zeros(p)
        A = sparse.csc_matrix(self.phi_t)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
        lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n, p
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.passModel(lp)

    def primal_threshold(self, phi_test):
        """Same value from the primal program (slow; used to test the dual)."""
        n, p = self.phi_t.shape[1], self.phi_t.shape[0]
        w = self.bounds[:, 1] / self.tau
        A = sparse.hstack([sparse.csr_matrix(self.phi_t.T), sparse.identity(n), -sparse.identity(n)]).tocsr()
        c = np.r_[-self.tau_t * np.asarray(phi_test, float), self.tau * w, (1.0 - self.tau) * w]
        res = linprog(c, A_eq=A, b_eq=self.a, bounds=[(None, None)] * p + [(0, None)] * (2 * n), method="highs")
        if res.status == 3:
            return np.inf
        return float(np.asarray(phi_test, float) @ res.x[:p])

    def threshold(self, phi_test):
        """g(D_t) for one test family with feature vector phi_test (p,)."""
        phi_test = np.asarray(phi_test, float)
        rhs = -self.tau_t * phi_test
        for r in range(len(rhs)):
            self.h.changeRowBounds(r, rhs[r], rhs[r])
        infeasible = (highspy.HighsModelStatus.kInfeasible, highspy.HighsModelStatus.kUnboundedOrInfeasible)
        for attempt in range(2):
            self.h.run()
            status = self.h.getModelStatus()
            if status in infeasible:
                self.h.clearSolver()  # start the next family from scratch rather than an infeasible basis
                return np.inf  # dual infeasible: the class can raise g(D_t) freely
            if status == highspy.HighsModelStatus.kOptimal:
                return float(-phi_test @ np.asarray(self.h.getSolution().row_dual))
            self.h.clearSolver()  # a warm start occasionally ends without a verdict: retry cold
        # Still no verdict from HiGHS's warm interface: solve cold through SciPy (same program).
        return self.threshold_scipy(phi_test)

    def threshold_scipy(self, phi_test):
        """Same value from a cold SciPy solve of the dual (reference for tests)."""
        phi_test = np.asarray(phi_test, float)
        res = linprog(-self.a, A_eq=self.phi_t, b_eq=-self.tau_t * phi_test, bounds=self.bounds, method="highs")
        if res.status == 2:
            return np.inf
        if res.status != 0:
            raise RuntimeError(f"family quantile LP failed: {res.message}")
        return float(-phi_test @ res.eqlin.marginals)


def conditional_sets(a_cal, y_cal, fam_cal, phi_cal, a_test, fam_test, phi_test_fam, alpha, mondrian=False):
    """Prediction sets for all test variants.

    a_cal, y_cal, fam_cal: calibration scores (true label), labels, families; phi_cal: (n, p) features.
    a_test: (m, 2) scores a(x, y) for y = 0, 1; fam_test: (m,) test family of each variant (any labels);
    phi_test_fam: dict-like mapping each test family to its feature vector (p,).
    Returns (m, 2) boolean membership.
    """
    a_cal, y_cal, fam_cal = np.asarray(a_cal), np.asarray(y_cal), np.asarray(fam_cal)
    phi_cal = np.asarray(phi_cal, float)
    fams = np.unique(fam_test)
    out = np.zeros(a_test.shape, bool)
    groups = [(None, np.ones(len(a_cal), bool))] if not mondrian else [(y, y_cal == y) for y in (0, 1)]
    for y, k in groups:
        lp = FamilyQuantileLP(a_cal[k], fam_cal[k], phi_cal[k], alpha)
        q = {f: lp.threshold(phi_test_fam[f]) for f in fams}
        qv = np.array([q[f] for f in fam_test])
        if y is None:
            out[:] = a_test <= qv[:, None] + THRESHOLD_TOL
        else:
            out[:, y] = a_test[:, y] <= qv + THRESHOLD_TOL
    return out
