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
Every exact minimizer of the program with the test weight raised from tau to
tau + TEST_EPS has a fitted value at least that largest one, so its threshold is
valid. A solver only finds such a minimizer up to its tolerances, and near a tie
it can stop at the wrong end of the tie, so every threshold is certified: the
basis HiGHS returns is checked in exact rational arithmetic (tau = 1 - alpha and
the weights 1 / N_k exact) to be optimal for the perturbed program, and its
fitted value is rounded up. A threshold that cannot be certified after a cold
re-solve with tight tolerances is set to +inf, which keeps every guarantee.
"""
import math
from collections import Counter
from fractions import Fraction

import highspy
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

# Extra weight of the test term (see the module docstring).
TEST_EPS = 1e-4
# Set membership tolerance: the threshold interpolates calibration scores, and rounding must not
# exclude a test score equal to it. It only enlarges sets.
THRESHOLD_TOL = 1e-9
U = 2.0 ** -53  # unit roundoff of float64
TIGHT = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}
# Certification outcomes summed over every program of the process (the run scripts write them per seed).
TOTALS = Counter()


def exact_decimal(x):
    """The decimal a float was written as (0.1 -> 1/10), so that tau = 1 - alpha is exact."""
    return Fraction(repr(float(x)))


def round_up(q):
    """Smallest float at least the rational q."""
    f = float(q)
    return f if Fraction(f) >= q else float(np.nextafter(f, np.inf))


def inverse_exact(B):
    """Inverse of a square matrix of Fractions by Gauss-Jordan elimination; None if singular."""
    p = len(B)
    M = [list(B[i]) + [Fraction(int(i == j)) for j in range(p)] for i in range(p)]
    for c in range(p):
        piv = next((r for r in range(c, p) if M[r][c] != 0), None)
        if piv is None:
            return None
        M[c], M[piv] = M[piv], M[c]
        inv = 1 / M[c][c]
        M[c] = [v * inv for v in M[c]]
        for r in range(p):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [v - f * w for v, w in zip(M[r], M[c])]
    return [row[p:] for row in M]


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

    Certificate (see the module docstring): for the basis returned, beta solves the p
    interpolation equations exactly; every other score's dual variable is put at the bound
    that the sign of its exact residual dictates; the basic dual variables then solve a
    p x p system, and the basis is optimal for the perturbed program if and only if they lie
    within their bounds. Floating-point evaluation with rigorous error bounds decides almost
    every case; ambiguous ones are decided in exact arithmetic. `stats` counts the outcomes.
    """

    def __init__(self, a_cal, fam_cal, phi_cal, alpha, test_eps=TEST_EPS):
        self.a = np.asarray(a_cal, float)
        _, codes, counts = np.unique(np.asarray(fam_cal), return_inverse=True, return_counts=True)
        w = 1.0 / counts[codes]
        self.tau = 1.0 - alpha
        self.tau_t = self.tau + test_eps
        phi_cal = np.asarray(phi_cal, float)
        self.phi_t = phi_cal.T
        self.bounds = np.column_stack([(self.tau - 1.0) * w, self.tau * w])
        p, n = self.phi_t.shape
        # Exact data of the certificate: one feature row and one size per family.
        self.codes, self.counts = codes, counts
        first = np.zeros(len(counts), int)
        first[codes[::-1]] = np.arange(n)[::-1]
        self.fam_phi = phi_cal[first]
        if not np.array_equal(self.fam_phi[codes], phi_cal):
            raise ValueError("feature rows must be equal within a family")
        self.tau_x = 1 - exact_decimal(alpha)
        self.taut_x = self.tau_x + exact_decimal(test_eps)
        self.cache, self.stats = {}, Counter()
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
        self.default_tol = {}
        for k in TIGHT:
            v = self.h.getOptionValue(k)
            self.default_tol[k] = float(v[1] if isinstance(v, tuple) else v)  # highspy returns (status, value)

    def _count(self, key, n=1):
        self.stats[key] += int(n)
        TOTALS[key] += int(n)

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

    def _solve(self):
        """Run HiGHS up to twice (warm, then cold); 'inf', 'opt' or None (no verdict)."""
        infeasible = (highspy.HighsModelStatus.kInfeasible, highspy.HighsModelStatus.kUnboundedOrInfeasible)
        for _ in range(2):
            self.h.run()
            status = self.h.getModelStatus()
            if status in infeasible:
                self.h.clearSolver()  # start the next family from scratch rather than an infeasible basis
                return "inf"
            if status == highspy.HighsModelStatus.kOptimal:
                return "opt"
            self.h.clearSolver()  # a warm start occasionally ends without a verdict: retry cold
        return None

    def threshold(self, phi_test):
        """Certified g(D_t) for one test family with feature vector phi_test (p,), rounded up."""
        phi_test = np.asarray(phi_test, float)
        rhs = -self.tau_t * phi_test
        for r in range(len(rhs)):
            self.h.changeRowBounds(r, rhs[r], rhs[r])
        verdict = self._solve()
        if verdict == "inf":
            self._count("infinite", 1)
            return np.inf  # dual infeasible: the class can raise g(D_t) freely
        q = self._certified(phi_test) if verdict == "opt" else None
        if q is not None:
            self._count("certified", 1)
            return q
        # Not certified: re-solve cold with tight tolerances and certify again.
        self.h.clearSolver()
        for k, v in TIGHT.items():
            self.h.setOptionValue(k, v)
        verdict = self._solve()
        q = self._certified(phi_test) if verdict == "opt" else None
        for k, v in self.default_tol.items():
            self.h.setOptionValue(k, v)
        if verdict == "inf":
            self._count("infinite_after_resolve", 1)
            return np.inf
        if q is not None:
            self._count("certified_after_resolve", 1)
            return q
        self._count("uncertified_set_to_inf", 1)
        return np.inf  # always valid (Lemma lem:enlarge)

    def _basis_entry(self, key):
        """Exact quantities of one basis that do not depend on the test family (cached)."""
        S, codes, phi = self.a, self.codes, self.fam_phi
        p = self.phi_t.shape[0]
        cols = [j for j in key if j >= 0]
        rows = [-1 - j for j in key if j < 0]
        # Basis matrix of the dual's equality constraints: structural columns Phi_j, logical columns e_i.
        B = [[Fraction(float(phi[codes[j], i])) for j in cols] + [Fraction(int(i == r)) for r in rows] for i in range(p)]
        Binv = inverse_exact(B)
        if Binv is None:
            return None
        # beta: Phi_j' beta = S_j for basic scores, beta_i = 0 for basic logicals (B' beta = c_B).
        c_b = [Fraction(float(S[j])) for j in cols] + [Fraction(0)] * len(rows)
        beta = [sum(Binv[s][i] * c_b[s] for s in range(p)) for i in range(p)]
        beta_f = np.array([float(b) for b in beta])
        beta_err = np.array([float(abs(b - Fraction(f))) for b, f in zip(beta, beta_f)]) * (1 + 4 * U)
        fitted = phi @ beta_f
        fit_err = np.abs(phi) @ beta_err + (p + 2) * U * (np.abs(phi) @ np.abs(beta_f))
        r = S - fitted[codes]
        r_err = 1.01 * (fit_err[codes] + 2 * U * (np.abs(S) + np.abs(fitted[codes])))
        sign = np.where(r > r_err, 1, np.where(r < -r_err, -1, 0))
        sign[cols] = 0
        free = set(cols)
        degenerate = []
        for j in np.flatnonzero(sign == 0):
            if j in free:
                continue
            k = codes[j]
            rx = Fraction(float(S[j])) - sum(Fraction(float(phi[k, i])) * beta[i] for i in range(p))
            if rx == 0:
                degenerate.append(int(j))  # another score on the fitted surface: its dual variable is free too
            else:
                sign[j] = 1 if rx > 0 else -1
        K = len(self.counts)
        n_pos = np.bincount(codes[sign > 0], minlength=K)
        n_neg = np.bincount(codes[sign < 0], minlength=K)
        P, Q = self.tau_x.numerator, self.tau_x.denominator  # tau = P / Q
        num = P * n_pos - (Q - P) * n_neg  # g_k = num_k / (Q N_k), the fixed duals summed per family
        g = num / (Q * self.counts.astype(float))
        terms = phi * g[:, None]
        C = np.array([math.fsum(terms[:, i]) for i in range(p)])
        C_err = 4 * U * np.abs(terms).sum(0) + 2 * U * np.abs(C)
        lo = [(self.tau_x - 1) / int(self.counts[codes[j]]) for j in cols]
        hi = [self.tau_x / int(self.counts[codes[j]]) for j in cols]
        absBinv = np.array([[float(abs(v)) for v in row] for row in Binv]) * (1 + 4 * U)
        return {"cols": cols, "rows": rows, "Binv": Binv, "absBinv": absBinv, "beta": beta, "C": C, "C_err": C_err,
                "lo": lo, "hi": hi, "num": num, "degenerate": degenerate, "C_x": None}

    def _exact_C(self, e):
        """sum_k Phi_k g_k in exact arithmetic, grouped by family size (rarely needed)."""
        if e["C_x"] is None:
            p, Q = self.phi_t.shape[0], self.tau_x.denominator
            out = [Fraction(0)] * p
            for N in np.unique(self.counts):
                ks = np.flatnonzero((self.counts == N) & (e["num"] != 0))
                for i in range(p):
                    s = sum((Fraction(float(self.fam_phi[k, i])) * int(e["num"][k]) for k in ks), Fraction(0))
                    out[i] += s / (Q * int(N))
            e["C_x"] = out
        return e["C_x"]

    def _certified(self, phi_test):
        """The rounded-up fitted value of the returned basis if it is certified optimal for the
        perturbed program, else None."""
        status, basic = self.h.getBasicVariables()
        if status != highspy.HighsStatus.kOk:
            return None
        key = tuple(int(j) for j in basic)
        if key not in self.cache:
            self.cache[key] = self._basis_entry(key)
        e = self.cache[key]
        if e is None:
            return None
        p = self.phi_t.shape[0]
        nc = len(e["cols"])
        phi_x = [Fraction(float(v)) for v in phi_test]
        exact = bool(e["rows"]) or bool(e["degenerate"])  # logicals must be exactly 0; degenerate scores need their duals
        if not exact:
            tphi = float(self.taut_x) * phi_test
            rho = -tphi - e["C"]
            rho_err = 3 * U * np.abs(tphi) + e["C_err"] + 2 * U * np.abs(rho)
            x = [sum(e["Binv"][s][i] * Fraction(float(rho[i])) for i in range(p)) for s in range(p)]
            x_err = (e["absBinv"] @ rho_err) * (1 + 4 * p * U)
            inside = all(e["lo"][s] + Fraction(float(x_err[s])) <= x[s] <= e["hi"][s] - Fraction(float(x_err[s]))
                         for s in range(nc))
            outside = any(x[s] + Fraction(float(x_err[s])) < e["lo"][s] or x[s] - Fraction(float(x_err[s])) > e["hi"][s]
                          for s in range(nc))
            if outside:
                return None
            exact = not inside
            self._count("exact_check", exact)
        else:
            self._count("exact_check", 1)
        if exact:
            C = self._exact_C(e)
            rho = [-self.taut_x * phi_x[i] - C[i] for i in range(p)]
            if e["degenerate"]:
                # Degenerate scores keep the dual values HiGHS gave them, clipped to their exact bounds.
                val = self.h.getSolution().col_value
                for j in e["degenerate"]:
                    N = int(self.counts[self.codes[j]])
                    v = min(max(Fraction(float(val[j])), (self.tau_x - 1) / N), self.tau_x / N)
                    for i in range(p):
                        rho[i] -= Fraction(float(self.fam_phi[self.codes[j], i])) * v
            x = [sum(e["Binv"][s][i] * rho[i] for i in range(p)) for s in range(p)]
            if not all(e["lo"][s] <= x[s] <= e["hi"][s] for s in range(nc)) or any(x[s] != 0 for s in range(nc, p)):
                return None
        return round_up(sum(phi_x[i] * e["beta"][i] for i in range(p)))

    def threshold_scipy(self, phi_test):
        """Same value from a cold SciPy solve of the dual (reference for tests; not certified)."""
        phi_test = np.asarray(phi_test, float)
        res = linprog(-self.a, A_eq=self.phi_t, b_eq=-self.tau_t * phi_test, bounds=self.bounds, method="highs")
        if res.status == 2:
            return np.inf
        if res.status != 0:
            raise RuntimeError(f"family quantile LP failed: {res.message}")
        return float(-phi_test @ res.eqlin.marginals)


def conditional_sets(a_cal, y_cal, fam_cal, phi_cal, a_test, fam_test, phi_test_fam, alpha, mondrian=False, stats=None):
    """Prediction sets for all test variants.

    a_cal, y_cal, fam_cal: calibration scores (true label), labels, families; phi_cal: (n, p) features.
    a_test: (m, 2) scores a(x, y) for y = 0, 1; fam_test: (m,) test family of each variant (any labels);
    phi_test_fam: dict-like mapping each test family to its feature vector (p,).
    stats: optional Counter that receives the certification outcomes.
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
        if stats is not None:
            stats.update(lp.stats)
        qv = np.array([q[f] for f in fam_test])
        if y is None:
            out[:] = a_test <= qv[:, None] + THRESHOLD_TOL
        else:
            out[:, y] = a_test[:, y] <= qv + THRESHOLD_TOL
    return out
