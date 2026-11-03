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
it can stop at the wrong end of the tie, so every threshold is exact: the basis
HiGHS returns is checked in exact rational arithmetic (tau = 1 - alpha and the
weights 1 / N_k exact) to be optimal for the perturbed program, and its fitted
value is rounded up. If neither that basis nor the basis of a cold re-solve with
tight tolerances is optimal in exact arithmetic (scores within the solver's
tolerance of the fitted surface can belong at the other bound), a rational dual
simplex warm-started from the solver's basis finds an exactly optimal basis or
proves the program unbounded. HiGHS's unboundedness verdicts are checked by the
same exact simplex. Only if it reached its pivot cap would a threshold be set to
+inf, which keeps every guarantee.
"""
import heapq
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
# Pivot cap of the exact simplex; at the cap the threshold is +inf (valid, and never reached in our runs).
EXACT_MAX_PIVOTS = 10_000
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


def dyadic(values):
    """Integers m and an exponent e with values = m / 2**e exactly (every float is dyadic)."""
    ratios = [float(v).as_integer_ratio() for v in values]
    e = max(d.bit_length() - 1 for _, d in ratios)
    return [m << (e - d.bit_length() + 1) for m, d in ratios], e


def common_denominator(fracs):
    """Integers z and D > 0 with fracs = z / D."""
    D = math.lcm(*(f.denominator for f in fracs))
    return [f.numerator * (D // f.denominator) for f in fracs], D


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
    every case; ambiguous ones are decided in exact arithmetic. When the certificate fails,
    the same construction (a basis whose nonbasic dual variables sit at the bounds their
    residual signs dictate is dual feasible) starts an exact dual simplex. `stats` counts
    the outcomes.
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
        self.cache, self.stats, self._ex = {}, Counter(), None
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
                return "inf"  # the caller checks the verdict, then clears the solver
            if status == highspy.HighsModelStatus.kOptimal:
                return "opt"
            self.h.clearSolver()  # a warm start occasionally ends without a verdict: retry cold
        return None

    def threshold(self, phi_test):
        """Exact g(D_t) for one test family with feature vector phi_test (p,), rounded up."""
        phi_test = np.asarray(phi_test, float)
        rhs = -self.tau_t * phi_test
        for r in range(len(rhs)):
            self.h.changeRowBounds(r, rhs[r], rhs[r])
        verdict = self._solve()
        if verdict == "inf":
            return self._exact_infinite(phi_test, "infinite")  # dual infeasible: the class can raise g(D_t) freely
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
            return self._exact_infinite(phi_test, "infinite_after_resolve")
        if q is not None:
            self._count("certified_after_resolve", 1)
            return q
        # Neither basis is optimal in exact arithmetic: exact simplex from the last one.
        q = self._exact_simplex(phi_test, *self._highs_start())
        if q is None:
            self._count("uncertified_set_to_inf", 1)
            return np.inf  # always valid (Lemma lem:enlarge)
        self._count("exact_simplex_infinite" if np.isinf(q) else "exact_simplex", 1)
        return q

    def _highs_start(self):
        """The solver's current basis and dual values, to warm-start the exact simplex (None if unavailable)."""
        status, basic = self.h.getBasicVariables()
        if status != highspy.HighsStatus.kOk:
            return None, None
        values = np.asarray(self.h.getSolution().col_value)
        return [int(j) for j in basic], (values if len(values) == len(self.a) else None)

    def _exact_infinite(self, phi_test, key):
        """Check an unboundedness verdict of HiGHS with the exact simplex from its last basis, then
        restart the solver cold for the next family."""
        q = self._exact_simplex(phi_test, *self._highs_start())
        self.h.clearSolver()
        if q is None:
            self._count(key + "_unverified", 1)
            return np.inf
        if not np.isinf(q):
            self._count(key + "_corrected", 1)  # the program has a finite exact optimum
            return q
        self._count(key, 1)
        return np.inf

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

    def _exact_data(self):
        """Integer data of the exact simplex, built on first use: family features M[k][i] / 2**E[i]
        and scores s[j] / 2**ES, both exact."""
        if self._ex is None:
            K, p = self.fam_phi.shape
            cols = [dyadic(self.fam_phi[:, i]) for i in range(p)]
            s, ES = dyadic(self.a)
            self._ex = {"M": [[cols[i][0][k] for i in range(p)] for k in range(K)], "E": [c[1] for c in cols],
                        "s": s, "ES": ES, "sizes": [(int(N), np.flatnonzero(self.counts == N)) for N in np.unique(self.counts)]}
        return self._ex

    def _exact_simplex(self, phi_test, basic=None, values=None, out=None):
        """Bounded dual simplex on the perturbed program in exact arithmetic (`out`, a dict, receives
        the final basis for tests).

        Variables: j < n is the dual variable of score j, n + i the logical of row i (fixed at 0).
        Starts from `basic` (as from getBasicVariables; None or an exactly singular basis gives the
        slack basis) with every nonbasic score at the bound its exact residual sign dictates, which
        is dual feasible; `values`, the solver's dual values, choose the bound for zero residuals.
        Bland's rule (smallest index) picks the leaving and the entering variable. Returns the
        rounded-up fitted value of an exactly optimal basis (its optimality is checked again at the
        end), np.inf when the ratio test is empty (the row of the leaving variable then certifies
        that no dual point exists, i.e. the program is unbounded), or None at EXACT_MAX_PIVOTS.
        """
        ex = self._exact_data()
        M, E, s, ES = ex["M"], ex["E"], ex["s"], ex["ES"]
        S, codes, counts = self.a, self.codes, self.counts
        n, (K, p) = len(S), self.fam_phi.shape
        P, Q = self.tau_x.numerator, self.tau_x.denominator  # bounds (P - Q) / (Q N_k) and P / (Q N_k)
        two_E, two_ES = [1 << e for e in E], 1 << ES
        phi_x = [Fraction(float(v)) for v in phi_test]
        b = [-self.taut_x * v for v in phi_x]

        def column(v):
            if v < n:
                return [Fraction(M[codes[v]][i], two_E[i]) for i in range(p)]
            return [Fraction(int(i == v - n)) for i in range(p)]

        def inverse(bas):
            cols = [column(v) for v in bas]
            return inverse_exact([[cols[t][i] for t in range(p)] for i in range(p)])

        def bound(k, upper):
            return Fraction(P if upper else P - Q, Q * int(counts[k]))

        def solve_beta():
            c = [Fraction(s[v], two_ES) if v < n else Fraction(0) for v in bas]
            beta = [sum((Binv[t][i] * c[t] for t in range(p)), Fraction(0)) for i in range(p)]
            G, D = common_denominator([beta[i] / two_E[i] for i in range(p)])
            return beta, [sum(M[k][i] * G[i] for i in range(p)) for k in range(K)], D  # fitted values F[k] / D

        def signs(F, D):
            """Exact sign of every score's residual s_j / 2**ES - F[codes[j]] / D."""
            lo, hi, on = np.empty(K), np.empty(K), np.zeros(K, bool)
            for k in range(K):
                x = F[k] / D  # correctly rounded
                m, d = x.as_integer_ratio()
                c = m * D - F[k] * d  # sign of x - F[k] / D
                if c == 0:
                    lo[k] = hi[k] = x
                    on[k] = True
                elif c < 0:
                    lo[k], hi[k] = x, np.nextafter(x, np.inf)
                else:
                    lo[k], hi[k] = np.nextafter(x, -np.inf), x
            l, h = lo[codes], hi[codes]
            return np.where(on[codes], np.sign(S - l), np.where(S >= h, 1, -1)).astype(int)

        bas = None if basic is None else [j if j >= 0 else n - 1 - j for j in basic]
        Binv = inverse(bas) if bas is not None and len(set(bas)) == p == len(bas) else None
        if Binv is None:
            bas = [n + i for i in range(p)]
            Binv = inverse(bas)
        isb = np.zeros(n, bool)
        isb[[v for v in bas if v < n]] = True
        beta, F, D = solve_beta()
        sg = signs(F, D)
        up = sg > 0
        zero = (sg == 0) & ~isb
        if zero.any() and values is not None:
            up[zero] = values[zero] > (2 * P - Q) / (2 * Q * counts[codes[zero]])  # the nearer bound
        nb = ~isb
        num = P * np.bincount(codes[nb & up], minlength=K) - (Q - P) * np.bincount(codes[nb & ~up], minlength=K)
        C = [Fraction(0)] * p  # sum over the nonbasic scores of Phi_j a_j, kept up to date below
        for N, ks in ex["sizes"]:
            ks = ks[num[ks] != 0]
            for i in range(p):
                C[i] += Fraction(sum(M[k][i] * int(num[k]) for k in ks), two_E[i] * Q * N)
        for it in range(EXACT_MAX_PIVOTS + 1):
            rho = [b[i] - C[i] for i in range(p)]
            x = [sum((Binv[t][i] * rho[i] for i in range(p)), Fraction(0)) for t in range(p)]
            bad = []
            for t, v in enumerate(bas):
                lo, hi = (bound(codes[v], False), bound(codes[v], True)) if v < n else (0, 0)
                if not lo <= x[t] <= hi:
                    bad.append((v, t, x[t] < lo))
            if not bad:
                sg, nb = signs(F, D), ~isb
                if np.any(sg[nb & up] < 0) or np.any(sg[nb & ~up] > 0) or np.any(sg[isb] != 0):
                    return None  # not reached: the pivots keep every sign
                self._count("exact_pivots", it)
                if out is not None:
                    out.update(beta=beta, basis=list(bas), x=x, up=up.copy(), basic=isb.copy())
                return round_up(sum(phi_x[i] * beta[i] for i in range(p)))
            if it == EXACT_MAX_PIVOTS:
                break
            v_r, t_r, inc = min(bad)  # leaves to the bound it violates: lower if inc
            H, DH = common_denominator([Binv[t_r][i] / two_E[i] for i in range(p)])
            A = [sum(M[k][i] * H[i] for i in range(p)) for k in range(K)]  # pivot row per family: A[k] / DH
            asg = np.array([(a > 0) - (a < 0) for a in A])
            cand = ~isb & (asg[codes] != 0) & (up == ((asg > 0) == inc)[codes])
            # Bound-flipping ratio test. Candidates in order of |reduced cost| / |pivot entry| (ties: smallest
            # index), merged over families (within a family the order is that of the scores). Passing a
            # candidate's breakpoint moves it to its other bound and reduces the leaving variable's
            # infeasibility by |pivot entry| / N_k; the first candidate that would remove it all enters.
            # A zero first ratio (a degenerate step) is the textbook step of Bland's rule, without flips.
            idx = np.flatnonzero(cand)
            if not len(idx):
                self._count("exact_pivots", it)
                return np.inf  # the row of the leaving variable certifies that no dual point exists
            o = idx[np.lexsort((idx, np.where(up[idx], S[idx], -S[idx]), codes[idx]))]
            starts = np.flatnonzero(np.r_[True, codes[o][1:] != codes[o][:-1]])
            seg_end = np.repeat(np.r_[starts[1:], len(o)], np.diff(np.r_[starts, len(o)]))

            def entry(pos):
                j = int(o[pos])
                return Fraction(abs(s[j] * D - F[codes[j]] * two_ES), abs(A[codes[j]])), j, pos

            heap = [entry(st) for st in starts]
            heapq.heapify(heap)
            lo, hi = (bound(codes[v_r], False), bound(codes[v_r], True)) if v_r < n else (0, 0)
            slope, flips, q = (lo - x[t_r]) if inc else (x[t_r] - hi), [], None
            while heap:
                t, j, pos = heapq.heappop(heap)
                step = Fraction(abs(A[codes[j]]), DH * int(counts[codes[j]]))
                if (t == 0 and not flips) or slope <= step:
                    q = j
                    break
                slope -= step
                flips.append(j)
                if pos + 1 < seg_end[pos]:
                    heapq.heappush(heap, entry(pos + 1))
            if q is None:
                self._count("exact_pivots", it)
                return np.inf  # every candidate passed and the leaving variable stays infeasible
            for j in flips:
                k = codes[j]
                for i in range(p):  # a_j moves by -1 / N_k (upper to lower) or +1 / N_k
                    C[i] += Fraction(-M[k][i] if up[j] else M[k][i], two_E[i] * int(counts[k]))
                up[j] = not up[j]
            bq = bound(codes[q], up[q])
            for i in range(p):
                C[i] -= Fraction(M[codes[q]][i], two_E[i]) * bq
            if v_r < n:
                isb[v_r], up[v_r] = False, not inc
                br = bound(codes[v_r], not inc)
                for i in range(p):
                    C[i] += Fraction(M[codes[v_r]][i], two_E[i]) * br
            isb[q] = True
            bas[t_r] = q
            Binv = inverse(bas)
            beta, F, D = solve_beta()
        self._count("exact_pivots", EXACT_MAX_PIVOTS)
        return None

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
