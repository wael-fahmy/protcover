"""Per-split machinery shared by the outer evaluation and the inner tuning split.

Method names
  B1_split    variant-level split conformal (cc=True: class-conditional, i.e. B2)
  B3_pool     family-level pooling = hierarchical conformal prediction (ProtCOVER (i))
  B3_sub1     Dunn et al. subsampling once
  B3_subR     Dunn et al. repeated subsampling
  B4_gwcp     group-weighted conformal over descriptor bins
  P_w         ProtCOVER (i)+(ii): family-level weighted conformal, cross-fitted ratios
  P_gate      ProtCOVER (i)+(iii): B3_pool sets with the ESS gate
  P_full      ProtCOVER (i)+(ii)+(iii)
  B5_support  P_w sets, abstaining on the lowest max-identity families at P_full's rate
  B5_random   P_w sets, abstaining on random families at P_full's rate
Every method has a marginal (cc=False) and a class-conditional (cc=True) version.
"""
import numpy as np
import pandas as pd

from .conformal import conformal_sets, gwcp_sets, subsample_sets
from .density_ratio import cross_fit_weights
from .descriptors import SCORE_STATS
from .gate import gate, kernel_ess, median_bandwidth, test_mass
from .metrics import _fam_mean, risk_coverage

RISK_GRID = np.round(np.arange(0.05, 1.0001, 0.05), 2)

DESCRIPTOR_SETS = {
    "seq": ["max_id_to_train"],
    "depth": ["log_neff", "log_neff_L"],
    "emb": ["emb_dist_to_train"],
    "support": ["max_id_to_train", "log_neff", "log_neff_L", "emb_dist_to_train"],
    "main": ["max_id_to_train", "log_neff", "log_neff_L", "emb_dist_to_train", "log_length", "log_n_variants"],
}
SCORES_SET = "main+scores"


def descriptor_columns(name, pred=None):
    if name == SCORES_SET:
        return DESCRIPTOR_SETS["main"] + [f"{pred}_{s}" for s in SCORE_STATS]
    return DESCRIPTOR_SETS[name]


class Split:
    """Calibration families split into a ratio-fitting part and a conformal part;
    test families split into cross-fitting folds. Everything here is label-free."""

    def __init__(self, desc, train, cal, test, ex, rng):
        self.desc = desc
        self.train, self.cal, self.test = np.sort(train), np.sort(cal), np.sort(test)
        perm = rng.permutation(self.cal)
        n_fit = int(round(ex["frac_cal_fit"] * len(self.cal)))
        self.cal_fit, self.cal_conf = np.sort(perm[:n_fit]), np.sort(perm[n_fit:])
        self.n_folds = ex["n_folds"]
        self.fold = rng.permutation(np.arange(len(self.test)) % self.n_folds)
        self._weights = {}
        self.tie = pd.Series(rng.random(len(self.test)), index=self.test)  # random tie-breaks and B5_random

        # Gate statistics on standardized support descriptors (scaled over cal and test families).
        cols = DESCRIPTOR_SETS[ex["gate_descriptors"]]
        x = desc.loc[np.r_[self.cal, self.test], cols]
        x = (x - x.mean()) / x.std(ddof=0)
        h = median_bandwidth(x.to_numpy())
        xt = x.loc[self.test].to_numpy()
        # Kernel ESS at several bandwidths (fractions of the median distance); the
        # fraction is a gate hyperparameter chosen on the inner split.
        self.ess_conf = {sc: pd.Series(kernel_ess(x.loc[self.cal_conf].to_numpy(), xt, sc * h), index=self.test)
                         for sc in ex["gate_scale_grid"]}
        self.ess_full = {sc: pd.Series(kernel_ess(x.loc[self.cal].to_numpy(), xt, sc * h), index=self.test)
                         for sc in ex["gate_scale_grid"]}
        self.max_id = desc.loc[self.test, "max_id_to_train"]

        # GWCP bins: max identity (0, low, high) x log Neff tercile.
        both = desc.loc[np.r_[self.cal, self.test]]
        pos = both["max_id_to_train"][both["max_id_to_train"] > 0]
        id_bin = np.where(both["max_id_to_train"] <= 0, 0, np.where(both["max_id_to_train"] <= pos.median(), 1, 2))
        neff_bin = pd.qcut(both["log_neff"], 3, labels=False).to_numpy()
        self.bin = pd.Series(3 * id_bin + neff_bin, index=both.index)
        self.q = np.bincount(self.bin.loc[self.test], minlength=9) / len(self.test)

    def weights(self, dset, clip, pred=None, seed=0):
        """Cached cross-fitted weights: (w_cal (n_folds x n_conf), w_test (n_test,))."""
        key = (dset, clip, pred if dset == SCORES_SET else None)
        if key not in self._weights:
            cols = descriptor_columns(dset, pred)
            x = self.desc[cols]
            self._weights[key] = cross_fit_weights(x.loc[self.cal_fit].to_numpy(), x.loc[self.cal_conf].to_numpy(),
                                                   x.loc[self.test].to_numpy(), self.fold, clip, seed)
        return self._weights[key]

    def test_mass(self, w_cal, w_test):
        return pd.Series(test_mass(w_cal.sum(1)[self.fold], w_test), index=self.test)


def weighted_hcp(split, a, label, vfam, alpha, cc, w_cal, w_test):
    """ProtCOVER (ii): family-level weighted conformal sets for all test variants."""
    ci, ti = np.isin(vfam, split.cal_conf), np.isin(vfam, split.test)
    cal_code = np.searchsorted(split.cal_conf, vfam[ci])
    test_code = np.searchsorted(split.test, vfam[ti])
    a_test = a[ti]
    out = np.zeros(a_test.shape, bool)
    a_cal = a[ci, label[ci]]
    for j in range(split.n_folds):
        sel = split.fold[test_code] == j
        out[sel] = conformal_sets(a_cal, label[ci], vfam[ci], a_test[sel], alpha, "family", cc,
                                  w_cal=w_cal[j][cal_code], w_test=w_test[test_code[sel]])
    return out


def _abstain(sets, fam_abstain, test_code):
    ab = np.asarray(fam_abstain)[test_code]
    s = sets.copy()
    s[ab] = True
    return s, ab


def run_methods(split, a, label, vfam, alpha, ex, gate_params, pred, rng, ablations=True):
    """Returns a list of (method, cc, sets, abstain-per-test-variant or None)."""
    cal, te = np.isin(vfam, split.cal), np.isin(vfam, split.test)
    tf = np.searchsorted(split.test, vfam[te])
    a_c, y_c, f_c = a[cal, label[cal]], label[cal], vfam[cal]
    out = []
    w_cal, w_test = split.weights("main", ex["clip"])
    pi = split.test_mass(w_cal, w_test)
    for cc in (False, True):
        out.append(("B1_split", cc, conformal_sets(a_c, y_c, f_c, a[te], alpha, "variant", cc), None))
        hcp = conformal_sets(a_c, y_c, f_c, a[te], alpha, "family", cc)
        out.append(("B3_pool", cc, hcp, None))
        out.append(("B3_sub1", cc, subsample_sets(a_c, y_c, f_c, a[te], alpha, 1, rng, cc), None))
        out.append(("B3_subR", cc, subsample_sets(a_c, y_c, f_c, a[te], alpha, ex["n_subsample"], rng, cc), None))
        out.append(("B4_gwcp", cc, gwcp_sets(a_c, y_c, f_c, split.bin.loc[f_c].to_numpy(), a[te], alpha, split.q, cc), None))

        pw = weighted_hcp(split, a, label, vfam, alpha, cc, w_cal, w_test)
        out.append(("P_w", cc, pw, None))
        m, _, sc = gate_params[("P_gate", cc)]
        out.append(("P_gate", cc, *_abstain(hcp, (split.ess_full[sc] < m).to_numpy(), tf)))
        m, tau, sc = gate_params[("P_full", cc)]
        fam_ab = gate(split.ess_conf[sc], pi, m, tau)
        out.append(("P_full", cc, *_abstain(pw, fam_ab, tf)))

        n_abs = int(fam_ab.sum())
        order = pd.DataFrame({"s": split.max_id, "t": split.tie}).sort_values(["s", "t"]).index
        sup = pd.Series(False, index=split.test)
        sup.loc[order[:n_abs]] = True
        out.append(("B5_support", cc, *_abstain(pw, sup.to_numpy(), tf)))
        rnd = pd.Series(False, index=split.test)
        rnd.loc[split.tie.sort_values().index[:n_abs]] = True
        out.append(("B5_random", cc, *_abstain(pw, rnd.to_numpy(), tf)))

        if ablations:
            for dset in ["seq", "depth", "emb", "support", SCORES_SET]:
                wc, wt = split.weights(dset, ex["clip"], pred)
                out.append((f"P_w[{dset}]", cc, weighted_hcp(split, a, label, vfam, alpha, cc, wc, wt), None))
            for clip in ex["clip_ablation"]:
                wc, wt = split.weights("main", clip)
                name = "P_w[noclip]" if clip is None else f"P_w[clip{clip:g}]"
                out.append((name, cc, weighted_hcp(split, a, label, vfam, alpha, cc, wc, wt), None))
    return out


def risk_scores(split, ex, scale):
    """Per-family abstention scores (higher = more support, retained first)."""
    w_cal, w_test = split.weights("main", ex["clip"])
    return {"ess": split.ess_conf[scale], "mass": -split.test_mass(w_cal, w_test),
            "support": split.max_id + 1e-9 * split.tie, "random": split.tie}


def _retained_cov(sets, label, code, n_fam, abstain, cc):
    keep = ~abstain
    cov = sets[np.arange(len(label)), label]
    if not cc:
        return np.nanmean(_fam_mean(cov, code, n_fam, keep))
    vals = [np.nanmean(_fam_mean(cov, code, n_fam, keep & (label == c))) if (keep & (label == c)).any() else np.nan
            for c in (0, 1)]
    return np.nanmin(vals)


def tune_gate(split, a, label, vfam, alpha, ex):
    """Choose the gate hyperparameters (m, tau, kernel scale) on an inner split of train families.

    The kernel scale minimizes the inner risk-coverage AURC of the ESS ranking
    on the ungated sets. Among (m, tau) settings whose inner abstention rate is
    within the pre-declared budget ex["gate_budget"], the retained
    family-weighted coverage (class-conditional versions: the minimum over the
    two classes) is computed; the least-abstaining setting within
    ex["gate_cov_tol"] of the best retained coverage is chosen.
    """
    cal, te = np.isin(vfam, split.cal), np.isin(vfam, split.test)
    tf, yt = np.searchsorted(split.test, vfam[te]), label[te]
    n_fam = len(split.test)
    a_c, y_c, f_c = a[cal, label[cal]], label[cal], vfam[cal]
    w_cal, w_test = split.weights("main", ex["clip"])
    pi = split.test_mass(w_cal, w_test)
    params, rows = {}, []
    for cc in (False, True):
        hcp = conformal_sets(a_c, y_c, f_c, a[te], alpha, "family", cc)
        pw = weighted_hcp(split, a, label, vfam, alpha, cc, w_cal, w_test)
        for method, sets, ess_by, taus in (("P_gate", hcp, split.ess_full, [np.inf]), ("P_full", pw, split.ess_conf, ex["gate_tau_grid"])):
            aurc = {sc: risk_coverage(sets, yt, vfam[te], ess_by[sc].set_axis(split.test), RISK_GRID)[1] for sc in ess_by}
            sc = min(aurc, key=lambda k: (aurc[k], k))
            ess = ess_by[sc]
            cands = []
            for m in ex["gate_m_grid"]:
                for tau in taus:
                    fam_ab = gate(ess, pi, m, tau)
                    ab = fam_ab[tf]
                    rate = fam_ab.mean()
                    cov = _retained_cov(sets, yt, tf, n_fam, ab, cc) if rate < 1 else np.nan
                    cands.append((m, tau, rate, cov))
                    rows.append({"method": method, "cc": cc, "scale": sc, "aurc": aurc[sc], "m": m, "tau": tau,
                                 "abstain_rate": rate, "ret_cov": cov})
            within = [c for c in cands if c[2] <= ex["gate_budget"] and not np.isnan(c[3])]
            top = max(c[3] for c in within)
            ok = [c for c in within if c[3] >= top - ex["gate_cov_tol"]]
            best = min(ok, key=lambda c: (c[2], c[0], -c[1]))
            params[(method, cc)] = (best[0], best[1], sc)
    return params, pd.DataFrame(rows)
