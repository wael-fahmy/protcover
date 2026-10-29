"""Family-level splits: S1 (exchangeable), S2 (covariate-shifted), S3 (orphan hold-out),
S4 (held-out regions of embedding space).

All splits assign whole families. Proportions are by family count.
"""
import numpy as np
import pandas as pd


def split_sizes(n, frac_train, frac_cal):
    n_train = int(round(frac_train * n))
    n_cal = int(round(frac_cal * n))
    return n_train, n_cal, n - n_train - n_cal


def draw_train(families, frac_train, rng):
    """Uniformly random train families (shared by S1 and every S2 beta for a seed)."""
    fam = np.asarray(sorted(families))
    n_train = int(round(frac_train * len(fam)))
    return np.sort(rng.choice(fam, size=n_train, replace=False))


def shifted_cal_test(nontrain, z, n_test, u, beta):
    """Weighted sampling without replacement (Efraimidis and Spirakis keys).

    Test families are drawn with weight exp(-beta * z), so poorly supported
    families (low z) are favoured; the rest become calibration families. The
    uniforms u are shared across beta, so beta = 0 is a uniform random split
    (S1) and the splits are coupled across the sweep.
    """
    nontrain = np.asarray(nontrain)
    keys = np.log(u) * np.exp(beta * np.asarray(z))
    order = np.argsort(-keys, kind="stable")
    test = np.sort(nontrain[order[:n_test]])
    cal = np.sort(nontrain[order[n_test:]])
    return cal, test


def region_split(families, region, frac_train, frac_cal, rng):
    """S4: test families are whole embedding regions, taken in random order until
    the test size is reached (the last region is split at random); train and
    calibration families are drawn uniformly from the remaining families."""
    fam = np.asarray(sorted(families))
    n_train, n_cal, n_test = split_sizes(len(fam), frac_train, frac_cal)
    reg = region.loc[fam]
    test = []
    for r in rng.permutation(np.unique(reg.to_numpy())):
        members = rng.permutation(reg.index[reg.to_numpy() == r].to_numpy())
        test.extend(members[: n_test - len(test)])
        if len(test) == n_test:
            break
    test = np.sort(np.asarray(test))
    rest = np.setdiff1d(fam, test)
    train = np.sort(rng.choice(rest, size=n_train, replace=False))
    cal = np.setdiff1d(rest, train)
    return train, cal, test


def orphan_split(families, orphan, frac_train, frac_cal, rng):
    """S3: test families are drawn uniformly from orphan families; train and
    calibration families are drawn uniformly from the remaining families."""
    fam = np.asarray(sorted(families))
    n_train, n_cal, n_test = split_sizes(len(fam), frac_train, frac_cal)
    pool = np.asarray(sorted(orphan[orphan].index))
    if len(pool) < n_test:
        raise ValueError(f"only {len(pool)} orphan families for {n_test} test families")
    test = np.sort(rng.choice(pool, size=n_test, replace=False))
    rest = np.setdiff1d(fam, test)
    train = np.sort(rng.choice(rest, size=n_train, replace=False))
    cal = np.setdiff1d(rest, train)
    return train, cal, test
