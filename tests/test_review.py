"""Splits and subsampling added after the second external review."""
import numpy as np
import pandas as pd

from protcover.pipeline import one_per_family
from protcover.splits import split_sizes, strict_region_split


def test_strict_region_split_holds_out_whole_regions_only():
    rng = np.random.default_rng(0)
    fam = np.arange(500)
    region = pd.Series(rng.integers(0, 20, len(fam)), index=fam)
    train, cal, test = strict_region_split(fam, region, 0.4, 0.3, np.random.default_rng(1))
    assert len(test) <= split_sizes(len(fam), 0.4, 0.3)[2]
    for r in region.loc[test].unique():  # every region in the test set is there in full
        assert set(region.index[region == r]) <= set(test)
    assert not (set(train) & set(cal)) and not (set(train) & set(test)) and not (set(cal) & set(test))
    assert set(train) | set(cal) | set(test) == set(fam)


def test_one_per_family_keeps_one_variant_per_family_and_class():
    rng = np.random.default_rng(0)
    vfam = rng.integers(0, 50, 2000)
    label = rng.integers(0, 2, 2000)
    sub = one_per_family(vfam, label, rng)
    assert np.array_equal(np.sort(vfam[sub[None]]), np.unique(vfam))
    for y in (0, 1):
        assert (label[sub[y]] == y).all()
        assert np.array_equal(np.sort(vfam[sub[y]]), np.unique(vfam[label == y]))
