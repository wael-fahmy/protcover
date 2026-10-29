"""Base model: per-predictor logistic regression on train families.

Raw scores are first clipped to the [0.1%, 99.9%] quantiles of the train
variants' scores (label-free, train families only). This guards against a few
extreme values in ProteinGym's files: for example two EVE scores near 5e5
against a 99.9th percentile near 20, which otherwise dominate the fit.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLIP_Q = 0.001


def within_protein_z(protein_id, score):
    """Score standardized within its protein over all scored variants (label-free)."""
    import pandas as pd

    s = pd.Series(np.asarray(score, float))
    g = s.groupby(np.asarray(protein_id))
    mu, sd = g.transform("mean"), g.transform("std").fillna(0.0)
    z = (s - mu) / sd.where(sd > 0, 1.0)
    return z.where(sd > 0, 0.0).to_numpy()


def clip_bounds(score, train_mask, q=CLIP_Q):
    return np.quantile(np.asarray(score)[train_mask], [q, 1 - q])


def features(variants, pred, bounds):
    s = np.clip(variants[pred].to_numpy(float), *bounds)
    return np.column_stack([s, within_protein_z(variants["protein_id"].to_numpy(), s)])


def fit(variants, pred, train_mask):
    bounds = clip_bounds(variants[pred].to_numpy(), train_mask)
    x = features(variants, pred, bounds)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    model.fit(x[train_mask], variants["label"].to_numpy()[train_mask])
    return model, bounds


def fit_predict(variants, pred, train_mask):
    """Fit on train-family variants; return p_hat(pathogenic | x) for every variant."""
    model, bounds = fit(variants, pred, train_mask)
    return model.predict_proba(features(variants, pred, bounds))[:, 1]


def nonconformity(p1):
    """a(x, y) = 1 - p_hat(y | x) for y = 0 and y = 1; shape (n, 2)."""
    p1 = np.asarray(p1)
    return np.column_stack([p1, 1.0 - p1])
