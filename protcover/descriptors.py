"""Label-free family descriptors.

Nothing here reads the label column. Score summaries use the predictor scores
of all scored variants of a family, which are available at test time.
"""
import numpy as np
import pandas as pd

SCORE_STATS = ("mean", "sd", "q10", "q50", "q90")


def _score_summaries(variants, fam_col, preds):
    g = variants.groupby(fam_col)
    parts = []
    for p in preds:
        s = g[p]
        parts.append(pd.DataFrame({
            f"{p}_mean": s.mean(), f"{p}_sd": s.std(ddof=0),
            f"{p}_q10": s.quantile(0.1), f"{p}_q50": s.quantile(0.5), f"{p}_q90": s.quantile(0.9),
        }))
    return pd.concat(parts, axis=1)


def load_embeddings(path):
    z = np.load(path, allow_pickle=False)
    e = z["embedding"].astype(float)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return pd.Series(list(e), index=z["protein_id"])


def embedding_regions(emb, fam, fam_col, n_regions, n_pcs, seed=0):
    """Label-free regions of protein space, one per family: k-means on the leading
    principal components of the family's mean embedding (unit-normalized proteins
    averaged over members, then renormalized, i.e. cosine geometry)."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    e = np.vstack(emb.loc[fam["protein_id"]].to_numpy())
    f = pd.DataFrame(e, index=fam[fam_col].to_numpy()).groupby(level=0).mean()
    x = f.to_numpy()
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    x = PCA(n_components=n_pcs, random_state=seed).fit_transform(x - x.mean(0))
    labels = KMeans(n_clusters=n_regions, n_init=10, random_state=seed).fit_predict(x)
    return pd.Series(labels, index=f.index, name="region")


def emb_dist_to_ref(emb, query_ids, ref_ids):
    """Cosine distance from each query protein to its nearest reference protein."""
    q = np.vstack(emb.loc[list(query_ids)].to_numpy())
    r = np.vstack(emb.loc[list(ref_ids)].to_numpy())
    return pd.DataFrame({"protein_id": list(query_ids), "emb_dist_to_train": 1.0 - (q @ r.T).max(axis=1)})


def family_descriptors(variants, proteins, fam, fam_col, preds, max_id=None, emb_dist=None):
    """One row per family. max_id / emb_dist: optional protein-level tables of
    max identity and embedding distance to the train proteins."""
    prot = proteins[["protein_id", "length", "msa_neff", "msa_neff_L"]].merge(fam[["protein_id", fam_col]], on="protein_id")
    if max_id is not None:
        prot = prot.merge(max_id, on="protein_id", how="left")
    if emb_dist is not None:
        prot = prot.merge(emb_dist, on="protein_id", how="left")
    prot["log_length"] = np.log(prot["length"])
    prot["log_neff"] = np.log1p(prot["msa_neff"])
    prot["log_neff_L"] = np.log(prot["msa_neff_L"] + 1e-3)
    g = prot.groupby(fam_col)
    d = pd.DataFrame({
        "n_proteins": g.size(),
        "log_length": g["log_length"].mean(),
        "log_neff": g["log_neff"].mean(),
        "log_neff_L": g["log_neff_L"].mean(),
    })
    if max_id is not None:
        d["max_id_to_train"] = g["max_id_to_train"].max()
    if emb_dist is not None:
        d["emb_dist_to_train"] = g["emb_dist_to_train"].min()
    v = variants[["protein_id"] + preds].merge(fam[["protein_id", fam_col]], on="protein_id")
    d["log_n_variants"] = np.log(v.groupby(fam_col).size())
    d = d.join(_score_summaries(v, fam_col, preds))
    d.index.name = "family"
    return d


def support_z(desc, cols=("max_id_to_train", "log_neff")):
    """Standardized support score: mean of z-scored support descriptors (higher = better supported)."""
    x = desc[list(cols)]
    return ((x - x.mean()) / x.std(ddof=0)).mean(axis=1)
