"""Build the tidy variant table and the per-protein table from ProteinGym files."""
import numpy as np
import pandas as pd

from .config import ROOT, predictor_names, processed_dir, raw_dir

LABEL_MAP = {"Pathogenic": 1, "Benign": 0}


def _gene_from_msa_filename(name):
    # "refseq-ACADM-NM_000016.5.a2m" -> "ACADM"; "SGCA_HUMAN_b03.a2m" -> None
    if name.startswith("refseq-"):
        return name.split("-")[1]
    return None


def build_variants(cfg):
    """One row per (protein, variant). Scores are oriented: higher = more pathogenic."""
    rdir = raw_dir(cfg)
    ref = pd.read_csv(rdir / "clinical_substitutions_reference.csv")
    preds = predictor_names(cfg)
    rows = []
    for pid in ref["DMS_id"]:
        s = pd.read_csv(rdir / "clinical_scores" / f"{pid}.csv", usecols=["protein", "mutant", "DMS_bin_score"] + preds)
        v = pd.read_csv(rdir / "clinical_substitutions" / f"{pid}.csv", usecols=["protein", "mutant", "DMS_bin_score"])
        # The score file must describe exactly the benchmark variants with the same labels.
        m = s.merge(v, on=["protein", "mutant"], how="outer", suffixes=("", "_bench"), indicator=True)
        if not (m["_merge"] == "both").all() or not (m["DMS_bin_score"] == m["DMS_bin_score_bench"]).all():
            raise ValueError(f"score file and benchmark file disagree for {pid}")
        rows.append(s)
    df = pd.concat(rows, ignore_index=True)
    df = df.rename(columns={"protein": "protein_id", "mutant": "variant"})
    df["label"] = df.pop("DMS_bin_score").map(LABEL_MAP).astype(int)
    for p in preds:
        df[p] = -cfg["predictors"][p]["directionality"] * df[p].astype(float)
    df["position"] = df["variant"].str[1:-1].astype(int)

    umap = pd.read_csv(ROOT / "data" / "uniprot_map.tsv", sep="\t")[["protein_id", "uniprot"]]
    df = df.merge(umap, on="protein_id", how="left")
    cols = ["protein_id", "uniprot", "variant", "position", "label"] + preds
    return df[cols].sort_values(["protein_id", "position", "variant"]).reset_index(drop=True)


def build_proteins(cfg, variants):
    """Per-protein label-free metadata: sequence, length, MSA depth, variant count."""
    rdir = raw_dir(cfg)
    ref = pd.read_csv(rdir / "clinical_substitutions_reference.csv")
    umap = pd.read_csv(ROOT / "data" / "uniprot_map.tsv", sep="\t")
    n_seqs, neff = [], []
    for wf in ref["weight_file_name"]:
        w = np.load(rdir / "clinical_msa_weights" / "subs" / wf)
        n_seqs.append(len(w))
        neff.append(float(w.sum()))
    prot = pd.DataFrame({
        "protein_id": ref["DMS_id"],
        "sequence": ref["target_seq"],
        "length": ref["target_seq"].str.len(),
        "msa_len": ref["MSA_len"],
        "msa_n_seqs": n_seqs,
        "msa_neff": neff,
        "alignment_source": ref["alignment_source"],
        "gene_from_msa": ref["MSA_filename"].map(_gene_from_msa_filename),
    })
    # Neff per covered position; ProteinGym's MSA_N_eff_L uses the number of
    # well-covered columns, which the clinical reference file does not provide,
    # so we normalise by the MSA length instead.
    prot["msa_neff_L"] = prot["msa_neff"] / prot["msa_len"]
    prot = prot.merge(umap[["protein_id", "uniprot", "uniprot_entry_name", "gene"]], on="protein_id", how="left")
    prot["gene"] = prot["gene"].fillna(prot["gene_from_msa"])
    prot = prot.drop(columns="gene_from_msa")
    prot = prot.merge(variants.groupby("protein_id").size().rename("n_variants").reset_index(), on="protein_id")
    return prot.sort_values("protein_id").reset_index(drop=True)


def load_variants(cfg):
    return pd.read_parquet(processed_dir(cfg) / "variants.parquet")


def load_proteins(cfg):
    return pd.read_parquet(processed_dir(cfg) / "proteins.parquet")
