"""ESM2 zero-shot scores of the benchmark variants and of the new-protein test set (inference only, CPU).

A second predictor for the evaluation on proteins outside the benchmark, computed with one score
function for every protein on both sides. Model esm2_t30_150M_UR50D (fair-esm 2.0.0). Score: the
wild-type marginal log-likelihood ratio, log p(mutant) - log p(wild type) at the variant's position
given the unmasked wild-type sequence (one forward pass per window; Meier et al. 2021). Sequences
longer than 1022 residues are tiled in windows of 1022 with stride 511, as for the embeddings
(scripts/embed_esm2.py), and the log-probabilities of a position are averaged over the windows that
cover it. Only windows that cover a scored variant are computed, which leaves every score unchanged.
The oriented score is the negated ratio (higher = more pathogenic), as for ESM-1b.

Benchmark proteins use the benchmark sequence (data/processed/proteins.parquet); new proteins the
sequence scored by Brandes et al. (data/processed/unseen_proteins.parquet). The wild type is checked
at every position. No label is read.

Writes data/processed/esm2_150M_scores.parquet (one row per variant: set, protein_id, variant,
position, wt, mt, llr, oriented, n_windows) and data/unseen/esm2_150M_summary.json (model,
checkpoint SHA-256, versions, counts, runtime). Progress is checkpointed per protein in
data/processed/esm2_150M_partial/, so an interrupted run resumes.

Usage: python -I scripts/score_esm2.py [--threads 8]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_esm2 import windows  # noqa: E402

from protcover.config import ROOT, load_config, processed_dir  # noqa: E402

MODEL = "esm2_t30_150M_UR50D"
OUT_NAME = "esm2_150M_scores.parquet"
SUMMARY = ROOT / "data" / "unseen" / "esm2_150M_summary.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_tasks(pdir):
    """One row per variant to score, with its protein's sequence (benchmark first, then new proteins)."""
    bp = pd.read_parquet(pdir / "proteins.parquet")[["protein_id", "sequence"]]
    bv = pd.read_parquet(pdir / "variants.parquet")[["protein_id", "variant", "position"]]
    bv = bv.assign(set="benchmark", wt=bv["variant"].str[0], mt=bv["variant"].str[-1]).merge(bp, on="protein_id")
    up = pd.read_parquet(pdir / "unseen_proteins.parquet")[["protein_id", "sequence"]]
    uv = pd.read_parquet(pdir / "unseen_variants.parquet")[["protein_id", "mutant", "position", "wt", "mt"]]
    uv = uv.rename(columns={"mutant": "variant"}).assign(set="unseen").merge(up, on="protein_id")
    cols = ["set", "protein_id", "variant", "position", "wt", "mt", "sequence"]
    return pd.concat([bv[cols], uv[cols]], ignore_index=True)


@torch.inference_mode()
def score_protein(model, alphabet, converter, seq, rows):
    """Wild-type marginal LLR of each variant in rows (1-based positions), averaging log-probabilities
    over the covering windows of the fixed tiling; windows that cover no variant are skipped."""
    pos0 = rows["position"].to_numpy() - 1
    wins = [(s, e) for s, e in windows(len(seq)) if ((pos0 >= s) & (pos0 < e)).any()]
    acc = {}
    for s, e in wins:
        _, _, toks = converter([("q", seq[s:e])])
        logp = torch.log_softmax(model(toks)["logits"][0, 1:e - s + 1].float(), dim=-1).numpy()
        for p in np.unique(pos0[(pos0 >= s) & (pos0 < e)]):
            a, n = acc.get(p, (0.0, 0))
            acc[p] = (a + logp[p - s], n + 1)
    wt_idx = np.array([alphabet.get_idx(c) for c in rows["wt"]])
    mt_idx = np.array([alphabet.get_idx(c) for c in rows["mt"]])
    llr = np.array([(acc[p][0][m] - acc[p][0][w]) / acc[p][1] for p, w, m in zip(pos0, wt_idx, mt_idx)])
    nwin = np.array([acc[p][1] for p in pos0])
    return llr, nwin, len(wins), sum(e - s for s, e in wins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    pdir = processed_dir(load_config())
    part = pdir / "esm2_150M_partial"
    part.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(pdir)
    seq_ok = np.array([s[p - 1] == w if 0 < p <= len(s) else False
                       for s, p, w in zip(tasks["sequence"], tasks["position"], tasks["wt"])])
    assert seq_ok.all(), f"{(~seq_ok).sum()} variants whose wild type does not match the sequence"
    model, alphabet = getattr(esm.pretrained, MODEL)()
    model.eval()
    converter = alphabet.get_batch_converter()
    ckpt = Path(torch.hub.get_dir()) / "checkpoints" / f"{MODEL}.pt"
    groups = list(tasks.groupby(["set", "protein_id"], sort=False))
    t0, n_win, n_tok, done = time.time(), 0, 0, 0
    for i, ((st, pid), rows) in enumerate(groups):
        f = part / f"{st}__{pid}.parquet"
        if f.exists():
            done += 1
            continue
        llr, nwin, w, tok = score_protein(model, alphabet, converter, rows["sequence"].iloc[0], rows)
        n_win, n_tok = n_win + w, n_tok + tok
        rows.drop(columns="sequence").assign(llr=llr, oriented=-llr, n_windows=nwin).to_parquet(f, index=False)
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(groups)} proteins ({time.time() - t0:.0f}s, {n_tok:,} tokens this session)", flush=True)
    out = pd.concat([pd.read_parquet(part / f"{st}__{pid}.parquet") for (st, pid), _ in groups], ignore_index=True)
    assert len(out) == len(tasks) and out["llr"].notna().all()
    out.to_parquet(pdir / OUT_NAME, index=False)
    summary = {"model": MODEL, "checkpoint": str(ckpt.name), "checkpoint_sha256": sha256(ckpt) if ckpt.exists() else None,
               "fair_esm": getattr(esm, "__version__", "2.0.0"), "torch": torch.__version__, "threads": args.threads,
               "score": "wild-type marginal log-likelihood ratio; oriented = -llr",
               "window": 1022, "stride": 511, "n_variants": int(len(out)),
               "n_variants_by_set": out["set"].value_counts().to_dict(),
               "n_proteins_by_set": out.groupby("set")["protein_id"].nunique().to_dict(),
               "proteins_resumed": done, "windows_this_session": n_win, "tokens_this_session": n_tok,
               "seconds_this_session": round(time.time() - t0), "output_sha256": sha256(pdir / OUT_NAME)}
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
