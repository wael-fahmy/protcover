"""Mean-pooled ESM2 embeddings of the unseen-protein test set (inference only, CPU).

Exactly the procedure of scripts/embed_esm2.py, whose functions are imported:
model esm2_t12_35M_UR50D (fair-esm 2.0.0), last layer (12); sequences longer than
1022 residues are embedded in windows of 1022 with stride 511, each residue's
representation is averaged over the windows that cover it, then mean pooled.

Reads data/processed/unseen_proteins.parquet (scripts/build_unseen.py) and writes
data/esm2_t12_35M_meanpool_unseen.npz with the keys of the benchmark file
(protein_id, embedding, model, layer) plus sequence_sha1, so a rerun can tell
whether the stored embeddings still match the sequences. Progress is
checkpointed in data/unseen/esm2_partial.npz, so an interrupted run resumes.
Before embedding, a few benchmark proteins (short and longer than 1022 residues)
are re-embedded and compared with data/esm2_t12_35M_meanpool.npz; the check is
written to data/unseen/esm2_embed_summary.json.

Usage: python -I scripts/embed_esm2_unseen.py [--threads 4] [--force]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_esm2 as E  # noqa: E402

from protcover.config import ROOT, load_config, processed_dir  # noqa: E402
from protcover.data import load_proteins  # noqa: E402

OUT = ROOT / "data" / "esm2_t12_35M_meanpool_unseen.npz"
PARTIAL = ROOT / "data" / "unseen" / "esm2_partial.npz"
SUMMARY = ROOT / "data" / "unseen" / "esm2_embed_summary.json"


def sha1(s):
    return hashlib.sha1(s.encode()).hexdigest()


def save(path, ids, embs, shas):
    order = np.argsort(np.asarray(ids))
    np.savez_compressed(path, protein_id=np.asarray(ids)[order].astype(str),
                        embedding=np.asarray(embs, dtype=np.float32)[order], model=E.MODEL, layer=E.LAYER,
                        sequence_sha1=np.asarray(shas)[order].astype(str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    cfg = load_config()
    prot = pd.read_parquet(processed_dir(cfg) / "unseen_proteins.parquet").sort_values(["length", "protein_id"])
    want = dict(zip(prot["protein_id"], prot["sequence"].map(sha1)))

    if OUT.exists() and not args.force:
        z = np.load(OUT)
        if "sequence_sha1" in z.files and dict(zip(z["protein_id"], z["sequence_sha1"])) == want:
            print(f"{OUT} is up to date ({len(want)} proteins)")
            return

    model, alphabet = getattr(E.esm.pretrained, E.MODEL)()
    model.eval()
    converter = alphabet.get_batch_converter()

    # Reproduce a few benchmark embeddings first.
    bp = load_proteins(cfg)
    ref = np.load(ROOT / "data" / "esm2_t12_35M_meanpool.npz")
    ref_emb = dict(zip(ref["protein_id"], ref["embedding"]))
    chk = pd.concat([bp[bp["length"] <= 1022].sort_values("protein_id").head(4),
                     bp[bp["length"] > 1022].sort_values("length").head(2)])
    diffs = {}
    for pid, seq in zip(chk["protein_id"], chk["sequence"]):
        e = E.embed(model, converter, seq)
        diffs[pid] = {"length": len(seq), "max_abs_diff": float(np.abs(e - ref_emb[pid]).max()),
                      "cosine": float(e @ ref_emb[pid] / np.linalg.norm(e) / np.linalg.norm(ref_emb[pid]))}
    print(json.dumps(diffs, indent=1), flush=True)
    assert all(d["max_abs_diff"] < 1e-3 for d in diffs.values()), "benchmark embeddings not reproduced"

    done = {}
    if PARTIAL.exists() and not args.force:
        z = np.load(PARTIAL)
        done = {p: (e, s) for p, e, s in zip(z["protein_id"], z["embedding"], z["sequence_sha1"]) if want.get(p) == s}
    print(f"{len(done)} of {len(prot)} proteins already embedded", flush=True)
    t0, n_new = time.time(), 0
    for pid, seq in zip(prot["protein_id"], prot["sequence"]):
        if pid in done:
            continue
        done[pid] = (E.embed(model, converter, seq), want[pid])
        n_new += 1
        if n_new % 250 == 0:
            ids = list(done)
            save(PARTIAL, ids, [done[p][0] for p in ids], [done[p][1] for p in ids])
            print(f"{len(done)}/{len(prot)} ({time.time() - t0:.0f}s)", flush=True)
    ids = list(want)
    save(OUT, ids, [done[p][0] for p in ids], [done[p][1] for p in ids])
    PARTIAL.unlink(missing_ok=True)
    summary = {"model": E.MODEL, "layer": E.LAYER, "window": E.WINDOW, "stride": E.STRIDE, "threads": args.threads,
               "n_proteins": len(ids), "n_over_1022": int((prot["length"] > 1022).sum()),
               "benchmark_reproduction_check": diffs, "seconds_this_run": round(time.time() - t0, 1),
               "torch": torch.__version__, "output": str(OUT.relative_to(ROOT)).replace("\\", "/")}
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {OUT} ({len(ids)} proteins) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
