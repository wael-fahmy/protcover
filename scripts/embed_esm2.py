"""Mean-pooled ESM2 embeddings of wild-type sequences (inference only, CPU).

Model: esm2_t12_35M_UR50D (fair-esm 2.0.0), last layer (12). Sequences longer
than 1022 residues are embedded in windows of 1022 with stride 511; each
residue's representation is averaged over the windows that cover it before
mean pooling over residues.

Writes data/esm2_t12_35M_meanpool.npz (protein_id, embedding), which is
committed so the rest of the pipeline does not need PyTorch.

Usage: python scripts/embed_esm2.py   (needs: pip install -r requirements-embed.txt)
"""
import time

import numpy as np
import torch
import esm

from protcover.config import ROOT, load_config
from protcover.data import load_proteins

MODEL = "esm2_t12_35M_UR50D"
LAYER = 12
WINDOW, STRIDE = 1022, 511
OUT = ROOT / "data" / "esm2_t12_35M_meanpool.npz"


def windows(n):
    if n <= WINDOW:
        return [(0, n)]
    starts = list(range(0, n - WINDOW, STRIDE)) + [n - WINDOW]
    return [(s, s + WINDOW) for s in starts]


@torch.no_grad()
def embed(model, converter, seq):
    acc = torch.zeros(len(seq), model.embed_dim)
    cnt = torch.zeros(len(seq), 1)
    for s, e in windows(len(seq)):
        _, _, toks = converter([("q", seq[s:e])])
        rep = model(toks, repr_layers=[LAYER])["representations"][LAYER][0, 1:e - s + 1]
        acc[s:e] += rep
        cnt[s:e] += 1
    return (acc / cnt).mean(0).numpy()


def main():
    torch.manual_seed(0)
    cfg = load_config()
    prot = load_proteins(cfg).sort_values("length")
    model, alphabet = getattr(esm.pretrained, MODEL)()
    model.eval()
    converter = alphabet.get_batch_converter()
    t0 = time.time()
    embs = []
    for i, seq in enumerate(prot["sequence"]):
        embs.append(embed(model, converter, seq))
        if (i + 1) % 250 == 0:
            print(f"{i + 1}/{len(prot)} ({time.time() - t0:.0f}s)", flush=True)
    order = np.argsort(prot["protein_id"].to_numpy())
    np.savez_compressed(OUT, protein_id=prot["protein_id"].to_numpy()[order].astype(str),
                        embedding=np.asarray(embs, dtype=np.float32)[order], model=MODEL, layer=LAYER)
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
