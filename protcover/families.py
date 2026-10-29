"""MMseqs2 wrappers: family clustering and max identity to a reference set."""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from .config import ROOT


def mmseqs_bin():
    """Locate MMseqs2: $MMSEQS, then tools/mmseqs/bin, then PATH."""
    if os.environ.get("MMSEQS"):
        return os.environ["MMSEQS"]
    for name in ("mmseqs.exe", "mmseqs"):
        p = ROOT / "tools" / "mmseqs" / "bin" / name
        if p.exists():
            return str(p)
    p = shutil.which("mmseqs")
    if p is None:
        raise FileNotFoundError("MMseqs2 not found; run data/download.sh or set $MMSEQS")
    return p


def _run(args, threads):
    exe = mmseqs_bin()
    env = os.environ.copy()
    # On Windows the workflow scripts need the bundled BusyBox tools on PATH.
    env["PATH"] = str(Path(exe).parent) + os.pathsep + env.get("PATH", "")
    cmd = [exe] + [str(a) for a in args] + ["--threads", str(threads), "-v", "1"]
    subprocess.run(cmd, check=True, env=env)


def write_fasta(ids, seqs, path):
    with open(path, "w") as f:
        for i, s in zip(ids, seqs):
            f.write(f">{i}\n{s}\n")


def _createdb(fasta, db):
    _run(["createdb", fasta, db], threads=1)


def cluster(proteins, min_seq_id, coverage, cov_mode=0, threads=8):
    """Cluster wild-type sequences; returns protein_id -> family_id (representative).

    Runs createdb, cluster and createtsv, which is what easy-cluster does apart
    from writing representative FASTA files (that step crashes in the Windows
    build of release 18-8cc5c, and we do not need it).
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_fasta(proteins["protein_id"], proteins["sequence"], tmp / "seqs.fasta")
        _createdb(tmp / "seqs.fasta", tmp / "db")
        _run(["cluster", tmp / "db", tmp / "clu", tmp / "work",
              "--min-seq-id", min_seq_id, "-c", coverage, "--cov-mode", cov_mode], threads)
        _run(["createtsv", tmp / "db", tmp / "db", tmp / "clu", tmp / "clu.tsv"], threads)
        clu = pd.read_csv(tmp / "clu.tsv", sep="\t", header=None, names=["family_id", "protein_id"])
    return clu[["protein_id", "family_id"]]


HIT_COLUMNS = ["query", "target", "fident", "alnlen", "qcov", "tcov", "evalue", "bits"]


def search(query, target, sensitivity=7.5, evalue=10.0, threads=8, max_seqs=500):
    """MMseqs2 search of query proteins against target proteins (no coverage filter)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_fasta(query["protein_id"], query["sequence"], tmp / "q.fasta")
        write_fasta(target["protein_id"], target["sequence"], tmp / "t.fasta")
        _createdb(tmp / "q.fasta", tmp / "qdb")
        _createdb(tmp / "t.fasta", tmp / "tdb")
        _run(["search", tmp / "qdb", tmp / "tdb", tmp / "aln", tmp / "work",
              "-s", sensitivity, "-e", evalue, "--max-seqs", max_seqs], threads)
        _run(["convertalis", tmp / "qdb", tmp / "tdb", tmp / "aln", tmp / "hits.m8",
              "--format-output", ",".join(HIT_COLUMNS)], threads)
        hits = pd.read_csv(tmp / "hits.m8", sep="\t", header=None, names=HIT_COLUMNS)
    return hits[hits["query"] != hits["target"]].reset_index(drop=True)


def all_vs_all(proteins, sensitivity=7.5, evalue=10.0, threads=8):
    return search(proteins, proteins, sensitivity, evalue, threads)


def max_identity(query, target, sensitivity=7.5, evalue=1e-3, threads=8):
    """Max identity of each query protein to any target protein over significant hits.

    No coverage filter, so shared domains count as support. Queries without a
    hit at E <= evalue get identity 0. Callers pass disjoint query and target sets.
    """
    hits = search(query, target, sensitivity, evalue, threads)
    best = hits.groupby("query")["fident"].max()
    out = query[["protein_id"]].copy()
    out["max_id_to_train"] = out["protein_id"].map(best).fillna(0.0).to_numpy()
    return out


def orphan_flags(fam, col, hits, evalue, min_id):
    """True for families with no significant hit (E <= evalue, identity >= min_id)
    from any member protein to a protein outside the family."""
    f = fam.set_index("protein_id")[col]
    h = hits[(hits["evalue"] <= evalue) & (hits["fident"] >= min_id)]
    cross = h[h["query"].map(f).to_numpy() != h["target"].map(f).to_numpy()]
    has_homolog = set(cross["query"].map(f))
    fams = pd.Index(f.unique())
    return pd.Series(~fams.isin(list(has_homolog)), index=fams)
