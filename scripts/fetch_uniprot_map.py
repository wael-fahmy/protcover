"""Map ProteinGym clinical RefSeq protein IDs to UniProtKB accessions.

Uses the UniProt ID mapping REST API (RefSeq_Protein -> UniProtKB). The result
is written to data/uniprot_map.tsv, which is committed so that the pipeline does
not depend on the API being available. Only public RefSeq identifiers are sent.

Usage: python scripts/fetch_uniprot_map.py
"""
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "raw" / "clinical_substitutions_reference.csv"
OUT = ROOT / "data" / "uniprot_map.tsv"
API = "https://rest.uniprot.org/idmapping"


def submit(ids):
    r = requests.post(f"{API}/run", data={"from": "RefSeq_Protein", "to": "UniProtKB", "ids": ",".join(ids)}, timeout=60)
    r.raise_for_status()
    return r.json()["jobId"]


def wait(job):
    while True:
        r = requests.get(f"{API}/status/{job}", timeout=60, allow_redirects=False)
        r.raise_for_status()
        js = r.json()
        if r.status_code == 303 or js.get("jobStatus") == "FINISHED" or "results" in js:
            return
        if js.get("jobStatus") not in ("NEW", "RUNNING"):
            raise RuntimeError(f"ID mapping job {job} failed: {js}")
        time.sleep(2)


def fetch(job):
    url = f"{API}/uniprotkb/results/stream/{job}"
    params = {"format": "tsv", "fields": "accession,id,gene_primary,reviewed,organism_id,length"}
    r = requests.get(url, params=params, timeout=300)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    header = lines[0].split("\t")
    rows = [line.split("\t") for line in lines[1:]]
    return pd.DataFrame(rows, columns=header)


def main():
    ids = pd.read_csv(REF)["DMS_id"].tolist()
    job = submit(ids)
    wait(job)
    df = fetch(job)
    df = df.rename(columns={"From": "protein_id", "Entry": "uniprot", "Entry Name": "uniprot_entry_name",
                            "Gene Names (primary)": "gene", "Reviewed": "reviewed", "Organism (ID)": "taxon",
                            "Length": "uniprot_length"})
    df["uniprot_length"] = pd.to_numeric(df["uniprot_length"])
    # One RefSeq ID can map to several entries; prefer reviewed human entries.
    df["is_reviewed"] = df["reviewed"].eq("reviewed")
    df["is_human"] = df["taxon"].eq("9606")
    df = df.sort_values(["protein_id", "is_reviewed", "is_human"], ascending=[True, False, False])
    df["n_mapped"] = df.groupby("protein_id")["uniprot"].transform("size")
    df = df.drop_duplicates("protein_id").drop(columns=["is_reviewed", "is_human"])
    df.to_csv(OUT, sep="\t", index=False)
    print(f"mapped {len(df)} of {len(ids)} RefSeq IDs -> {OUT}")


if __name__ == "__main__":
    main()
