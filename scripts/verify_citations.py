"""Check every entry of paper/refs.bib against Crossref (DOI) or the arXiv API.

For each entry, the registered title and first-author family name are fetched
and compared with the .bib entry. Writes results/citation_check.csv and exits
non-zero if any entry fails to resolve or mismatches.

Usage: python scripts/verify_citations.py
"""
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from protcover.config import ROOT, results_dir

BIB = ROOT / "paper" / "refs.bib"
UA = {"User-Agent": "protcover-citation-check/0.1 (mailto:wfahmy@lakeheadu.ca)"}


def parse_bib(text):
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        fields = dict(re.findall(r"(\w+)\s*=\s*\{(.*?)\},?\s*\n", m.group(3) + "\n", re.S))
        entries.append({"key": m.group(2).strip(), **{k.lower(): v for k, v in fields.items()}})
    return entries


def norm(s):
    s = re.sub(r"\\[`'\"^~v.]\{?(\w)\}?", r"\1", s or "")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.replace("{", "").replace("}", "").lower())


def first_family(author_field):
    first = author_field.split(" and ")[0]
    return first.split(",")[0] if "," in first else first.split()[-1]


def crossref(doi):
    r = requests.get(f"https://api.crossref.org/works/{doi}", headers=UA, timeout=60)
    if r.status_code != 200:
        return None
    it = r.json()["message"]
    return (it.get("title") or [""])[0], (it.get("author") or [{}])[0].get("family", "")


def arxiv(eid, tries=5):
    # The arXiv API asks for about 3 s between calls and answers with plain text when rate-limited.
    for k in range(tries):
        r = requests.get("https://export.arxiv.org/api/query", params={"id_list": eid}, headers=UA, timeout=60)
        if r.status_code == 200 and r.text.lstrip().startswith("<"):
            break
        time.sleep(5 * (k + 1))
    else:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = ET.fromstring(r.text).find("a:entry", ns)
    if entry is None or entry.find("a:title", ns) is None:
        return None
    name = entry.find("a:author/a:name", ns).text
    return " ".join(entry.find("a:title", ns).text.split()), name.split()[-1]


def main():
    rows = []
    for e in parse_bib(BIB.read_text(encoding="utf-8")):
        src, ident = ("doi", e["doi"]) if "doi" in e else ("arxiv", e.get("eprint"))
        got = (crossref(ident) if src == "doi" else arxiv(ident)) if ident else None
        time.sleep(3.0 if src == "arxiv" else 0.5)
        if got is None:
            rows.append({"key": e["key"], "source": src, "id": ident, "status": "NOT FOUND"})
            continue
        title, fam = got
        t_ok = norm(e.get("title")) == norm(title) or norm(e.get("title")) in norm(title) or norm(title) in norm(e.get("title"))
        # Some registry records (e.g. books) list no authors; then only the title is checked.
        a_ok = norm(first_family(e.get("author", ""))) == norm(fam) if fam else None
        status = "OK" if t_ok and a_ok else ("OK (title only, registry lists no author)" if t_ok and a_ok is None else "MISMATCH")
        rows.append({"key": e["key"], "source": src, "id": ident, "status": status,
                     "title_match": t_ok, "author_match": a_ok, "registered_title": title, "registered_first_author": fam})
    df = pd.DataFrame(rows)
    df.to_csv(results_dir() / "citation_check.csv", index=False)
    print(df[["key", "status", "title_match", "author_match", "registered_first_author"]].to_string(index=False))
    bad = df[~df["status"].str.startswith("OK")]
    if len(bad):
        print(f"\n{len(bad)} entries need attention")
        sys.exit(1)


if __name__ == "__main__":
    main()
