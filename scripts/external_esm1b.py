"""Per-protein access to the Brandes et al. (2023) ESM-1b LLR archive by HTTP range requests.

The archive ALL_hum_isoforms_ESM1b_LLR.zip (1.3 GB, one CSV per UniProt
canonical sequence or isoform) lives in the Hugging Face Space
ntranoslab/esm_variants. Instead of downloading it whole, we read its central
directory once and then fetch only the entries we need. Every fetched entry is
checked against the CRC-32 stored in the central directory, and cached under
data/raw/external/esm1b/.

Each CSV is a 20 x L table: rows are the mutant amino acid, columns are
"<wild type> <position>" (1-based), values are the log-likelihood ratio
log p(mut) - log p(wt). The wild-type sequence is read off the column headers.
"""
import io
import struct
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

HF_SPACE = "ntranoslab/esm_variants"
# Pinned Space revision (commit of 2023-10-24, current on 2026-10-29).
HF_REVISION = "90e7e89167f7ba5ccce395855e3bc8dc258fa1f4"
ZIP_NAME = "ALL_hum_isoforms_ESM1b_LLR.zip"
ZIP_SIZE = 1335361543
# Git LFS SHA-256 of the archive, as reported by the Hub (X-Linked-ETag).
ZIP_LFS_SHA256 = "42b90ce74a007ce7147f8e4be4035fe7f11881215fecce0983441cc229207cd0"
BASE = f"https://huggingface.co/spaces/{HF_SPACE}/resolve/{HF_REVISION}"
ZIP_URL = f"{BASE}/{ZIP_NAME}"
ISOFORM_LIST_URL = f"{BASE}/isoform_list.csv"


def _get(url, start=None, end=None, tries=5):
    headers = {} if start is None else {"Range": f"bytes={start}-{end}"}
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                data = r.read()
            if start is not None and len(data) != end - start + 1:
                raise IOError(f"short range read {len(data)} != {end - start + 1}")
            return data
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


class BrandesLLR:
    def __init__(self, cache_dir):
        self.cache = cache_dir
        (self.cache / "llr").mkdir(parents=True, exist_ok=True)
        self.entries = self._central_directory()
        self.isoform_list = self._isoform_list()
        self.by_base = {}
        for a in self.entries:
            self.by_base.setdefault(a.split("-")[0], []).append(a)

    def _central_directory(self):
        p = self.cache / "central_directory.bin"
        if not p.exists():
            tail = _get(ZIP_URL, ZIP_SIZE - 65536, ZIP_SIZE - 1)
            i = tail.rfind(b"PK\x05\x06")
            _, _, _, _, n_tot, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
            if cd_off == 0xFFFFFFFF or n_tot == 0xFFFF:
                raise ValueError("zip64 archive not supported")
            p.write_bytes(_get(ZIP_URL, cd_off, cd_off + cd_size - 1))
        cd = p.read_bytes()
        entries, pos = {}, 0
        while pos < len(cd):
            f = struct.unpack("<IHHHHHHIIIHHHHHII", cd[pos:pos + 46])
            if f[0] != 0x02014B50:
                raise ValueError("bad central directory signature")
            nl, el, cl = f[10], f[11], f[12]
            name = cd[pos + 46:pos + 46 + nl].decode()
            if name.endswith("_LLR.csv"):
                acc = name.rsplit("/", 1)[-1][:-len("_LLR.csv")]
                entries[acc] = {"method": f[4], "crc": f[7], "csize": f[8], "usize": f[9], "offset": f[16], "name_len": nl}
            pos += 46 + nl + el + cl
        return entries

    def _isoform_list(self):
        p = self.cache / "isoform_list.csv"
        if not p.exists():
            p.write_bytes(_get(ISOFORM_LIST_URL))
        df = pd.read_csv(p)
        # txt looks like "NUDT4B (NUD4B) | A0A024RBG1": the gene symbol comes first.
        df["gene"] = df["txt"].str.split(" ").str[0]
        return df

    def candidates(self, uniprot, gene):
        """Archive entries to try for one protein: the mapped UniProt entry and its
        isoforms, then (as a fallback) every entry listed under the same gene symbol."""
        def iso_key(a):
            return (0, 0) if "-" not in a else (1, int(a.split("-")[1]))
        first = []
        if isinstance(uniprot, str):
            first = sorted((a for a in self.by_base.get(uniprot, []) if a == uniprot or a.startswith(uniprot + "-")), key=iso_key)
        fallback = []
        if isinstance(gene, str):
            fallback = sorted(a for a in self.isoform_list.loc[self.isoform_list["gene"] == gene, "id"] if a in self.entries and a not in first)
        return first, fallback

    def fetch(self, acc):
        """Return the raw CSV bytes for one archive entry (cached, CRC-checked)."""
        e = self.entries[acc]
        p = self.cache / "llr" / f"{acc}_LLR.csv"
        if p.exists():
            raw = p.read_bytes()
            if zlib.crc32(raw) == e["crc"]:
                return raw
        # Local header is 30 bytes + name + extra field; over-read the extra field.
        start = e["offset"]
        end = start + 30 + e["name_len"] + 1024 + e["csize"]
        blob = _get(ZIP_URL, start, min(end, ZIP_SIZE - 1))
        if struct.unpack("<I", blob[:4])[0] != 0x04034B50:
            raise ValueError(f"bad local header for {acc}")
        nl, el = struct.unpack("<HH", blob[26:30])
        data = blob[30 + nl + el:30 + nl + el + e["csize"]]
        raw = zlib.decompress(data, -15) if e["method"] == 8 else data
        if zlib.crc32(raw) != e["crc"] or len(raw) != e["usize"]:
            raise ValueError(f"CRC or size mismatch for {acc}")
        p.write_bytes(raw)
        return raw

    def prefetch(self, accs, workers=8):
        accs = [a for a in dict.fromkeys(accs) if not (self.cache / "llr" / f"{a}_LLR.csv").exists()]
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(self.fetch, accs))

    def sequence(self, acc):
        """Wild-type sequence of one entry, read from its CSV header."""
        raw = self.fetch(acc)
        header = raw.split(b"\n", 1)[0].decode().strip().split(",")[1:]
        return "".join(c.split(" ")[0] for c in header)

    def load(self, acc):
        """20 x L LLR table and the wild-type sequence implied by its columns."""
        llr = pd.read_csv(io.BytesIO(self.fetch(acc)), index_col=0)
        seq = "".join(c.split(" ")[0] for c in llr.columns)
        return llr, seq
