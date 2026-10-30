#!/usr/bin/env bash
# Download the small mapping files used by scripts/build_unseen.py (the
# unseen-protein ClinVar test set) into data/raw/unseen/ (gitignored).
# Usage: bash data/unseen/download.sh   (run from the repository root)
#
# The ClinVar files are shared with the external temporal set and come from
# data/external/download.sh. This script adds:
#   * the HGNC complete set, monthly archive of 2026-10-06 (gene symbol, NCBI
#     GeneID and UniProt accessions per human gene; about 17 MB), and
#   * the human RefSeq protein sequences of RefSeq release 237 (mRNA_Prot,
#     19 parts, about 35 MB), used to check which RefSeq protein has exactly the
#     sequence that Brandes et al. scored.
# NCBI overwrites mRNA_Prot at every RefSeq release, so the release number and
# Last-Modified headers are stored next to the files, and a file that already
# exists is never downloaded again.
set -euo pipefail

OUT="$(cd "$(dirname "$0")/../.." && pwd)/data/raw/unseen"
mkdir -p "${OUT}/hgnc" "${OUT}/refseq_protein"

fetch() {
  # fetch URL DEST: download only if DEST does not exist yet
  local url="$1" dest="$2"
  if [[ -s "${dest}" ]]; then
    echo "exists: ${dest}"
  else
    echo "downloading: ${url}"
    curl -fsI "${url}" | tr -d '\r' | grep -i '^last-modified' > "${dest}.last_modified" || true
    curl -fL --retry 3 -R -s -o "${dest}.part" "${url}"
    mv "${dest}.part" "${dest}"
  fi
}

HGNC="https://storage.googleapis.com/public-download-files/hgnc/archive/archive/monthly/tsv/hgnc_complete_set_2026-10-06.tsv"
fetch "${HGNC}" "${OUT}/hgnc/hgnc_complete_set_2026-10-06.tsv"

RS="https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/mRNA_Prot"
if [[ ! -s "${OUT}/refseq_protein/RELEASE_NUMBER" ]]; then
  curl -fs "https://ftp.ncbi.nlm.nih.gov/refseq/release/RELEASE_NUMBER" > "${OUT}/refseq_protein/RELEASE_NUMBER"
fi
echo "RefSeq release $(cat "${OUT}/refseq_protein/RELEASE_NUMBER")"
for i in $(seq 1 19); do
  fetch "${RS}/human.${i}.protein.faa.gz" "${OUT}/refseq_protein/human.${i}.protein.faa.gz"
done
# All parts must be gzip files.
for f in "${OUT}"/refseq_protein/human.*.protein.faa.gz; do gzip -t "${f}"; done

( cd "${OUT}" && sha256sum hgnc/*.tsv refseq_protein/*.faa.gz > SHA256SUMS )
echo "done. checksums in ${OUT}/SHA256SUMS"
