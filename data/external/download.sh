#!/usr/bin/env bash
# Download the ClinVar files used to build the external temporal test set
# (scripts/build_external.py) into data/raw/external/ (gitignored).
# Usage: bash data/external/download.sh   (run from the repository root)
#
# The two 2023 files are permanent monthly archives. variant_summary.txt.gz and
# hgvs4variation.txt.gz at the top of tab_delimited/ are overwritten weekly by
# NCBI, so this script stores their HTTP Last-Modified headers next to them and
# never re-downloads a file that already exists. The ESM-1b LLR files are not
# fetched here: build_external.py fetches them per protein by HTTP range
# requests (scripts/external_esm1b.py).
set -euo pipefail

CV="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited"
OUT="$(cd "$(dirname "$0")/../.." && pwd)/data/raw/external"
mkdir -p "${OUT}/clinvar_archive" "${OUT}/clinvar_current"

fetch() {
  # fetch URL DEST: download only if DEST does not exist yet
  local url="$1" dest="$2"
  if [[ -s "${dest}" ]]; then
    echo "exists: ${dest}"
  else
    echo "downloading: ${url}"
    curl -fL --retry 3 -R -o "${dest}.part" "${url}"
    mv "${dest}.part" "${dest}"
  fi
}

# Snapshot boundary (release of 2023-06-01) and the next monthly release
# (2023-07-06), used only for a sensitivity flag.
for m in 2023-06 2023-07; do
  fetch "${CV}/archive/2023/variant_summary_${m}.txt.gz" "${OUT}/clinvar_archive/variant_summary_${m}.txt.gz"
done

# Current labels and protein HGVS.
for f in variant_summary.txt.gz hgvs4variation.txt.gz; do
  if [[ ! -s "${OUT}/clinvar_current/${f}" ]]; then
    curl -fsI "${CV}/${f}" | tr -d '\r' | grep -i '^last-modified' > "${OUT}/clinvar_current/${f}.last_modified"
    fetch "${CV}/${f}.md5" "${OUT}/clinvar_current/${f}.md5"
  fi
  fetch "${CV}/${f}" "${OUT}/clinvar_current/${f}"
done

# NCBI publishes an MD5 for the current files; check it.
( cd "${OUT}/clinvar_current" && for f in variant_summary.txt.gz hgvs4variation.txt.gz; do
    want="$(awk '{print $1}' "${f}.md5")"; got="$(md5sum "${f}" | awk '{print $1}')"
    if [[ "${want}" != "${got}" ]]; then echo "MD5 mismatch for ${f}" >&2; exit 1; fi
    echo "md5 ok: ${f}"
  done )

( cd "${OUT}" && sha256sum clinvar_archive/*.txt.gz clinvar_current/*.txt.gz > SHA256SUMS )
echo "done. checksums in ${OUT}/SHA256SUMS"
