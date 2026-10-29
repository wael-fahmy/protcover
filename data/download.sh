#!/usr/bin/env bash
# Download the ProteinGym v1.3 clinical substitutions benchmark, its reference
# file and the precomputed zero-shot clinical scores into data/raw/.
# Usage: bash data/download.sh   (run from the repository root)
set -euo pipefail

PG_VERSION="v1.3"
PG_BASE="https://marks.hms.harvard.edu/proteingym/ProteinGym_${PG_VERSION}"
# Reference files live in the GitHub repo; pin the commit for reproducibility.
PG_COMMIT="144fe22b07dfaeec2b366f2346203a9838a55b4c"
PG_RAW="https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/${PG_COMMIT}"

RAW_DIR="$(cd "$(dirname "$0")" && pwd)/raw"
mkdir -p "${RAW_DIR}"

fetch() {
  # fetch URL DEST: download only if DEST does not exist yet
  local url="$1" dest="$2"
  if [[ -s "${dest}" ]]; then
    echo "exists: ${dest}"
  else
    echo "downloading: ${url}"
    curl -fL --retry 3 -o "${dest}.part" "${url}"
    mv "${dest}.part" "${dest}"
  fi
}

unpack() {
  # unpack ZIP DIR: extract into its own directory
  local zip="$1" dir="$2"
  if [[ -d "${dir}" ]]; then
    echo "exists: ${dir}"
  else
    mkdir -p "${dir}.part"
    unzip -q "${zip}" -d "${dir}.part"
    mv "${dir}.part" "${dir}"
  fi
}

fetch "${PG_BASE}/clinical_ProteinGym_substitutions.zip" "${RAW_DIR}/clinical_ProteinGym_substitutions.zip"
fetch "${PG_BASE}/zero_shot_clinical_substitutions_scores.zip" "${RAW_DIR}/zero_shot_clinical_substitutions_scores.zip"
# Per-sequence MSA weights: label-free source for MSA depth (Neff = sum of weights).
fetch "${PG_BASE}/clinical_msa_weights.zip" "${RAW_DIR}/clinical_msa_weights.zip"
fetch "${PG_RAW}/reference_files/clinical_substitutions.csv" "${RAW_DIR}/clinical_substitutions_reference.csv"
fetch "${PG_RAW}/reference_files/reference_files_description.md" "${RAW_DIR}/reference_files_description.md"
fetch "${PG_RAW}/config.json" "${RAW_DIR}/proteingym_config.json"
# Published per-gene AUCs, used only to confirm our score orientation.
fetch "${PG_RAW}/benchmarks/clinical_zero_shot/substitutions/AUC/Summary_performance_clinical_substitutions_AUC.csv" "${RAW_DIR}/published_clinical_AUC_summary.csv"

unpack "${RAW_DIR}/clinical_ProteinGym_substitutions.zip" "${RAW_DIR}/clinical_substitutions"
unpack "${RAW_DIR}/zero_shot_clinical_substitutions_scores.zip" "${RAW_DIR}/clinical_scores"
unpack "${RAW_DIR}/clinical_msa_weights.zip" "${RAW_DIR}/clinical_msa_weights"

# MMseqs2 release 18-8cc5c (used for family clustering and identity search).
MMSEQS_TAG="18-8cc5c"
TOOLS_DIR="$(cd "$(dirname "$0")/.." && pwd)/tools"
mkdir -p "${TOOLS_DIR}"
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) MMSEQS_ASSET="mmseqs-win64.zip" ;;
  Linux*) MMSEQS_ASSET="mmseqs-linux-avx2.tar.gz" ;;
  Darwin*) MMSEQS_ASSET="mmseqs-osx-universal.tar.gz" ;;
esac
fetch "https://github.com/soedinglab/MMseqs2/releases/download/${MMSEQS_TAG}/${MMSEQS_ASSET}" "${TOOLS_DIR}/${MMSEQS_ASSET}"
if [[ ! -d "${TOOLS_DIR}/mmseqs" ]]; then
  if [[ "${MMSEQS_ASSET}" == *.zip ]]; then
    unzip -q "${TOOLS_DIR}/${MMSEQS_ASSET}" -d "${TOOLS_DIR}"
  else
    tar -xzf "${TOOLS_DIR}/${MMSEQS_ASSET}" -C "${TOOLS_DIR}"
  fi
fi

# Record checksums so later runs can confirm the inputs are unchanged.
( cd "${RAW_DIR}" && sha256sum ./*.zip ./*.csv ./*.json > SHA256SUMS )
echo "done. checksums in ${RAW_DIR}/SHA256SUMS"
