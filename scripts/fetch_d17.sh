#!/usr/bin/env bash
# Download the second corpora of the post-hoc analyses D17-C and D19 (reports/deviations.md) and check them against
# the files we used.
#   bash scripts/fetch_d17.sh
# ICSI Meeting Corpus core NXT annotations (CC BY 4.0; https://groups.inf.ed.ac.uk/ami/icsi/) -> data/icsi/
# GovReport CRS test split (US government works; https://huggingface.co/datasets/launch/gov_report) -> data/govreport/
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/icsi data/govreport/data
[ -f data/icsi/ICSI_core_NXT.zip ] || curl -sSL -o data/icsi/ICSI_core_NXT.zip \
  https://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/ICSI_core_NXT.zip
(cd data/icsi && unzip -q -o ICSI_core_NXT.zip)
[ -f data/govreport/data/crs_test.jsonl ] || curl -sSL -o data/govreport/data/crs_test.jsonl \
  https://huggingface.co/datasets/launch/gov_report/resolve/main/data/crs_test.jsonl
sha256sum -c - <<'EOF'
cf4860245b9ca9c9ed11a66e5a74cfea2a99a686a52f7580aa0a6bc2225895a9  data/icsi/ICSI_core_NXT.zip
bef184f1f4249ee4f145214b9f836a277152d5074395b879bd9ea891c519dece  data/govreport/data/crs_test.jsonl
EOF
