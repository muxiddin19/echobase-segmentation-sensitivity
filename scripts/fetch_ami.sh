#!/usr/bin/env bash
# AMI manual annotations 1.6.2 and Mix-Headset audio for the meetings in configs/ami_meetings.txt (from the AMI site).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/ami/annotations data/ami/audio
wget -q -nc https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip -P data/ami/annotations
(cd data/ami/annotations && unzip -q -n ami_public_manual_1.6.2.zip)
for m in $(grep -v "^[#\[]" configs/ami_meetings.txt); do
  wget -q -nc "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/$m/audio/$m.Mix-Headset.wav" -P data/ami/audio \
    || echo "MISSING $m"
done
ls data/ami/audio | wc -l
echo FETCH_DONE
