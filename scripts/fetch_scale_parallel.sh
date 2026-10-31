#!/usr/bin/env bash
# AMI Mix-Headset audio for configs/ami_scale.txt into the NAS dataset folder: fixed random order (seed 0), so the
# meetings finished by any cut-off are an unbiased sample; 16 parallel connections; -c resumes partial files.
cd ~/muhiddin/icde_pilots
grep -v "^#" configs/ami_scale.txt | python3 -c "import random,sys; m=sys.stdin.read().split(); random.Random(0).shuffle(m); print(\"\n\".join(m))" > logs/scale_order.txt
xargs -P 16 -I{} wget -q -c -P /nas/muhiddin/datasets/AMI/audio \
  "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{}/audio/{}.Mix-Headset.wav" < logs/scale_order.txt
echo FETCH_DONE
