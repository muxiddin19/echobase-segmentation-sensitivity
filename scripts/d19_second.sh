#!/usr/bin/env bash
# D19 (reports/deviations.md): second scorer (Qwen2.5-3B fp32, split over both TITAN Xp) on D17-C and D18, gpulab2.
#   nohup bash scripts/d19_second.sh > journal/d19.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH=. HF_HUB_OFFLINE=1 EB_MODEL=Qwen/Qwen2.5-3B-Instruct
mkdir -p journal reports
hostname; date
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
nice -n 10 $PY -m echobase.d17_corpora check --gpus 0,1 --out reports/d19_3b_determinism.json \
    || { echo "determinism check failed"; exit 1; }
date; nice -n 10 $PY -m echobase.d17_corpora icsi --gpus 0,1 > journal/d19_icsi.log 2>&1; echo "icsi EXIT $?"
date; nice -n 10 $PY -m echobase.d17_corpora crs --gpus 0,1 > journal/d19_crs.log 2>&1; echo "crs EXIT $?"
date; nice -n 10 $PY -m echobase.d18_canonical score --gpus 0,1 > journal/d19_canon.log 2>&1; echo "canon EXIT $?"
date; echo DONE
