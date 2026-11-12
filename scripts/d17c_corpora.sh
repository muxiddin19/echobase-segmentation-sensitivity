#!/usr/bin/env bash
# D17-C (reports/deviations.md): the primary scorer (fp32, frozen prompts) on ICSI and CRS, gpulab2, both TITAN Xp.
#   nohup bash scripts/d17c_corpora.sh > journal/d17c.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH=. HF_HUB_OFFLINE=1
mkdir -p journal reports
hostname; date
CUDA_VISIBLE_DEVICES=0 nice -n 10 $PY -m echobase.d17_corpora check --gpus 0 --out reports/d17_corpora_determinism.json \
    || { echo "determinism check failed"; exit 1; }
CUDA_VISIBLE_DEVICES=0 nice -n 10 $PY -m echobase.d17_corpora icsi --gpus 0 > journal/d17c_icsi.log 2>&1 & A=$!
CUDA_VISIBLE_DEVICES=1 nice -n 10 $PY -m echobase.d17_corpora crs --gpus 0 > journal/d17c_crs.log 2>&1 & B=$!
wait $A; ra=$?; wait $B; rb=$?
date; echo "EXIT $ra $rb"
