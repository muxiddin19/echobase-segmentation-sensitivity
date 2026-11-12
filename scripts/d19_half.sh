#!/usr/bin/env bash
# D19 extension (reports/deviations.md): Qwen2.5-0.5B (fp32, one TITAN Xp each) on D17-C and D18, gpulab2.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH=. HF_HUB_OFFLINE=1 EB_MODEL=Qwen/Qwen2.5-0.5B-Instruct
mkdir -p journal reports
hostname; date
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
CUDA_VISIBLE_DEVICES=0 nice -n 10 $PY -m echobase.d17_corpora check --gpus 0 --out reports/d19_half_determinism.json \
    || { echo "determinism check failed"; exit 1; }
CUDA_VISIBLE_DEVICES=0 nice -n 10 $PY -m echobase.d17_corpora icsi --gpus 0 > journal/d19h_icsi.log 2>&1 & A=$!
CUDA_VISIBLE_DEVICES=1 nice -n 10 $PY -m echobase.d17_corpora crs --gpus 0 > journal/d19h_crs.log 2>&1
rb=$?; echo "crs EXIT $rb"
CUDA_VISIBLE_DEVICES=1 nice -n 10 $PY -m echobase.d18_canonical score --gpus 0 > journal/d19h_canon.log 2>&1; echo "canon EXIT $?"
wait $A; echo "icsi EXIT $?"
date; echo DONE
