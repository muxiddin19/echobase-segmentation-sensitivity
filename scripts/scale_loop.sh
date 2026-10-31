#!/usr/bin/env bash
# D5 scale analysis on one GPU (gp231 GPU 0): diarize the AMI meetings whose audio has finished downloading, score them
# with the second model, repeat until the download is done and every downloaded meeting is processed.
#   bash scripts/scale_loop.sh <gpu>
# Never runs a round while another user's process is on the card. Stop with: touch logs/scale_stop
set -u
cd "$(dirname "$0")/.."
GPU=${1:-0}
M=Qwen/Qwen2.5-0.5B-Instruct
export HF_HUB_OFFLINE=1 PYTHONPATH=.
while [ ! -e logs/scale_stop ]; do
  others=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | while read -r p; do
             [ -n "$p" ] && [ "$(ps -o user= -p "$p" 2>/dev/null)" != "$USER" ] && echo "$p"; done)
  if [ -n "$others" ]; then echo "$(date) GPU $GPU busy (pids $others); waiting"; sleep 300; continue; fi
  .venv/bin/python -m echobase.run_diar_scale --meetings logs/scale_order.txt --gpu "$GPU"
  for m in $(cat logs/scale_order.txt); do [ -f "outputs/diar/$m.json" ] && echo "$m"; done > logs/scale_done.txt
  .venv/bin/python -m echobase.post_score ami --meetings logs/scale_done.txt --versions manual,w1old,placebo \
    --model "$M" --gpus "$GPU"
  echo "$(date) round done: $(wc -l < logs/scale_done.txt) meetings diarized and scored"
  if grep -q FETCH_DONE logs/fetch_scale.log && [ "$(wc -l < logs/scale_done.txt)" -ge "$(ls data/ami/audio | grep -c -f logs/scale_order.txt)" ]; then
    echo SCALE_DONE; break
  fi
  sleep 600
done
