#!/usr/bin/env bash
# D17-B (reports/deviations.md): the primary scorer in bf16 under two batch budgets, one per card, on gpulab2.
#   nohup bash scripts/d17b_bf16.sh > journal/d17b_bf16.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
M=Qwen/Qwen2.5-1.5B-Instruct
export PYTHONPATH=. HF_HUB_OFFLINE=1 POST_DTYPE=bfloat16
mkdir -p journal
# Pascal has no bf16 memory-efficient attention kernel: use PyTorch's math attention (same function, D17-B amendment)
SDP="import sys, runpy, torch; torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_flash_sdp(False); runpy.run_module('echobase.post_score', run_name='__main__')"
hostname; date; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
run() {  # $1 card, $2 budget
  local C=v2-bf16-b$2
  POST_BUDGET=$2 CUDA_VISIBLE_DEVICES=$1 nice -n 10 $PY -c "$SDP" ami --model $M --gpus 0 \
      --versions manual,placebo,w1old --cols action_item,question --cache-name $C &&
  POST_BUDGET=$2 CUDA_VISIBLE_DEVICES=$1 nice -n 10 $PY -c "$SDP" wiki --model $M --gpus 0 \
      --cols doc_date --cache-name $C && echo "DONE b$2 $(date)"
}
run 0 3072 > journal/d17b_b3072.log 2>&1 & A=$!
run 1 1024 > journal/d17b_b1024.log 2>&1 & B=$!
wait $A; ra=$?; wait $B; rb=$?
date; echo "EXIT $ra $rb"
