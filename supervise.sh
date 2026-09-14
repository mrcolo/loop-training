#!/usr/bin/env bash
# Keeps a long finetune alive. Relaunches from the last checkpoint if the run
# dies, and stops once the time budget is spent or training exits cleanly.
set -u
cd /home/stem-user/loop
HOURS=${HOURS:-8}
DEADLINE=$(( $(date +%s) + HOURS * 3600 ))
FILTER='flash_attn|Flash Attention|varlen|WeightNorm|FutureWarning|_dynamo|torch/_inductor|UserWarning|Triggered internally|Python.h|compilation terminated'

for attempt in $(seq 1 100); do
  now=$(date +%s)
  if [ "$now" -ge "$DEADLINE" ]; then echo "SUPERVISOR: budget of ${HOURS}h spent"; break; fi
  RESUME=""
  [ -f runs/outpaint/dit.safetensors ] && RESUME="--resume runs/outpaint/dit.safetensors"
  echo "SUPERVISOR: attempt $attempt at $(date -Is) ${RESUME:-(fresh)}"
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python -u train.py "$@" $RESUME 2>&1 \
    | grep --line-buffered -viE "$FILTER" >> train.log
  rc=${PIPESTATUS[0]}
  echo "SUPERVISOR: train.py exited rc=$rc at $(date -Is)"
  if [ "$rc" -eq 0 ]; then echo "SUPERVISOR: training finished cleanly"; break; fi
  df -h / | tail -1
  sleep 20
done
echo "SUPERVISOR: done at $(date -Is)"
