#!/bin/bash
# Restart the finetune if it dies. Resumes from the last checkpoint when one exists.
cd /home/stem-user/loop
OUT=runs/base
while true; do
  ARGS=(--latents latents.npy
        --dit models/stable-audio-3-medium-base/dit_base.safetensors
        --objective rectified_flow
        --out "$OUT" --steps 20000 --batch 1 --accum 4
        --muon-lr 2e-4 --adam-lr 1e-5
        --seconds 47.0 95.0 190.0 380.0
        --p-full 0.55 --p-segments 0.10 --ctx-min 20.0 --ctx-max 60.0 --min-gen 15.0
        --demo-at 7289.25 --demo-context 30.0 --demo-seconds 190.0
        --val-every 50 --demo-every 500 --save-every 500)
  [ -f "$OUT/dit.safetensors" ] && ARGS+=(--resume "$OUT/dit.safetensors")
  echo "=== $(date -Is) starting: ${ARGS[*]}"
  .venv/bin/python -u train.py "${ARGS[@]}"
  code=$?
  echo "=== $(date -Is) exited $code"
  [ $code -eq 0 ] && break
  sleep 20
done
