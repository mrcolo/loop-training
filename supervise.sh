#!/bin/bash
# Keep the finetune running. Resumes from the latest checkpoint, waits for the
# card if something else is using it, and restarts on any non-zero exit.
cd /home/stem-user/loop
OUT=${OUT:-runs/base}
STEPS=${STEPS:-20000}
NEED_GB=${NEED_GB:-22}

wait_for_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    (( (total - used) / 1024 >= NEED_GB )) && return
    echo "=== $(date -Is) waiting for the card: $(( (total-used)/1024 )) GB free, need ${NEED_GB}"
    sleep 60
  done
}

while true; do
  wait_for_gpu
  ARGS=(--latents latents.npy
        --dit models/stable-audio-3-medium-base/dit_base.safetensors
        --objective rectified_flow
        --out "$OUT" --steps "$STEPS" --batch 1 --accum 4
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
  [ $code -eq 0 ] && { echo "=== reached --steps $STEPS, done"; break; }
  sleep 20
done
