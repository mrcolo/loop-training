#!/bin/bash
# Second wave: the AdamW implementation itself. v0 uses bitsandbytes AdamW8bit,
# which approximates the moments. FlashAdamW is bit-for-bit AdamW with int8 state,
# and torch fused is full precision. Same seed and steps as wave one.
cd /home/stem-user/loop
while pgrep -f "^\.venv/bin/python -u train\.py" > /dev/null; do sleep 60; done
STEPS=${STEPS:-1200}; SEED=1234
COMMON=(--latents latents.npy
        --dit models/stable-audio-3-medium-base/dit_base.safetensors
        --objective rectified_flow --steps "$STEPS" --batch 1 --accum 4
        --seconds 47.0 95.0 190.0 380.0 --ctx-min 20.0 --ctx-max 60.0 --min-gen 15.0
        --p-full 0.55 --p-segments 0.10 --muon-lr 2e-4 --adam-lr 1e-5
        --val-every 25 --demo-every 100000 --save-every 100000 --seed "$SEED")
for impl in flash fused; do
  [ -d "runs/ablate/adam_$impl/tb" ] && { echo "=== skip adam_$impl"; continue; }
  mkdir -p "runs/ablate/adam_$impl"; cp runs/base/prompt_cond.pt "runs/ablate/adam_$impl/" 2>/dev/null
  echo "=== $(date -Is) arm: adam_$impl"
  .venv/bin/python -u train.py "${COMMON[@]}" --adam-impl "$impl" \
      --out "runs/ablate/adam_$impl" > "runs/ablate/adam_$impl.log" 2>&1
  echo "=== $(date -Is) adam_$impl exit $?"
done
echo "=== $(date -Is) wave two complete"
