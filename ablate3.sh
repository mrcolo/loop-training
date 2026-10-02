#!/bin/bash
# Wave three: push the AdamW rate further down, since 1e-6 beat 1e-5 and the
# ordering has not turned over yet.
cd /home/stem-user/loop
while pgrep -f "^\.venv/bin/python -u train\.py" > /dev/null; do sleep 60; done
STEPS=1200; SEED=1234
COMMON=(--latents latents.npy --dit models/stable-audio-3-medium-base/dit_base.safetensors
        --objective rectified_flow --steps "$STEPS" --batch 1 --accum 4
        --seconds 47.0 95.0 190.0 380.0 --ctx-min 20.0 --ctx-max 60.0 --min-gen 15.0
        --p-full 0.55 --p-segments 0.10 --muon-lr 2e-4
        --val-every 25 --demo-every 100000 --save-every 100000 --seed "$SEED")
for lr in 1e-7 0; do
  name="adam_${lr}"
  [ -d "runs/ablate/$name/tb" ] && continue
  mkdir -p "runs/ablate/$name"; cp runs/base/prompt_cond.pt "runs/ablate/$name/" 2>/dev/null
  echo "=== $(date -Is) arm: $name"
  .venv/bin/python -u train.py "${COMMON[@]}" --adam-lr "$lr" --out "runs/ablate/$name" \
     > "runs/ablate/$name.log" 2>&1
  echo "=== $(date -Is) $name exit $?"
done
echo "=== $(date -Is) wave three complete"
