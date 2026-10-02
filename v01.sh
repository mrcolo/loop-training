#!/bin/bash
# v0.1 candidate: Muon alone on the transformer matrices, the AdamW group frozen.
# The ablation says those 94 M parameters contribute nothing over 1200 steps; this
# runs the same 8000 steps v0's best checkpoint came from, so the comparison is
# against a number we already have (0.2978 in domain, paired, eight renders).
# Checkpoints are kept this time rather than overwritten.
cd /home/stem-user/loop
while pgrep -f "^\.venv/bin/python -u train\.py" > /dev/null; do sleep 60; done
.venv/bin/python -u train.py \
  --latents latents.npy --dit models/stable-audio-3-medium-base/dit_base.safetensors \
  --objective rectified_flow --out runs/v01 --steps 8000 --batch 1 --accum 4 \
  --muon-lr 2e-4 --adam-lr 0 \
  --seconds 47.0 95.0 190.0 380.0 \
  --p-full 0.55 --p-segments 0.10 --ctx-min 20.0 --ctx-max 60.0 --min-gen 15.0 \
  --demo-at 7289.25 --demo-context 30.0 --demo-seconds 190.0 \
  --val-every 50 --demo-every 500 --save-every 1000 --keep-checkpoints \
  > runs/v01.log 2>&1
echo "=== $(date -Is) v0.1 exit $?, evaluating"
.venv/bin/python -u evaluate.py --dit models/stable-audio-3-medium/dit_arc.safetensors \
  --objective rf_denoiser --resume runs/v01/dit_008000.safetensors --alpha 1 \
  --steps 8 --cfg 1.0 --write --seeds 0 1 --n-offsets 4 --out v01_indomain \
  > v01_indomain.log 2>&1
.venv/bin/python -u eval_songs.py --resume runs/v01/dit_008000.safetensors \
  --out v01_songs --stock > v01_songs.log 2>&1
echo "=== $(date -Is) v0.1 evaluation complete"
