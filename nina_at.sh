#!/bin/bash
# At the given checkpoint, pause training, outpaint the Ninajirachi track with the
# merged eight-step model, and hand the card back.
cd /home/stem-user/loop
STEP=${1:-9000}
NINA="/home/stem-user/.claude/uploads/d54a759c-ffbc-4db8-aa9c-40dd43c4927a/648939e8-Angel_Music_by_Ninajirachi_MGNA_Crrrta.flac"
until grep -q "saved at step $STEP" runs/base/train.log; do sleep 20; done
echo "=== $(date -Is) step $STEP saved, pausing to render Nina"
pkill -f "supervise\.sh"; pkill -f "python -u train\.py"
while pgrep -f "python -u train\.py" > /dev/null; do sleep 3; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 3; done
for seed in 0 1; do
  .venv/bin/python -u sample.py --audio "$NINA" --stream-arc \
     --resume runs/base/dit.safetensors --out "nina_step${STEP}" --tag "s${seed}" \
     --seconds 190 --context 30 --at 0 --steps 8 --cfg 1.0 --seed $seed
done
echo "=== $(date -Is) nina renders done, resuming training"
STEPS=20000 exec ./supervise.sh
