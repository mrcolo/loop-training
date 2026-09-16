#!/bin/bash
# At the given checkpoint: pause training, run the standing song evaluation,
# hand the card back. The supervisor is stopped first so it cannot race for memory.
cd /home/stem-user/loop
STEP=${1:-10000}
until grep -q "saved at step $STEP" runs/base/train.log; do sleep 20; done
echo "=== $(date -Is) step $STEP saved, pausing for the song evaluation"
pkill -f "supervise\.sh"; pkill -f "python -u train\.py"
while pgrep -f "python -u train\.py" > /dev/null; do sleep 3; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 3; done
.venv/bin/python -u eval_songs.py --out "songs_step${STEP}" --stock > "songs_step${STEP}.log" 2>&1
echo "=== $(date -Is) song evaluation exit $?, resuming training"
STEPS=20000 exec ./supervise.sh
