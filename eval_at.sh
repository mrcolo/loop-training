#!/bin/bash
# Wait for a checkpoint, take the card for an evaluation, give it back.
# The supervisor restarts the trainer on any non-zero exit, so it has to be
# stopped first or it races the evaluation for memory.
cd /home/stem-user/loop
STEP=${1:-3500}
until grep -q "saved at step $STEP" runs/base/train.log; do sleep 20; done
echo "=== $(date -Is) step $STEP saved, pausing for the evaluation"
pkill -f "supervise\.sh"
pkill -f "python -u train\.py"
while pgrep -f "python -u train\.py" > /dev/null; do sleep 3; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 3; done
.venv/bin/python -u evaluate.py --write --seeds 0 --n-offsets 4 > eval.log 2>&1
echo "=== $(date -Is) evaluation exit $?, resuming training"
exec ./supervise.sh
