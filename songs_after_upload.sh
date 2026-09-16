#!/bin/bash
# Evaluate the six ALESSIO tracks once the upload is out of the way.
# Both want the network, and running them together starves both.
cd /home/stem-user/loop
COUNT=/tmp/claude-1001/-home-stem-user-loop/d54a759c-ffbc-4db8-aa9c-40dd43c4927a/scratchpad/hubcount.py
echo "=== $(date -Is) waiting for all 10 delta files on the hub"
while true; do
  n=$(.venv/bin/python "$COUNT" 2>/dev/null)
  case "$n" in ''|*[!0-9]*) n=0 ;; esac
  [ "$n" -ge 10 ] && break
  sleep 120
done
echo "=== $(date -Is) upload complete, pausing training for the song evaluation"
pkill -f "supervise\.sh"; pkill -f "python -u train\.py"
while pgrep -f "python -u train\.py" > /dev/null; do sleep 3; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 3; done
S=$(.venv/bin/python -c "from safetensors import safe_open; print(safe_open('runs/base/dit.safetensors',framework='pt').metadata()['step'])")
echo "=== evaluating checkpoint step $S"
.venv/bin/python -u eval_songs.py --out "songs_step${S}" --stock > "songs_step${S}.log" 2>&1
echo "=== $(date -Is) song evaluation exit $?, resuming training"
STEPS=20000 exec ./supervise.sh
