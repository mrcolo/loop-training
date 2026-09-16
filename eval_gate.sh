#!/bin/bash
# Runs once the upload is clear of the network. In order:
#   1. six ALESSIO tracks, yours vs stock          (also caches the post-trained
#      transformer to disk, so everything after it is instant)
#   2. Wings at 190 s
#   3. Wings at 320 s
# Then training resumes.
cd /home/stem-user/loop
COUNT=/tmp/claude-1001/-home-stem-user-loop/d54a759c-ffbc-4db8-aa9c-40dd43c4927a/scratchpad/hubcount.py
echo "=== $(date -Is) waiting for all 10 delta files on the hub"
while true; do
  n=$(.venv/bin/python "$COUNT" 2>/dev/null); case "$n" in ''|*[!0-9]*) n=0 ;; esac
  [ "$n" -ge 10 ] && break; sleep 120
done
echo "=== $(date -Is) upload complete, pausing training"
pkill -f "supervise\.sh"; pkill -f "python -u train\.py"
while pgrep -f "python -u train\.py" > /dev/null; do sleep 3; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 3; done
S=$(.venv/bin/python -c "from safetensors import safe_open; print(safe_open('runs/base/dit.safetensors',framework='pt').metadata()['step'])")
echo "=== evaluating checkpoint step $S"
.venv/bin/python -u eval_songs.py --out "songs_step${S}" --stock > "songs_step${S}.log" 2>&1
echo "=== songs exit $?; wings 190 s"
.venv/bin/python -u eval_songs.py --songs /home/stem-user/wings --seconds 190 \
   --out "wings190_step${S}" --stock > "wings190_step${S}.log" 2>&1
echo "=== wings190 exit $?; wings 320 s"
.venv/bin/python -u eval_songs.py --songs /home/stem-user/wings --seconds 320 \
   --out "wings320_step${S}" --stock > "wings320_step${S}.log" 2>&1
echo "=== $(date -Is) all renders done, resuming training"
STEPS=20000 exec ./supervise.sh
