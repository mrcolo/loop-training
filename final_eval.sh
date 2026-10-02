#!/bin/bash
# Fires when training reaches its target. The post-trained transformer is cached
# now, so none of this downloads anything; the whole sequence is a few minutes.
cd /home/stem-user/loop
until grep -q "saved at step 20000" runs/base/train.log; do sleep 60; done
echo "=== $(date -Is) step 20000 reached, running the final evaluation"
pkill -f "supervise\.sh"
while pgrep -f "python -u train\.py" > /dev/null; do sleep 5; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 3; done

# in domain: the paired A/B that produced 0.2978 at step 8000
.venv/bin/python -u evaluate.py --dit models/stable-audio-3-medium/dit_arc.safetensors \
   --objective rf_denoiser --alpha -1 --steps 8 --cfg 1.0 --write --seeds 0 1 \
   --n-offsets 4 --out final_indomain > final_indomain.log 2>&1
echo "=== in-domain exit $?"

# out of domain: the six ALESSIO tracks, and Wings at both lengths
.venv/bin/python -u eval_songs.py --out final_songs --stock > final_songs.log 2>&1
.venv/bin/python -u eval_songs.py --songs /home/stem-user/wings --seconds 190 \
   --out final_wings190 --stock > final_wings190.log 2>&1
.venv/bin/python -u eval_songs.py --songs /home/stem-user/wings --seconds 320 \
   --out final_wings320 --stock > final_wings320.log 2>&1
echo "=== $(date -Is) final evaluation complete"

# and the property check on the shipped configuration
.venv/bin/python -u probe_denoiser.py --dit models/stable-audio-3-medium/dit_arc.safetensors \
   --objective rf_denoiser --resume runs/base/dit.safetensors \
   --tag "step 20000, merged" > final_probe.log 2>&1
echo "=== $(date -Is) probe exit $?, done"
