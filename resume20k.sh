#!/bin/bash
# Wait for the base transformer to finish downloading, then run to 20000 steps.
cd /home/stem-user/loop
while pgrep -f "python -u fetch_model\.py" > /dev/null; do sleep 30; done
if [ ! -s models/stable-audio-3-medium-base/dit_base.safetensors ]; then
  echo "=== $(date -Is) base transformer missing, not starting"; exit 1
fi
echo "=== $(date -Is) base transformer ready, resuming to 20000"
STEPS=20000 exec ./supervise.sh
