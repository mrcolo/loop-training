#!/usr/bin/env bash
# Wait for the latent stream, then start the long finetune under the supervisor.
set -u
cd /home/stem-user/loop
until grep -q "^wrote latents.npy" encode.log 2>/dev/null; do
  if ! pgrep -f encode_latents.py >/dev/null; then
    echo "CHAIN: encoder died before finishing; aborting"; exit 1
  fi
  sleep 60
done
echo "CHAIN: latents ready at $(date -Is), starting training"
rm -f train.log; rm -rf runs/outpaint
HOURS=7 ./supervise.sh \
  --audio /home/alessio/Desktop/2n1t3-audio.flac --latents latents.npy \
  --seconds 190 --batch 1 --steps 16000 \
  --save-every 1000 --demo-every 2000 --demo-at 7289.25
