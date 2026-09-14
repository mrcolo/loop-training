#!/usr/bin/env bash
# Wait for the ARC weights, cache the conditioning, reclaim the text encoder,
# then run the long finetune under the supervisor.
set -u
cd /home/stem-user/loop
# the fetch runs under its own retry wrapper, so wait on its output, not its pid
for _ in $(seq 1 240); do
  grep -q "^wrote models" fetcharc.log 2>/dev/null && break
  sleep 60
done
grep -q "^wrote models" fetcharc.log || { echo "CHAIN: weights never arrived"; exit 1; }
echo "CHAIN: weights ready $(date -Is)"
rm -rf runs/outpaint; mkdir -p runs/outpaint

echo "CHAIN: building conditioning cache"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u train.py \
  --audio /home/alessio/Desktop/2n1t3-audio.flac --latents latents.npy \
  --seconds 190 --batch 1 --steps 1 --save-every 100000 --demo-every 100000 \
  > condcache.log 2>&1
if [ -f runs/outpaint/conditioning.pt ]; then
  echo "CHAIN: cache built, reclaiming t5gemma"
  rm -rf models/stable-audio-3-medium/t5gemma-b-b-ul2
  df -h / | tail -1
else
  echo "CHAIN: no cache, keeping t5gemma"
fi

echo "CHAIN: starting 40000-step run $(date -Is)"
rm -f train.log
HOURS=13 ./supervise.sh --audio /home/alessio/Desktop/2n1t3-audio.flac --latents latents.npy \
  --seconds 190 --batch 1 --steps 40000 --lr 5e-5 --lr-1d 5e-5 \
  --save-every 2000 --demo-every 4000 --demo-alpha 0.25 --demo-at 7289.25
echo "CHAIN: done $(date -Is)"
