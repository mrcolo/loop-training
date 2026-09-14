#!/usr/bin/env bash
# Waits for the source track to be readable, then starts the finetune.
# stem-user cannot read /home/alessio/Desktop, so either drop a copy at
# ./audio.flac or, as alessio: chmod o+x ~ ~/Desktop && chmod o+r ~/Desktop/2n1t3-audio.flac
set -u
cd /home/stem-user/loop
SRC=/home/alessio/Desktop/2n1t3-audio.flac
until [ -r "$SRC" ] || [ -r audio.flac ]; do sleep 20; done
[ -r "$SRC" ] || SRC=audio.flac
echo "training on $SRC"
exec env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python train.py --audio "$SRC" "$@"
