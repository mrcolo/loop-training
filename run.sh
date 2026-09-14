#!/usr/bin/env bash
# Waits for the bfloat16 checkpoint to finish downloading, then trains.
set -eu
cd /home/stem-user/loop
until grep -q "^wrote models" fetch.log 2>/dev/null; do sleep 30; done
exec env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python train.py "$@"
