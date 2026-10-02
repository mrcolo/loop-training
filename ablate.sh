#!/bin/bash
# Ablation for v0.1. Each arm differs from v0 in exactly one thing, runs the same
# number of steps from the same initialisation with the same seed, and is read on
# the frozen validation slice. 1200 steps is where v0's curve had clearly
# separated from its start (0.9275 -> ~0.919) while still costing under 2 hours.
cd /home/stem-user/loop
STEPS=${STEPS:-1200}
SEED=1234
COMMON=(--latents latents.npy
        --dit models/stable-audio-3-medium-base/dit_base.safetensors
        --objective rectified_flow --steps "$STEPS" --batch 1 --accum 4
        --seconds 47.0 95.0 190.0 380.0
        --ctx-min 20.0 --ctx-max 60.0 --min-gen 15.0
        --val-every 25 --demo-every 100000 --save-every 100000
        --seed "$SEED")

run () {                       # run <name> <extra args...>
  local name=$1; shift
  [ -d "runs/ablate/$name/tb" ] && { echo "=== skip $name (done)"; return; }
  mkdir -p "runs/ablate/$name"
  cp runs/base/prompt_cond.pt "runs/ablate/$name/" 2>/dev/null
  echo "=== $(date -Is) arm: $name  $*"
  .venv/bin/python -u train.py "${COMMON[@]}" --out "runs/ablate/$name" "$@" \
      > "runs/ablate/$name.log" 2>&1
  echo "=== $(date -Is) $name exit $?"
}

# v0 as the control, then one change at a time
run v0            --muon-lr 2e-4 --adam-lr 1e-5 --p-full 0.55 --p-segments 0.10
run muon_5e-5     --muon-lr 5e-5 --adam-lr 1e-5 --p-full 0.55 --p-segments 0.10
run muon_1e-3     --muon-lr 1e-3 --adam-lr 1e-5 --p-full 0.55 --p-segments 0.10
run muon_4e-4     --muon-lr 4e-4 --adam-lr 1e-5 --p-full 0.55 --p-segments 0.10
run adam_5e-5     --muon-lr 2e-4 --adam-lr 5e-5 --p-full 0.55 --p-segments 0.10
run adam_1e-6     --muon-lr 2e-4 --adam-lr 1e-6 --p-full 0.55 --p-segments 0.10
run pfull_80      --muon-lr 2e-4 --adam-lr 1e-5 --p-full 0.80 --p-segments 0.10
echo "=== $(date -Is) all arms complete"
