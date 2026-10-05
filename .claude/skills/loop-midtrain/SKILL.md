---
name: loop-midtrain
description: Mid-train (full finetune) Stable Audio 3 Medium on the 26.4 h 2n1t3 DJ-set latents to produce a "loop" model — e.g. loop v0, v0.1 — and evaluate it. Use when asked to retrain loop, run a new loop version, change loop's hyperparameters, recover a loop checkpoint, or compare loop runs. Not for LoRA on new material (that is lora.py).
---

# Mid-training loop on the 2n1t3 corpus

Everything here was learned the hard way on this repo (`/home/stem-user/loop`,
GitHub `mrcolo/loop-training`, weights `fcolooo/loop-0`). Each rule below cost
at least a day when it was missing.

## The data

| file | what |
| --- | --- |
| `latents.npy` | 26.4 h of 2n1t3 DJ sets, Los Angeles. float16, 256 ch @ 10.77 Hz, 501 MB, memory-mapped |
| `energy_index.npz` | per-8 s RMS / sub-bass; training gates at rms >= 0.15 |
| `runs/base/prompt_cond.pt` | the cached prompt embedding; copy into each new `--out` dir |

One pass over the corpus is **~134 optimiser steps** (batch 1 x accum 4, mean
window 178 s). Know which pass you are on; steps alone mislead.

## The models

| file | role |
| --- | --- |
| `models/stable-audio-3-medium-base/dit_base.safetensors` | **what you train** |
| `models/stable-audio-3-medium/dit_arc.safetensors` | post-trained transformer; what you ship on |
| `models/stable-audio-3-medium/model.safetensors` | autoencoder + conditioners |

## Non-negotiables

1. **Train `-base`, never the released checkpoint.** MSE on the adversarially
   post-trained weights turns a sample predictor into a mean predictor; ping-pong
   then produces quiet mush while the loss looks fine. `DIAGNOSIS.md`.
2. **Checkpoints are deltas from base** (fp16). Ship by adding the delta to
   `dit_arc`. Verified to keep 8-step sampling intact up to 20000 steps.
3. **Pass `--keep-checkpoints`.** v0's best checkpoint (step 8000) was lost to
   overwriting. Disk has room; ~2.9 GB per save.
4. **Pass `--seed`** for anything you intend to compare.
5. **Never judge a run by validation loss.** Three times it ranked things in the
   wrong order (post-trained run "healthy"; step 20000 "better" than 8000; v0.1
   "best" ablation arm, worst audio). Score on audio, below.
6. **One GPU job and one large network transfer at a time.** Concurrent
   downloads/uploads starved each other repeatedly. Kill the supervisor before
   the trainer when pausing, or it relaunches into the evaluation.
7. Use `pgrep -f "python -u train[.]py"` style patterns — a bare pattern
   matches your own shell and `pkill` kills it.

## The v0 recipe (tag `loop-v0`, commit `e82f967`)

```bash
mkdir -p runs/NAME && cp runs/base/prompt_cond.pt runs/NAME/
python train.py --latents latents.npy \
  --dit models/stable-audio-3-medium-base/dit_base.safetensors \
  --objective rectified_flow --out runs/NAME --steps 8000 --batch 1 --accum 4 \
  --muon-lr 2e-4 --adam-lr 1e-5 \
  --seconds 47 95 190 380 --p-full 0.55 --p-segments 0.10 \
  --ctx-min 20 --ctx-max 60 --min-gen 15 \
  --demo-at 7289.25 --demo-context 30 --demo-seconds 190 \
  --val-every 50 --demo-every 500 --save-every 1000 --keep-checkpoints --seed 1234
```

4.8 s/step on the 3090, 20.8 GiB VRAM, ~39 GB host RAM (dataloader workers fork
after the weights load). 8000 steps ~ 11 h. `supervise.sh` restarts on crash
and resumes from the latest checkpoint.

## Where to stop

The model saturates around **pass 30-60 (step 4000-8000)**. Measured in domain:

| checkpoint | distance to real continuation (lower better) |
| --- | --- |
| stock post-trained, 8 step | 0.4537 |
| v0 @ 8000 | **0.2978** |
| v0 @ 20000 | 0.3377 |
| v0.1 (AdamW group frozen) @ 8000 | 0.3539 |

Going past 8000 made it worse. Evaluate several kept checkpoints and pick the
peak rather than assuming the last one.

## Evaluate (audio, not loss)

```bash
# in domain: 4 energy-gated offsets x 2 seeds, matched trajectories, vs stock
python evaluate.py --dit models/stable-audio-3-medium/dit_arc.safetensors \
  --objective rf_denoiser --resume runs/NAME/dit_008000.safetensors --alpha 1 \
  --steps 8 --cfg 1.0 --write --seeds 0 1 --n-offsets 4 --out eval_NAME
# out of domain: the six ALESSIO tracks in /home/stem-user/eval
python eval_songs.py --resume runs/NAME/dit_008000.safetensors --out songs_NAME --stock
# property check: must stay a sample predictor (amplitude well above correlation)
python probe_denoiser.py --dit models/stable-audio-3-medium/dit_arc.safetensors \
  --objective rf_denoiser --resume runs/NAME/dit_008000.safetensors
```

`--alpha 1` adds the delta to `dit_arc`. `--alpha -1` subtracts it, which is
only correct when `--dit` is already a merged file. Getting this backwards
produced a nonsense "stock" model once.

## What the ablation found (1200 steps, scored on loss — treat as hypotheses)

- Muon lr: 2e-4 ~ 4e-4 best; 5e-5 slower; **1e-3 (Stability's pretrain value) barely trains**.
- AdamW lr: lower looked better on loss (1e-6, 1e-7, 0) but **freezing it lost on audio**.
  1e-6 is the only untested-on-audio candidate.
- AdamW 5e-5 (Stability's value) worse. p-full 0.80 vs 0.55: no difference.
- FlashAdamW / torch fused / bitsandbytes 8-bit: identical curves, identical speed.

Records: `runs/ablate/*/tb`.

## Picking a checkpoint: score against the truth, not the seed

Seeds are taken from the start of each track, which is usually an intro. Real
tracks then drop into heavier, bassier material. Scoring by distance to the
*seed* therefore rewards a checkpoint for staying intro-like and punishes the
one that delivers the drop. On loop-0-evancloud this picked step 4500, which
had the thinnest low end of the run; by distance to the real continuation it
was the worst adapter checkpoint and step 1500/3000 were best. `eval_songs.py`
now reports `vs truth` first. Select on it, and listen before shipping.

## Ship

```bash
python shard_delta.py --resume runs/NAME/dit_008000.safetensors --out /tmp/shards
# upload per shard with HF_HUB_DISABLE_XET=1, retry per shard; this link
# sustains ~400 kB/s up and drops long transfers
```

Update the `fcolooo/loop-0` card, tag the training commit on GitHub, and put
the training code under `training/` in the HF repo.
