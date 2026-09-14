---
license: other
---

# stems-loop

Full finetune of [Stable Audio 3 Medium](https://huggingface.co/stabilityai/stable-audio-3-medium)
for **outpainting** a single track: give it the first part of the audio, it generates what comes next.

Stable Audio 3 already ships with inpainting conditioning (`inpaint_mask` and
`inpaint_masked_input` as local-additive conditions), and its `CAUSAL_MASK` mode
keeps a random-length prefix and masks everything after it. That *is* outpainting,
so the finetune changes no architecture at all: only the DiT weights move.

## Files

| file | what it does |
| --- | --- |
| `dataset.py` | random fixed-length excerpts, seeked straight out of the source file |
| `train.py` | the finetune: online autoencoding, causal masks, rectified-flow loss |
| `fetch_model.py` | downloads the checkpoint, converting float32 to bfloat16 in flight |

## Run

```bash
python fetch_model.py                      # ~4.6 GB instead of 10.4 GB
python train.py --audio track.flac --prompt "..."
tensorboard --logdir runs
```

## Notes

- **The autoencoder runs online.** Every step encodes fresh excerpts rather than
  reading precomputed latents. It is the honest way to train but it is not free:
  the 0.85B autoencoder costs roughly 1 s per 24 s excerpt, against 0.11 s for a
  DiT forward and backward. Expect the encoder to dominate the step time.
- **Loss is computed only on the masked region**, which is the part the model is
  actually being asked to generate.
- 1.45B parameters are trained in float32 with 8-bit Adam moments, and the frozen
  autoencoder is held in bfloat16. That fits a 24 GB card with room to spare
  (15.7 GiB peak at 47 s excerpts, less at the 24 s default).
- `--p-full` keeps a fraction of each batch fully masked so unconditional
  generation does not drift while the model learns to continue audio.
