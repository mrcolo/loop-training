# Known-good configuration

These settings produced the best outpainting we have heard from this project, on
`stabilityai/stable-audio-3-medium` (the adversarially distilled checkpoint).

## Training

| setting | value | why |
| --- | --- | --- |
| base checkpoint | `stable-audio-3-medium` | the ARC-distilled release, not `-base` |
| excerpt length | 190 s (2046 latent frames) | cost per minute of audio is flat with length |
| batch | 1 | maximises optimiser steps at fixed audio throughput |
| steps | **5000** | the checkpoint the alpha sweep and final evaluation used |
| optimiser | AdamW, 8-bit moments | float32 master weights + grads already cost 11.6 GB |
| learning rate | **5e-5**, all parameters | published rate for full Stable Audio finetunes |
| betas / weight decay | (0.9, 0.999) / 1e-3 | same source |
| warmup | 100 steps, linear | |
| objective | rectified flow, loss on generated **and** context regions | matches the reference training step |
| mask mix | 5% segments / 10% full / 70% causal / 15% spans | causal is outpainting; the rest augments context shape |
| CFG dropout | 0.1 | from the model config |
| timesteps | truncated logistic-normal, flipped, length-shifted | inherited from pretraining |
| latents | precomputed, float16 | the autoencoder is 89% of an online step |
| excerpt gate | rms >= 0.15 | skips dead air in a 26 h DJ set, keeps 95% |
| prompt | fixed neutral caption, encoded once | |

## Checkpointing — this one matters

Store the **delta from the base weights in float16**, never the weights in
bfloat16. Mean weight magnitude is 0.049 and the bfloat16 step there is 1.9e-4;
a learned update of 8.9e-5 is *smaller than the rounding step*, so saving weights
in bfloat16 discards most of the finetune and leaves noise. Deltas in float16 sit
2048x above their own rounding floor.

## Sampling — and this one

| setting | value |
| --- | --- |
| schedule | `model.sampling_dist_shift`, **not** `model.dist_shift` |
| sampler | pingpong for `rf_denoiser`, euler for `rectified_flow` |
| steps | 8 |
| cfg scale | 1.0 |
| **update strength (alpha)** | **0.25** |

Alpha is the discovery. Train at 5e-5, which is hot enough for the model to
actually move, then apply a quarter of the learned update at inference. Full
strength hisses: high-frequency energy 0.183 against the source's 0.143. A
quarter gives 0.127, cleaner than the source, while envelope distance improves
from the pretrained model's 0.378 to 0.346. Effective rate is about 1.25e-5.

## Provenance

The evaluated model was a fresh run to 4000 steps, resumed toward 10000 and
paused at the rolling checkpoint saved at step 5000. Every alpha sweep and the
five-context evaluation below used that step-5000 delta at alpha 0.25.

## Measured

| model | energy >8 kHz | envelope distance | contexts won |
| --- | --- | --- | --- |
| truth | 0.143 | — | — |
| pretrained | 0.146 | 0.378 | — |
| finetuned, alpha 0.25 | 0.127 | 0.346 | 3/5 |
