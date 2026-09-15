# Configuration

Superseding an earlier version of this file that recommended training the
adversarially post-trained release. That was wrong, measurably so; see
[DIAGNOSIS.md](DIAGNOSIS.md). Train the base checkpoint and transplant the delta.

## Training

| setting | value | why |
| --- | --- | --- |
| transformer | `stable-audio-3-medium-base` | the only checkpoint a squared-error objective is valid on |
| autoencoder, conditioner | from `stable-audio-3-medium` | identical across the two releases |
| objective | rectified flow, velocity target | `--objective rectified_flow` |
| loss | generation + context, each averaged separately | the report specifies both terms; `mask_loss_weight: 1.0` |
| window lengths | 47 / 95 / 190 / 380 s, drawn per excerpt | duration is a conditioning input; one length freezes it |
| batch | 1 with 4-step accumulation | |
| optimiser | Muon on attention and feed-forward matrices, AdamW on the rest | what `MuonAdamW` in the config means |
| Muon | lr 2e-4, momentum 0.95, 1.36 B params | config pretrains at 1e-3; a fifth of that for a finetune |
| AdamW | lr 1e-5, betas (0.9, 0.95), wd 0.01, 94 M params | config pretrains at 5e-5 |
| schedule | 200-step warmup, then inverse power law, gamma 1e6, power 0.5 | from the config |
| EMA | 0.9995, power-law warmup, host-resident | the config sets `use_ema: true` and samples from the average |
| mask mix | 55% full / 10% segments / 35% causal | reference is 80/10/10; causal is raised because outpainting is the task |
| causal context | uniform 20-60 s, in **seconds** | a uniform *fraction* spends almost nothing on the lengths people ask for |
| minimum generated span | 15 s | otherwise a 60 s draw against a 47 s window teaches copying |
| CFG dropout | 0.1 | from the config |
| timesteps | truncated logit-normal, flipped, length-shifted | inherited from pretraining |
| grad clip | 1.0 | |
| latents | precomputed, float16 | the autoencoder is 89% of an online step |
| excerpt gate | rms >= 0.15 | skips dead air in a 26 h DJ set, keeps 95% |
| peak memory | 20.8 GiB | RTX 3090, 4.9 s per optimiser step |

## Checkpointing

Store the **delta from the base weights in float16**, never the weights in
bfloat16. Mean weight magnitude is 0.049 and the bfloat16 step there is 1.9e-4;
a learned update can be smaller than that rounding step, so saving weights in
bfloat16 discards most of the finetune and leaves noise. Deltas in float16 sit
far above their own rounding floor, at the same file size. The delta is also the
format shipping needs, so this costs nothing.

Save the **EMA**, not the live weights.

## Sampling

The two checkpoints want different samplers, and getting this wrong looks exactly
like a broken model.

| | base | post-trained + delta |
| --- | --- | --- |
| objective | `rectified_flow` | `rf_denoiser` |
| sampler | Euler | ping-pong |
| steps | 50 | 8 |
| guidance | 4.0 | 1.0 |

Both use `model.sampling_dist_shift`, **not** `model.dist_shift`. They are
different objects; the model defaults the sampling one to a LogSNR schedule when
the config omits it, as this one does. Using the training shift at inference
warps the whole trajectory.

Keep the transformer in float32 and let autocast handle the matmuls. Casting its
weights to bfloat16 is harmless for one forward pass and compounds badly across a
50-step solve.

## Margins to watch

| quantity | at step 1000 | at step 4000 |
| --- | --- | --- |
| update size vs the post-training offset | 0.85x | 1.46x |
| inpainting-conditioning bias movement | 7.6% | -- |

The first decides whether the transplant still works; `delta_margin.py` measures
it for about 100 MB of traffic. The transplant was verified intact at 1.3x. The
second is the fastest-moving parameter in the model because it is a tiny vector;
it ran away to 94% under an earlier uniform learning rate.
