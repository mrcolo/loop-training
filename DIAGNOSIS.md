# Why finetuning damaged the model

## The measurement

Ping-pong sampling, which the released `stable-audio-3-medium` is post-trained
for, rebuilds the signal from scratch at every one of its 8 steps:

```
denoised = x - t * model(x, t)
x        = (1 - t_next) * denoised + t_next * fresh_noise
```

So `denoised` has to be a plausible full-scale piece of audio on its own, at
every noise level. A model trained with mean-squared error does not do that: the
MSE optimum is the conditional mean `E[x0 | x_t]`, whose scale shrinks in
proportion to how much it actually knows. The signature is exact — for a mean
predictor, `std(denoised) / std(z)` equals the correlation with the truth.

`probe_denoiser.py` measures both, on a real excerpt, unconditional mask:

| t | released checkpoint |  | after 1000 MSE steps |  |
| --- | --- | --- | --- | --- |
| | std ratio | corr | std ratio | corr |
| 0.95 | **0.832** | 0.321 | **0.551** | 0.502 |
| 0.90 | 0.896 | 0.565 | 0.686 | 0.662 |
| 0.80 | 0.964 | 0.719 | 0.784 | 0.785 |
| 0.60 | 0.999 | 0.872 | 0.892 | 0.899 |
| 0.40 | 1.004 | 0.946 | 0.948 | 0.954 |
| 0.20 | 1.001 | 0.985 | 0.983 | 0.986 |

The released checkpoint emits a full-scale sample even where it knows almost
nothing: at t = 0.95 it correlates 0.32 with the truth and still outputs 0.83x
its amplitude. After a thousand MSE steps the two columns are equal at every
noise level. That is a conditional-mean predictor, and it is what MSE training
converges to by construction.

At the first ping-pong step the generated signal has lost a third of its
amplitude while the sampler keeps injecting noise at full scale. That is the
"monotone, with a noise feeling" the outputs had.

Note the correlation *improved*, 0.32 to 0.50. The model was getting better at
the objective it was given the whole time. Validation loss fell monotonically
throughout. Neither number could have revealed this.

## Why the recipe could not be fixed by tuning

The adversarial post-training stage exists precisely to remove this property.
From the Stable Audio 3 report, on the distillation stage that precedes it: "the
MSE objective causes the student to regress toward the conditional mean
E[x0 | x_t], producing outputs that lack fine-grained detail". The adversarial
stage works "by supplanting the MSE-based conditional mean loss (of both flow
matching and distillation warmup) with an adversarial loss".

Running MSE on those weights re-imposes the exact loss the stage was run to
remove. No learning rate makes that objective correct; a lower one only makes
the regression slower.

## What the tooling says

Stability's own trainer refuses the post-trained checkpoint outright:

```python
if model_name not in base_models:
    raise ValueError(f"LoRA training requires a base model. Got '{model_name}'")
```

Their MLX trainer states it directly: "Training uses the BASE checkpoint
(stabilityai/stable-audio-3-*-base, rectified_flow), not the shipped ARC weights
inference uses." Their docs add the part that matters for shipping: "LoRAs are
trained on the base checkpoint. Once trained, they can be applied to the
post-trained model and will work as expected."

The same holds in images. The FLUX.1-schnell training adapter card: a distilled
model is "impossible to train on directly because every step you train breaks
down the compression more and more".

## What is actually wrong, ranked

1. **The checkpoint.** MSE on adversarially post-trained weights. Measured above.
   Fix: train `stable-audio-3-medium-base`, and add the resulting delta to the
   post-trained weights at inference, which is what everyone else does.
2. **The optimizer.** The config asks for Muon on the attention and feed-forward
   matrices at 1e-3, with AdamW at 5e-5 on everything else, betas (0.9, 0.95),
   weight decay 0.01. AdamW at a single hand-picked rate is not an approximation
   of an orthogonalised update; it is too large for some matrices and too small
   for others simultaneously.
3. **No weight averaging.** The config sets `use_ema: true` with beta 0.9995 and
   a power-law warmup, and inference uses the average. There was none.
4. **Mask mix.** The reference draws full / segments / causal at 0.8 / 0.1 / 0.1.
   This run used 0.40 full, which lets unconditional generation decay faster.

## What was *not* wrong

The two-term loss. The report specifies exactly it: "the loss is split into two
independently averaged terms: a generation loss over the inpainted embeddings
(m=0) and a context preservation loss over the kept audio (m=1)", and the
shipped config sets `mask_loss_weight: 1.0`. `gen_loss + ctx_loss` is that.

The timestep sampler, the velocity target, the distribution shift, the
conditioning-dropout probability and the inference schedule all match the
reference as well.
