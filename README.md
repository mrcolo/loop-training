# stems-loop

Full finetune of [Stable Audio 3 Medium](https://huggingface.co/stabilityai/stable-audio-3-medium)
for **audio outpainting**: give the model the first 30–45 seconds of a recording and it
generates up to three more minutes in the style of whatever it was trained on.

Six files do the work. No training framework, no config trees, no Lightning.

| file | role |
| --- | --- |
| [`dataset.py`](dataset.py) | windows of a latent stream, or excerpts seeked out of the source audio |
| [`train.py`](train.py) | the finetune: inpainting masks, rectified-flow loss, Muon + AdamW, EMA |
| [`muon.py`](muon.py) | the optimizer the model is actually trained with |
| [`fetch_model.py`](fetch_model.py) | downloads a checkpoint, converting float32 to bfloat16 in flight |
| [`sample.py`](sample.py) | outpaint excerpts with a checkpoint, or with the released model for A/B |
| [`probe_denoiser.py`](probe_denoiser.py) | tells you whether a checkpoint predicts a sample or the mean |
| [`encode_latents.py`](encode_latents.py) | encode a recording to a latent stream once |
| [`scan_energy.py`](scan_energy.py) | index a long recording by loudness and sub-bass |

Want to generate tracks rather than train? [`GENERATING.md`](GENERATING.md) is the
end-to-end recipe, including the weights at
<https://huggingface.co/fcolooo/loop-0>.

Adding a large corpus? [`ENCODING_BRIEF.md`](ENCODING_BRIEF.md) covers the format,
the scaling numbers, and the mistakes that silently corrupt latents.

A write-up of the method, with figures and the full hyperparameter set, is in
[`paper/stems-loop.pdf`](paper/stems-loop.pdf) ([source](paper/stems-loop.tex)).

---

## Train the base checkpoint, ship on the post-trained one

This is the single most important thing in the repository, so it goes first.

Stability publishes two weights for this model. `stable-audio-3-medium-base` is the
rectified-flow model: it predicts a velocity, it is sampled with an ODE solver at about
50 steps with classifier-free guidance. `stable-audio-3-medium` is that model after
adversarial post-training: it predicts a *sample* rather than a conditional mean, and it
is sampled with ping-pong in 8 steps at guidance 1.

**A mean-squared-error finetune is only valid on the first one.** The post-trained
weights were produced by replacing exactly that loss with an adversarial one. Training
them with it again walks the model back, and the sampler they ship with cannot tolerate
the result.

`probe_denoiser.py` measures it directly. Ping-pong rebuilds the signal every step,
so `denoised = x - t·model(x, t)` has to be a full-scale plausible clip at every noise
level. For a conditional-mean predictor it is not: its scale collapses to exactly its
correlation with the truth.

| t | released weights |  | after 1000 MSE steps |  |
| --- | --- | --- | --- | --- |
| | std ratio | corr | std ratio | corr |
| 0.95 | **0.832** | 0.321 | **0.551** | 0.502 |
| 0.90 | 0.896 | 0.565 | 0.686 | 0.662 |
| 0.80 | 0.964 | 0.719 | 0.784 | 0.785 |
| 0.60 | 0.999 | 0.872 | 0.892 | 0.899 |
| 0.20 | 1.001 | 0.985 | 0.983 | 0.986 |

The released model emits a full-scale sample where it knows almost nothing. A thousand
MSE steps later the two columns are equal at every noise level, and a third of the
amplitude is gone at the first ping-pong step while the sampler still injects noise at
full scale. The audio goes flat and hissy. Meanwhile the correlation *improved* and
validation loss fell the whole way, because the model was getting better at the objective
it was given. No scalar in the training log can see this. [`DIAGNOSIS.md`](DIAGNOSIS.md)
has the full account.

**The transplant is measured, not assumed.** Their advice is written for LoRA
adapters, which are small by construction, and a full finetune is not: after 1000
steps our update is already 0.85x the size of the entire post-training offset.
So `probe_transplant.py` streams the post-trained weights from the Hub, adds our
delta, and reruns the diagnostic. At 3000 steps:

| | t=0.95 | t=0.9 | t=0.6 |
| --- | --- | --- | --- |
| base | 0.535 / 0.533 | 0.605 / 0.605 | 0.869 / 0.873 |
| base + delta | 0.578 / 0.558 | 0.639 / 0.626 | 0.879 / 0.877 |
| post-trained | 0.920 / 0.393 | 0.937 / 0.483 | 0.999 / 0.842 |
| **post-trained + delta** | **0.965 / 0.408** | 0.952 / 0.506 | 0.995 / 0.849 |

Amplitude over correlation. The base model sits on the conditional-mean line and
stays there, which is correct. The post-trained model separates, and keeps
separating after the delta is added. Few-step sampling survives intact, and the
correlation rises, so the finetune's knowledge transfers rather than merely
surviving.

Stability's own trainer refuses the post-trained checkpoint outright:

```python
if model_name not in base_models:
    raise ValueError(f"LoRA training requires a base model. Got '{model_name}'")
```

and their documentation says what to do with the result: "LoRAs are trained on the base
checkpoint. Once trained, they can be applied to the post-trained model and will work as
expected." Checkpoints here store the **delta** from the base weights, so the same
applies: train against base, add the delta to the post-trained weights, sample in 8 steps.

The same rule holds in images. From the FLUX.1-schnell training adapter: a distilled
model is "impossible to train on directly because every step you train breaks down the
compression more and more".

---

## How the outpainting works

Stable Audio 3 already ships with inpainting conditioning. Its DiT takes two extra
local-additive conditions, `inpaint_mask` and `inpaint_masked_input`, and its mask
sampler has a `CAUSAL_MASK` mode that keeps a random-length prefix and masks everything
after it.

That is outpainting. So this finetune changes no architecture at all — no new input
channels, no adapter layers, no surgery on the input projection. Only the DiT weights
move. Every step:

1. Draw a random window from the precomputed latent stream.
2. Choose a mask: fully masked, scattered segments, or a causal prefix.
3. Add rectified-flow noise and ask the model to predict the velocity.
4. Take two losses — one over the region being generated, one over the context.

### The objective

Rectified flow, exactly as the report states it:

```
x_t    = (1 - t) * z + t * noise
target = noise - z
```

`t` comes from a truncated logit-normal, flipped, then warped by the model's
sequence-length-dependent distribution shift. `train.py` reuses the library's own
`truncated_logistic_normal_rescaled` and `dist_shift` rather than reimplementing them.

The loss is **two independently averaged terms**, not one. From the report: "the loss is
split into two independently averaged terms: a generation loss over the inpainted
embeddings (m=0) and a context preservation loss over the kept audio (m=1)." The shipped
config sets `mask_loss_weight: 1.0`, so they are summed. Pooling them into a single ratio
instead would weight each region by its size, which quietly down-weights exactly the
long-context examples outpainting is about.

### The optimizer

The config asks for `MuonAdamW`, and that is not a detail. Muon orthogonalises the
momentum before applying it, so every matrix takes a step of the same spectral size no
matter how its gradients are scaled. AdamW does not, and substituting it means choosing
one learning rate that is simultaneously too large for some matrices and too small for
others. [`muon.py`](muon.py) is a 90-line implementation; fused projections are split
before orthogonalisation, which is what the config's `fused_layer_patterns` is for.

| | reference pretraining | here |
| --- | --- | --- |
| Muon, attention and feed-forward matrices | 1e-3, momentum 0.95 | 2e-4, momentum 0.95 |
| AdamW, everything else | 5e-5, betas (0.9, 0.95), wd 0.01 | 1e-5, same |
| schedule | inverse power law, γ=1e6, power 0.5 | same, plus 200-step warmup |
| EMA | 0.9995, power-law warmup | same, updated every 8 steps |

Learning rates are a fifth of the pretraining values because this is a finetune of a
converged model, not a run from scratch.

### The mask mix

The reference draws full / segments / causal at 0.8 / 0.1 / 0.1, with the causal prefix a
uniform fraction of the window. Two deliberate changes:

- **Causal gets a larger share** (0.35 against 0.1) because outpainting is the task. The
  fully masked share stays high at 0.55, because that is what keeps unconditional
  generation from decaying.
- **Causal context is drawn in seconds, not as a fraction.** A 30 s seed is 16% of a
  190 s window and 8% of a 380 s one, so a uniform fraction spends almost nothing on the
  context lengths people actually ask for. Sampling seconds directly targets the 20–60 s
  band at every window length.

---

## Install

Requires a CUDA GPU with 24 GB and Python 3.12.

```bash
git clone https://github.com/mrcolo/loop-training
cd stems-loop
uv venv --python 3.12
uv pip install torch torchaudio --torch-backend=auto
uv pip install diffusers transformers accelerate safetensors soundfile \
               tensorboard einops einops-exts bitsandbytes "setuptools<81"
uv pip install --no-deps git+https://github.com/Stability-AI/stable-audio-3
```

`--no-deps` on the last line is deliberate: the library pins `torch==2.7.1`, and
reinstalling a pinned torch on top of a working CUDA build wastes several gigabytes for
no benefit. It runs fine against torch 2.10. `setuptools<81` is needed because
TensorBoard still imports `pkg_resources`, which setuptools 81 removed.

---

## Getting the weights

Both published checkpoints are **9.2 GB of float32**. `fetch_model.py` streams them and
converts every tensor to bfloat16 as it arrives, so the float32 file never lands on disk.

```bash
echo "hf_YOUR_TOKEN" > ~/.hf_token     # both repos are gated; accept their terms first
python fetch_model.py                                   # released weights + autoencoder
python fetch_model.py --base --only model. \
       --name dit_base.safetensors \
       --out models/stable-audio-3-medium-base          # base transformer only, 2.9 GB
```

`--only model.` keeps just the transformer. The autoencoder and the duration conditioner
are identical across the two checkpoints, so there is no reason to store them twice, and
on a full disk it is the difference between fitting and not. It resumes at the last whole
tensor if the connection drops.

---

## Training

```bash
python encode_latents.py --audio your-audio.flac --out latents.npy
python scan_energy.py    --audio your-audio.flac --out energy_index.npz
python train.py --latents latents.npy \
                --dit models/stable-audio-3-medium-base/dit_base.safetensors \
                --objective rectified_flow
tensorboard --logdir runs
```

The autoencoder is 89% of a training step when it runs online, so encoding once buys
roughly **7x more optimiser steps for a fixed time budget**. 26 hours of audio becomes a
500 MB memory-mapped array.

`--dit` overlays a transformer-only checkpoint on top of whatever `--model` provides, and
refuses to start if a single tensor fails to land — otherwise a mismatched key trains from
random initialisation and looks entirely normal while doing it.

### Options

| flag | default | meaning |
| --- | --- | --- |
| `--latents` | — | precomputed stream; falls back to `--audio` and online encoding |
| `--dit` | — | transformer-only checkpoint to overlay, e.g. the base weights |
| `--objective` | from config | `rectified_flow` for base, `rf_denoiser` for the released weights |
| `--seconds` | `47 95 190 380` | window lengths sampled per step |
| `--batch` / `--accum` | `1` / `4` | microbatch and gradient accumulation |
| `--muon-lr` | `2e-4` | attention and feed-forward matrices |
| `--adam-lr` | `1e-5` | everything else |
| `--ema` | `0.9995` | weight average; this is what gets saved |
| `--p-full` | `0.55` | fully masked share |
| `--p-segments` | `0.10` | scattered segments |
| `--ctx-min` / `--ctx-max` | `20` / `60` | causal context length, in seconds |
| `--demo-steps` | 50 or 8 | follows the objective |
| `--demo-cfg` | 4.0 or 1.0 | follows the objective |

### Why several window lengths

The model conditions on duration and its schedule shift is defined between 256 and 4096
latent frames, which at a 4096x downsampling ratio is 23.8 s to 380.4 s. Training at one
length freezes that pathway; the run samples 47, 95, 190 and 380 s.

---

## Results

Trained 8000 steps on 26.4 h of one DJ set, then merged onto the post-trained
weights. Evaluated by outpainting 160 s from a 30 s seed at four offsets drawn
from the energy gate, two seeds each, **both models on identical trajectories**.

| model | sampler | distance to real continuation | level | level spread |
| --- | --- | --- | --- | --- |
| stock, base weights | 50 steps | 0.5547 | 1.10x | 0.11 |
| this finetune, base weights | 50 steps | 0.3817 | 1.15x | 0.16 |
| stock, post-trained | 8 steps | 0.4537 | 1.03x | 0.24 |
| **this finetune, merged** | **8 steps** | **0.2978** | 1.11x | **0.09** |
| the real recording | — | — | 1.04x | 0.33 |

**34% closer than stock at the same 8-step cost, winning 7 of 8 paired renders.**
It also beats its own 50-step base version, 0.2978 against 0.3817, because the
post-trained weights are simply better at few-step sampling than base weights are
at many-step sampling.

The consistency column matters as much as the mean. Stock produced one render at
0.47x level, a near-collapse; this model's worst is 0.96x, and its spread is a
third of stock's. It is not only closer on average, it fails less.

The denoiser probe on the merged file, with 30 s of context at t=0.95:

| | stock | merged |
| --- | --- | --- |
| amplitude of the one-step estimate | 0.959 | 0.983 |
| correlation with the truth | 0.514 | 0.643 |

Still a sample predictor, which is what makes 8 steps possible, and 25% better
correlated with what the track actually does next.

### When to stop

Training was stopped at 8000 steps because a four-offset evaluation put step 8000
at 0.3817 against step 4000's 0.3660 -- inside the noise, marginally behind. The
first 2000 steps bought three times what the last 2000 did. Validation loss was
still falling when the run stopped, which is exactly why it is not the instrument
to stop on.

## Reading the results

**`val/loss` is not sufficient.** It was monotone through every broken configuration this
project went through, including one that was measurably destroying the model. Listen to
the demos, and run `probe_denoiser.py` on a checkpoint before believing it.

Training loss is a single batch at a single randomly drawn timestep, and rectified-flow
loss depends strongly on that timestep, so it swings by 0.2 between logs whether or not
the model is improving. `val/loss` holds everything fixed — the same excerpts, masks,
timesteps and noise — so successive values are comparable.

Also logged: `train/loss_gen`, `train/loss_context`, `train/grad_norm`, `train/muon_lr`,
`train/sec_per_step`, `train/gpu_gb`, and demo audio under the **Audio** tab, rendered
from the EMA weights.

---

## Hardware

Single RTX 3090. Precomputed latents, batch 1 with 4-step accumulation:

| quantity | value |
| --- | --- |
| trainable parameters | 1.453 B |
| of which on Muon | 1.42 B |
| step time | ~0.9 s per microbatch |
| latent read from disk | 0.03 s, on 2 workers |

### Fitting 1.45 B trainable parameters in 24 GB

Float32 master weights and float32 gradients alone are 11.6 GB. The rest fits because:

- **Muon's momentum buffer is bfloat16**, 2.9 GB instead of 5.8 GB. Only the direction of
  the orthogonalised result survives, so the buffer's precision barely matters.
- **8-bit AdamW moments** for the parameters Muon does not take.
- **The EMA lives in host memory** and is updated every 8 steps, which costs a copy and
  buys back 2.9 GB of VRAM.
- **Gradient checkpointing**, which the library enables by default.

The text encoder is dropped entirely after startup. The prompt is fixed for a run, so its
output is a constant: encode once, free the 1.2 GB, reuse the tensor every step. The
duration conditioner is a few thousand parameters and stays resident, because training at
one duration would freeze that pathway.

---

## Shipping

Training produces a delta from the base transformer. `ship.py` adds it to the
post-trained transformer and writes one file you can sample in eight steps:

```bash
python ship.py --resume runs/base/dit.safetensors \
               --drop-base models/stable-audio-3-medium-base/dit_base.safetensors

python sample.py --audio YOUR.flac \
                 --dit models/stable-audio-3-medium/dit_shipped.safetensors \
                 --objective rf_denoiser --steps 8 --cfg 1.0 \
                 --context 30 --seconds 190
```

The post-trained transformer is streamed from the Hub and merged in memory, so
no intermediate copy is written. `--drop-base` removes the base transformer
first, which is usually the only way the result fits on a full disk; it is safe
once training has finished, since the delta is all that is needed from then on.
`--alpha` scales the delta if you want something between the two models.

**Check the margin before you ship.** `delta_margin.py` fetches a single
transformer layer of the post-trained checkpoint by byte range, about 100 MB,
and reports how large the finetune's update is next to the post-training offset.
It was 0.85x at step 1000 and 1.46x at step 4000, growing roughly as the square
root of steps. The transplant was verified intact at 1.3x; re-run
`probe_transplant.py` before shipping if the margin has grown much beyond that.

## Resuming

```bash
python train.py --resume runs/outpaint/dit.safetensors --steps 20000 [...]
```

Checkpoints hold the **delta** from the base weights in float16, about 2.9 GB. Storing the
weights themselves does not work: the mean weight is 0.049, a bfloat16 step at that
magnitude is 1.9e-4, and a learned update can be smaller than that — the file comes back
roughly half signal and half rounding noise, and random weight noise blurs a generative
model. Float16 is relative-precision, so the delta survives essentially exactly at the
same file size. It is also the right format for shipping, since the delta is what gets
added to the post-trained weights.

Optimizer state is deliberately not saved. Float32 moments for 1.45 B parameters are
several gigabytes per checkpoint, which is not worth the disk for a finetune that restarts
with warmup anyway.

---

## Notes and caveats

- **`flash_attn` is not required.** Without it the library falls back to SDPA and torch
  logs a wall of dynamo warnings about failing to compile `flex_attention`. They are
  noise; training is unaffected. Installing `python3-dev` lets the compile succeed.
- **Weights are licensed by Stability AI** under the Stable Audio Community License, and
  the bundled T5Gemma text encoder under the Gemma Terms of Use. This repository is
  training code only and ships no weights. Check both before any commercial use.

## License

MIT for the code in this repository. The model weights it trains are not covered by it.
