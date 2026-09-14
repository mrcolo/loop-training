# stems-loop

Full finetune of [Stable Audio 3 Medium](https://huggingface.co/stabilityai/stable-audio-3-medium)
for **audio outpainting**: give the model the first part of a recording and it generates
what comes next, in the style of whatever you trained it on.

Three files do the work. No training framework, no config trees, no Lightning.

| file | role |
| --- | --- |
| [`dataset.py`](dataset.py) | random fixed-length excerpts, seeked straight out of the source audio |
| [`train.py`](train.py) | the finetune: online autoencoding, causal masks, rectified-flow loss |
| [`fetch_model.py`](fetch_model.py) | downloads the checkpoint, converting float32 to bfloat16 in flight |
| [`sample.py`](sample.py) | outpaint excerpts with a checkpoint, or with the base model for A/B |
| [`encode_latents.py`](encode_latents.py) | encode a recording to a latent stream once |
| [`scan_energy.py`](scan_energy.py) | index a long recording by loudness and sub-bass |

Adding a large corpus? [`ENCODING_BRIEF.md`](ENCODING_BRIEF.md) covers the format,
the scaling numbers, and the mistakes that silently corrupt latents.

A write-up of the method, with figures and the full hyperparameter set, is in
[`paper/stems-loop.pdf`](paper/stems-loop.pdf) ([source](paper/stems-loop.tex)).

---

## How the outpainting works

Stable Audio 3 already ships with inpainting conditioning. Its DiT takes two extra
local-additive conditions, `inpaint_mask` and `inpaint_masked_input`, and its mask
sampler has a `CAUSAL_MASK` mode that keeps a random-length prefix and masks everything
after it.

That is outpainting. So this finetune changes no architecture at all — no new input
channels, no adapter layers, no surgery on the input projection. Only the DiT weights
move. Every step:

1. Draw a random excerpt from the source audio.
2. Encode it with the frozen autoencoder, **online**, at that moment.
3. Keep a random-length prefix as context; mask the rest.
4. Add rectified-flow noise and ask the model to predict the flow.
5. Take the loss **only over the masked region** — the part it is actually being asked to generate.

Because the conditioning already exists in the pretrained model, step zero is a
well-behaved starting point rather than a randomly initialised branch.

### The objective

Stable Audio 3 Medium uses `rf_denoiser`: rectified flow, where

```
x_t    = (1 - t) * z + t * noise
target = noise - z
```

and `t` comes from a truncated logistic-normal sampler, flipped, then warped by the
model's sequence-length-dependent distribution shift. `train.py` reproduces that exactly,
reusing the library's own `truncated_logistic_normal_rescaled` and `dist_shift` rather
than reimplementing them.

---

## Install

Requires a CUDA GPU with 24 GB (see [Hardware](#hardware)) and Python 3.12.

```bash
git clone https://github.com/mrcolo/stems-loop
cd stems-loop
uv venv --python 3.12
uv pip install torch torchaudio --torch-backend=auto
uv pip install diffusers transformers accelerate safetensors soundfile \
               tensorboard einops einops-exts bitsandbytes "setuptools<81"
uv pip install --no-deps git+https://github.com/Stability-AI/stable-audio-3
```

`--no-deps` on the last line is deliberate: the library pins `torch==2.7.1`, and
reinstalling a pinned torch on top of a working CUDA build wastes several gigabytes
for no benefit. It runs fine against torch 2.10.

`setuptools<81` is needed because TensorBoard still imports `pkg_resources`, which
setuptools 81 removed.

---

## Getting the weights

The published checkpoint is **9.2 GB of float32**. `fetch_model.py` streams it and
converts every tensor to bfloat16 as it arrives, so the float32 file never lands on disk:

```bash
echo "hf_YOUR_TOKEN" > ~/.hf_token     # the repo is gated; accept its terms first
python fetch_model.py                  # writes ~4.6 GB instead of 9.2 GB
```

It resumes at the last whole tensor if the connection drops, which matters on a slow
link — this is a two-hour download at 1 MB/s.

Training keeps master weights in float32 in memory regardless, so the only cost of
bfloat16 storage is a single rounding at load. If you have the disk and want to avoid
even that, download the original with `huggingface-cli` and point `--model` at it;
nothing else changes.

---

## Training

```bash
python train.py --audio your-audio.flac
tensorboard --logdir runs
```

The source can be one long file or many. There is no preprocessing step, no manifest,
and no latent cache to build: `dataset.py` opens the file, seeks to a random offset and
reads only the frames it needs. A 26-hour, 10.5 GB flac costs no memory and 31 ms per
excerpt.

### Options

| flag | default | meaning |
| --- | --- | --- |
| `--audio` | `audio.flac` | source audio; pass it more than once for several files |
| `--model` | `models/stable-audio-3-medium` | where `fetch_model.py` put the weights |
| `--out` | `runs/outpaint` | checkpoints, demo audio and TensorBoard logs |
| `--prompt` | `""` | text condition; fixed for the whole run, encoded once |
| `--seconds` | `24.0` | excerpt length, 258 latent frames |
| `--steps` | `3000` | training steps |
| `--batch` | `4` | excerpts per step |
| `--lr` | `1e-5` | AdamW learning rate |
| `--warmup` | `100` | linear warmup steps |
| `--p-full` | `0.1` | fraction of each batch fully masked instead of causally masked |
| `--workers` | `2` | dataloader workers |
| `--val-every` | `50` | steps between frozen validation passes |
| `--demo-every` | `250` | steps between generated demos |
| `--demo-at` | `3600.0` | offset in seconds of the reference excerpt used for demos |
| `--demo-steps` | `8` | sampler steps for demos; the model is distilled for few-step sampling |
| `--save-every` | `500` | steps between checkpoints |
| `--resume` | — | path to a `dit.safetensors` to continue from |

### Why 24 seconds

The model's schedule shift is defined between 256 and 4096 latent frames. At its 4096
sample downsampling ratio, 24 s is 258 frames — just inside the lower bound, and the
cheapest excerpt that stays in the regime the model was trained for. Longer excerpts
work; they cost encode time roughly linearly.

---

## Results

Evaluated by outpainting 158 s from a 32 s seed, at the 8 sampler steps this
checkpoint is distilled for, across five contexts spread over a 26.4 h source.
Envelope distance is the mean absolute difference between the generated region's
24-band log-spectral envelope and the true continuation's.

| model | mean envelope distance | level vs truth | wins |
| --- | --- | --- | --- |
| pretrained | 0.378 | 0.89x | — |
| finetuned, 4000 steps | **0.364** | 0.87x | 3/5 |

### Three defects that made this look impossible

Getting here required fixing three things, each of which alone produced a
convincing but false negative result.

**The sampler used the wrong schedule.** The model carries two distribution
shifts: `dist_shift` for training and `sampling_dist_shift` for inference, the
latter defaulting to a LogSNR schedule when the config omits it, as this one
does. Their `generate()` uses the sampling one. Using the training schedule at
inference moved the base model's output level to 1.28x the source; correcting it
gives 0.89x.

**Checkpoints quantised the finetune away.** Storing weights in bfloat16 sounds
harmless until you compare magnitudes: mean weight 0.049, bfloat16 step at that
magnitude 1.9e-4, and the learned update after 2000 steps at a low learning rate
averaged 8.9e-5 — *smaller than the rounding step*. The saved file was roughly
half signal and half noise, and random weight noise blurs a generative model,
which shows up as quiet, over-smoothed audio. Checkpoints now store the **delta**
from the base weights in float16; because float16 is relative-precision, small
values survive essentially exactly at the same file size.

**The learning rate was 25x too low.** 2e-6, derived by equating Adam to the Muon
step size in the *pretraining* config. Published full finetunes of Stable Audio
use AdamW at 5e-5, betas (0.9, 0.999), weight decay 1e-3. At 5e-5 the update is
3.2% of weight magnitude and 2048x above the storage floor, against 0.47x before.

The lesson worth keeping: validation loss fell smoothly throughout all of the
broken configurations. It never once indicated that the saved model was noise.

## Reading the results

**`val/loss` is the number that matters.** Training loss is a single batch at a single
randomly drawn timestep, and rectified-flow loss depends strongly on that timestep, so
it swings by 0.2 between logs whether or not the model is improving. Reading a trend
from it is guessing.

`val/loss` holds everything fixed — the same four excerpts, the same masks, the same
four timesteps, the same noise — so successive values are directly comparable. From a
real run on 26 hours of source audio:

| step | 1 | 50 | 100 | 150 | 200 | 250 | 300 | 350 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `val/loss` | 1.1005 | 0.9730 | 0.8890 | 0.8753 | 0.8722 | 0.8648 | 0.8633 | 0.8595 |

Monotonic, which is what learning looks like. Over the same window `train/loss` bounced
between 0.63 and 0.98 and told you nothing.

Also logged: `train/grad_norm`, `train/lr`, `train/sec_per_step`, `train/gpu_gb`, and
demo audio under the **Audio** tab — the model's continuation of a fixed reference
excerpt, plus the ground truth logged once at step 1 for comparison.

---

## Hardware

Measured on a single RTX 3090, batch 4, 24 s excerpts:

| quantity | value |
| --- | --- |
| trainable parameters | 1.453 B (the DiT) |
| frozen parameters | 0.852 B (the autoencoder) |
| peak VRAM | 15.7 GiB |
| step time | 4.7 s |
| autoencoder encode | 4.0 s of that |
| DiT forward + backward | 0.45 s of that |
| excerpt read from disk | 0.03 s, on 2 workers |

**The autoencoder dominates.** Encoding costs roughly nine times a DiT step. That is the
price of keeping it online, and it is a deliberate choice: Stability's own config trains
with `pre_encoded: true` against a precomputed latent cache. Precomputing latents for 26
hours of audio would be faster per step but turns a zero-setup script into a pipeline
with a cache to build, invalidate and store. If you want that trade, encode once and
feed latents in directly — the loop below the encode call does not care where `z` came from.

### Fitting 1.45 B trainable parameters in 24 GB

Float32 master weights and float32 gradients alone are 11.6 GB. Three things make the
rest fit:

- **8-bit Adam moments** (`bitsandbytes`) instead of float32, saving 8.7 GB.
- **The frozen autoencoder held in bfloat16**, saving 1.7 GB. It never receives gradients.
- **Gradient checkpointing**, which the library enables by default.

The text encoder is dropped entirely after startup. The prompt and excerpt length are
fixed for a run, so its output is a constant: encode once, free the 1.2 GB, reuse the
tensor every step.

---

## Resuming

```bash
python train.py --resume runs/outpaint/dit.safetensors --steps 12000
```

Checkpoints hold the DiT in bfloat16, about 2.9 GB. Optimizer state is deliberately not
saved — float32 Adam moments for 1.45 B parameters are 11.6 GB per checkpoint, which is
not worth the disk for a finetune that restarts with warmup anyway.

---

## Notes and caveats

- **`flash_attn` is not required.** Without it the library falls back to SDPA and torch
  logs a wall of dynamo warnings about failing to compile `flex_attention`. They are
  noise; training is unaffected. Installing `python3-dev` lets the compile succeed.
- **`--p-full` exists to stop drift.** Training exclusively on causal masks teaches the
  model to continue audio but lets its unconditional generation decay. Keeping a tenth of
  each batch fully masked preserves it.
- **Loss is masked, not global.** Including the context region would let the model lower
  the loss by copying input it was already given, which is not the task.
- **Weights are licensed by Stability AI** under the Stable Audio Community License, and
  the bundled T5Gemma text encoder under the Gemma Terms of Use. This repository is
  training code only and ships no weights. Check both before any commercial use.

## License

MIT for the code in this repository. The model weights it trains are not covered by it.
