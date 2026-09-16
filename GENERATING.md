# Generating tracks

Everything below is what produced the audio in this project: 30 seconds of a real
recording as a seed, up to 350 more seconds generated, sampled in 8 steps.

Weights: **https://huggingface.co/fcolooo/loop-0** (private)

---

## What you are assembling

Three pieces have to come together, and it is worth knowing why.

| piece | where from | why |
| --- | --- | --- |
| autoencoder + duration conditioner | `stabilityai/stable-audio-3-medium` | turns audio into latents and back; identical across both releases |
| post-trained transformer | `stabilityai/stable-audio-3-medium` | the checkpoint that samples in 8 steps |
| the finetune | `fcolooo/loop-0`, as a delta | what makes it sound like the target material |

The finetune is stored as a **delta**, the difference from the *base*
transformer, because that is the only checkpoint a squared-error objective can
legitimately train (see [DIAGNOSIS.md](DIAGNOSIS.md)). You add that delta to the
*post-trained* transformer at inference. This is Stability's own prescribed path
and it is measured to work: the merged model keeps the few-step behaviour intact.

---

## One-time setup

```bash
git clone https://github.com/mrcolo/loop-training && cd loop-training
uv venv --python 3.12
uv pip install torch torchaudio --torch-backend=auto
uv pip install diffusers transformers accelerate safetensors soundfile \
               tensorboard einops einops-exts bitsandbytes "setuptools<81"
uv pip install --no-deps git+https://github.com/Stability-AI/stable-audio-3

echo "hf_YOUR_TOKEN" > ~/.hf_token          # both repos are gated; accept their terms

# the autoencoder and conditioner
python fetch_model.py

# the finetune delta, ~2.9 GB in nine shards
python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
snapshot_download("fcolooo/loop-0", local_dir="loop-0",
                  allow_patterns=["delta-*", "*.index.json"],
                  token=Path.home().joinpath(".hf_token").read_text().strip())
PY
python - <<'PY'   # reassemble the shards into one file
from pathlib import Path
from safetensors.torch import load_file, save_file
sd = {}
for f in sorted(Path("loop-0").glob("delta-*.safetensors")):
    sd.update(load_file(str(f)))
save_file(sd, "runs/base/dit.safetensors", metadata={"step": "12000"})
print(len(sd), "tensors")
PY
```

**Cache the post-trained transformer once.** It is 2.9 GB, it never changes, and
without a local copy every generation re-downloads 9.2 GB. The first run below
fetches and saves it; everything after is instant. This one detail is the
difference between 30 seconds and 30 minutes per track.

---

## Generating

### A folder of tracks, with a stock comparison

This is what produced the ALESSIO and Wings renders.

```bash
python eval_songs.py \
    --songs /path/to/folder-of-audio \
    --resume runs/base/dit.safetensors \
    --out my_renders \
    --seconds 190 --context 30 --steps 8 --cfg 1.0 --seed 0 \
    --stock
```

It takes the first `--context` seconds of every file as the seed, generates out
to `--seconds`, writes an mp3 per track, and prints a distance and level per
track plus a mean.

`--stock` renders the released model on **identical noise** as well, so each
track gives you a real A/B rather than a number in isolation. Drop it if you only
want your own.

### One file

```bash
python sample.py --audio YOUR.flac \
    --dit models/stable-audio-3-medium/dit_arc.safetensors \
    --resume runs/base/dit.safetensors \
    --objective rf_denoiser \
    --out renders --tag mine \
    --seconds 190 --context 30 --at 0 --steps 8 --cfg 1.0 --seed 0
```

`--at` takes one or more offsets in seconds, so one invocation can seed from
several points of a long recording.

---

## The settings that matter

| flag | use | why |
| --- | --- | --- |
| `--seconds` | 47 to 380 | model ceiling is 380.4 s. 190 and 320 both tested |
| `--context` | 30 to 45 | what the finetune was trained for; the mask draws 20-60 s |
| `--steps` | **8** | the post-trained weights are distilled for this |
| `--cfg` | **1.0** | guidance above 1 pushes the level hot on a distilled model |
| `--seed` | any int | see the warning below |
| `--objective` | `rf_denoiser` | picks ping-pong. Getting this wrong is the classic failure |

**If you are sampling the base transformer instead**, it is a different model:
`--objective rectified_flow --steps 50 --cfg 4.0`. Sampling base weights with
ping-pong produces quiet mush, which looks like a broken model and is not.

**Seeding.** Ping-pong redraws noise inside its own loop from the global
generator. Seeding only the initial latent leaves two models on different
trajectories, so any A/B measures the sampler instead of the weights. Both
scripts call `torch.manual_seed` as well; if you write your own, do the same.

---

## Reading the output

Both scripts print, per track, over the generated region only:

- **vs seed** — distance between the generated audio's 32-band log-mel envelope
  and the seed's. Lower means it continues in character. It rewards conservatism,
  so a model that invents more is penalised; use it alongside your ears.
- **level** — generated RMS over the seed's. Near 1.0 is right. A number like
  0.5 means the model partly gave up, which is the failure mode that motivated
  this whole project.

---

## What to expect

Trained on 26.4 hours of DJ sets from one Los Angeles venue, so it holds that
groove and imposes it on whatever you give it. Measured:

| material | stock | this model |
| --- | --- | --- |
| the 26.4 h source, 8 renders | 0.4537 | **0.2978** |
| six held-out songs by the same artist | **0.3431** | 0.4366 |
| a pop track, 190 s | **0.2120** | 0.3962 |

In domain it is a third closer. Out of domain it is worse by a similar margin,
because specialising is what it did. Where it is consistently better is
stability: on a 320 s generation the released model dropped to 0.69x level while
this one held 1.02x.
