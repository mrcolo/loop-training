#!/usr/bin/env python
"""Full finetune of Stable Audio 3 Medium for outpainting one recording.

The recipe follows the one Stability ships in the model's own `model_config.json`
rather than a generic diffusion finetune:

  * velocity target on a rectified-flow path, timesteps from a truncated
    logit-normal that is flipped and then warped by the model's length-dependent
    distribution shift;
  * inpainting masks as local-additive conditioning, with the loss split into two
    independently averaged terms, one over the region being generated and one over
    the context that must be preserved;
  * Muon on the attention and feed-forward matrices, AdamW on everything else;
  * an exponential moving average of the weights, which is what gets shipped.

It expects the **base** checkpoint. The released `stable-audio-3-medium` weights
have been adversarially post-trained to predict a sample in a few steps; a
mean-squared error objective pulls them back toward the conditional mean, which
is exactly the thing that post-training existed to remove. See the README.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import bitsandbytes as bnb
import soundfile as sf
import torch
import torchaudio
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import Excerpts, LatentExcerpts, collate
from muon import Muon, fused_chunks
from stable_audio_3.inference.sampling import (
    build_schedule,
    sample_discrete_euler,
    sample_flow_pingpong,
    truncated_logistic_normal_rescaled,
)
from stable_audio_3.factory import create_multi_conditioner_from_conditioning_config
from stable_audio_3.loading_utils import copy_state_dict, load_diffusion_cond
from stable_audio_3.models.inpainting import MaskType, random_inpaint_mask

MUON = ("self_attn.to_qkv", "self_attn.to_out", "cross_attn.to_q",
        "cross_attn.to_kv", "cross_attn.to_out", ".ff.ff.")


def load(model_dir: Path, device, conditioner: bool = True, dit: Path | None = None,
         objective: str | None = None):
    """Build the model with the text encoder pointed at the local weights.

    `dit` overlays a second checkpoint holding only the transformer, which is how
    the base transformer is paired with the autoencoder from the released
    checkpoint without keeping two 4 GB files on disk. `objective` overrides the
    config's, since the two checkpoints share an autoencoder but not a target.
    """
    cfg = json.loads((model_dir / "model_config.json").read_text())
    for c in cfg["model"]["conditioning"]["configs"]:
        if c["type"] == "t5gemma":
            c["config"] = {k: v for k, v in c["config"].items() if k not in ("repo_id", "subfolder")}
            c["config"]["model_path"] = str(model_dir / "t5gemma-b-b-ul2")
    if not conditioner:
        cfg = json.loads(json.dumps(cfg))
        cfg["model"]["conditioning"]["configs"] = []
    model = load_diffusion_cond(cfg, str(model_dir / "model.safetensors"), device=device)
    if dit:
        weights = load_file(dit)
        copy_state_dict(model, {k: v.to(device) for k, v in weights.items()})
        # The released checkpoint on disk holds only the autoencoder, so a transformer
        # key that fails to land here leaves that tensor at its random initialisation
        # and training proceeds looking entirely normal. Refuse instead.
        strip = lambda k: k[6:] if k.startswith("model.") else k
        have = {strip(strip(k)) for k in weights}
        missing = [k for k in model.model.state_dict() if strip(strip(k)) not in have]
        if missing:
            raise SystemExit(f"{dit} is missing {len(missing)} transformer tensors, "
                             f"first: {missing[:3]}")
    if objective:
        model.diffusion_objective = objective
    return cfg, model


def build_number_conditioner(model_dir: Path, device):
    """Just the duration conditioner, rebuilt from the checkpoint.

    It is a few thousand parameters, so it stays resident and runs every step
    while the 1.2 GB text encoder does not.
    """
    cfg = json.loads((model_dir / "model_config.json").read_text())
    cc = {"configs": [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "number"],
          "cond_dim": cfg["model"]["conditioning"]["cond_dim"]}
    number = create_multi_conditioner_from_conditioning_config(cc)
    weights = {}
    with safe_open(str(model_dir / "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith("conditioner."):
                weights[k[len("conditioner."):]] = f.get_tensor(k)
    number.load_state_dict(weights, strict=False)
    return number.to(device).eval().requires_grad_(False)


def outpaint_mask(latents: torch.Tensor, fps: float, p_full: float, p_segments: float,
                  ctx_seconds=(20.0, 60.0), min_gen: float = 15.0):
    """1 = given as context, 0 = to be generated.

    The reference draws full / segments / causal at 0.8 / 0.1 / 0.1, and its causal
    prefix is a uniform fraction of the window. Two changes, both aimed at this task:
    causal gets a larger share because outpainting is the job, and its context length
    is drawn in *seconds* rather than as a fraction. A 30 s seed is 16% of a 190 s
    window and 8% of a 380 s one, so a uniform fraction spends almost nothing on the
    lengths people actually ask for. The large remaining share of fully masked items
    is what stops unconditional generation decaying, and it is kept high for that reason.

    Drawing context in seconds needs a floor on what is left to generate. A 20-60 s
    context against a 47 s window otherwise clamps to the whole window nearly half the
    time, and an excerpt with two latent frames to generate teaches the model to copy.
    `min_gen` reserves a minimum generated span and shortens the context to fit.
    """
    B, _, T = latents.shape
    keep = torch.ones(1, T, dtype=torch.bool, device=latents.device)
    masks = []
    for _ in range(B):
        r = random.random()
        if r < p_full:
            m = torch.zeros(1, 1, T, device=latents.device)
        elif r < p_full + p_segments:
            _, m = random_inpaint_mask(latents[:1], padding_masks=keep,
                                       force_mask_type=MaskType.RANDOM_SEGMENTS)
        else:
            hi = max(1.0 / fps, min(ctx_seconds[1], T / fps - min_gen))
            lo = min(ctx_seconds[0], hi)
            k = min(T - 1, max(1, round(random.uniform(lo, hi) * fps)))
            m = torch.zeros(1, 1, T, device=latents.device)
            m[..., :k] = 1.0
        masks.append(m)
    mask = torch.cat(masks, 0)
    return latents * mask, mask


def masked_mean(err: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    """Mean error inside a region, per sample, then averaged over the batch."""
    n = region.sum(dim=(1, 2)) * err.shape[1]
    return ((err * region).sum(dim=(1, 2)) / n.clamp(min=1)).mean()


def split_params(dit):
    """Muon for the transformer matrices, AdamW for everything else.

    The paper is specific: "Muon ... is applied to attention QKV projections and
    feed-forward network projections, while AdamW ... handles all remaining
    parameters." Embeddings, norm gains, biases and the input and output
    projections are not matrices Muon is meant for.
    """
    muon, adam, chunks = [], [], {}
    for name, q in dit.named_parameters():
        if q.ndim == 2 and any(k in name for k in MUON):
            muon.append(q)
            chunks[id(q)] = fused_chunks(name, q.shape, 1536)
        else:
            adam.append(q)
    return muon, adam, chunks


class EMA:
    """Moving average of the weights, kept on the host.

    The reference averages with beta 0.9995 under a power-law warmup and uses the
    average for inference. Holding it in pinned host memory rather than on the
    card costs a copy every few steps and buys back 2.9 GB of VRAM.
    """

    def __init__(self, dit, beta: float = 0.9995, every: int = 8):
        self.shadow = {k: v.detach().to("cpu", torch.float32).clone()
                       for k, v in dit.state_dict().items()}
        self.beta, self.every = beta, every

    def update(self, dit, step: int):
        if step % self.every:
            return
        d = min(self.beta, 1 - (1 + step) ** -0.75) ** self.every
        with torch.no_grad():
            for k, v in dit.state_dict().items():
                self.shadow[k].mul_(d).add_(v.detach().to("cpu", torch.float32), alpha=1 - d)


@torch.no_grad()
def frozen_val(encode, data, dev, n_excerpts: int = 4, n_t: int = 4):
    """Freeze a validation slice: the same excerpts, masks, timesteps and noise
    every time. Training loss is a single batch at a random timestep and swings
    far too much to read; this is the number that actually shows learning."""
    torch.manual_seed(0)
    items = []
    for i in range(n_excerpts):
        z = encode(data.at(600 + i * 7200)[0][None])
        mask = torch.ones_like(z[:, :1])
        mask[..., int(z.shape[-1] * (0.25 + 0.5 * i / max(n_excerpts - 1, 1))):] = 0
        for j in range(n_t):
            items.append((z, mask, torch.full((1,), (j + 0.5) / n_t, device=dev), torch.randn_like(z)))
    return items


@torch.no_grad()
def validate(model, cond, items):
    """Mean error over the frozen slice, reported separately for each region."""
    c = {k: tuple(x[:1] for x in v) for k, v in cond.items()}
    gen_total = ctx_total = 0.0
    for z, mask, t, noise in items:
        c["inpaint_mask"], c["inpaint_masked_input"] = [mask], [z * mask]
        with torch.autocast("cuda", torch.bfloat16):
            pred = model(z * (1 - t[:, None, None]) + noise * t[:, None, None], t, cond=c)
        err = (pred.float() - (noise - z)).square()
        gen_total += masked_mean(err, 1 - mask).item()
        ctx_total += masked_mean(err, mask).item()
    return gen_total / len(items), ctx_total / len(items)


@torch.no_grad()
def demo(model, cond, z, steps: int, tb, step: int, out: Path, sr: int,
         ctx_seconds: float = 30.0, fps: float = 44100 / 4096, cfg_scale: float = 1.0):
    """Continue a fixed reference excerpt and log the result."""
    keep = min(z.shape[-1] - 1, max(1, round(ctx_seconds * fps)))
    mask = torch.zeros_like(z[:, :1])
    mask[..., :keep] = 1.0
    c = {k: tuple(t[:1] for t in v) for k, v in cond.items()}
    c["inpaint_mask"], c["inpaint_masked_input"] = [mask], [z * mask]

    sigmas = build_schedule(steps, dist_shift=model.sampling_dist_shift,
                            effective_seq_len=z.shape[-1], device=z.device)
    # ping-pong is for the adversarially post-trained checkpoint, which predicts a
    # sample; the base checkpoint predicts a velocity and wants an ODE solver.
    sampler = (sample_flow_pingpong if model.diffusion_objective == "rf_denoiser"
               else sample_discrete_euler)
    # Fixed noise, so renders are comparable across steps.
    gen = torch.Generator(device=z.device).manual_seed(0)
    noise = torch.randn(z.shape, generator=gen, device=z.device, dtype=z.dtype)
    with torch.autocast("cuda", torch.bfloat16):
        sampled = sampler(model, noise, sigmas, disable_tqdm=True, cond=c, cfg_scale=cfg_scale)

    for tag, latent in [("outpaint", sampled)] + ([("truth", z)] if step <= 1 else []):
        with torch.autocast("cuda", torch.bfloat16):
            audio = model.pretransform.decode(latent.to(torch.bfloat16))
        audio = audio.float().clamp(-1, 1)[0].cpu()
        sf.write(out / f"{tag}_{step:06d}.mp3", audio.T.numpy(), sr,
                 format="MP3", subtype="MPEG_LAYER_III")
        tb.add_audio(tag, torchaudio.functional.resample(audio.mean(0), sr, sr // 2),
                     step, sample_rate=sr // 2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, default=Path("audio.flac"))
    p.add_argument("--model", type=Path, default=Path("models/stable-audio-3-medium"))
    p.add_argument("--dit", type=Path, help="transformer-only checkpoint to overlay, e.g. the base weights")
    p.add_argument("--objective", choices=["rectified_flow", "rf_denoiser"],
                   help="override the config; the base transformer is rectified_flow")
    p.add_argument("--out", type=Path, default=Path("runs/outpaint"))
    p.add_argument("--prompt", default='TrackType: Music, VocalType: Instrumental, Genre: Electronic. Electronic dance music recorded from a live DJ set, club sound system, driving drums and synthesizer bass.')
    p.add_argument("--seconds", type=float, nargs="+", default=[47.0, 95.0, 190.0, 380.0],
                   help="excerpt lengths sampled per step; the model conditions on duration, "
                        "so a single length freezes that pathway")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accum", type=int, default=4, help="gradient accumulation microbatches")
    p.add_argument("--muon-lr", type=float, default=2e-4,
                   help="attention and feed-forward matrices; the config pretrains at 1e-3")
    p.add_argument("--adam-lr", type=float, default=1e-5,
                   help="everything else; the config pretrains at 5e-5")
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--ema", type=float, default=0.9995)
    p.add_argument("--ema-every", type=int, default=8)
    p.add_argument("--p-full", type=float, default=0.55, help="fully masked; the config uses 0.8")
    p.add_argument("--p-segments", type=float, default=0.10, help="scattered segments masked")
    p.add_argument("--ctx-min", type=float, default=20.0, help="shortest causal context, seconds")
    p.add_argument("--ctx-max", type=float, default=60.0, help="longest causal context, seconds")
    p.add_argument("--min-gen", type=float, default=15.0,
                   help="seconds the causal case always leaves to generate; context is shortened to fit")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--latents", type=Path, help="precomputed stream from encode_latents.py")
    p.add_argument("--index", default="energy_index.npz", help="energy index from scan_energy.py")
    p.add_argument("--min-rms", type=float, default=0.15, help="skip excerpts quieter than this")
    p.add_argument("--min-sub", type=float, default=0.0, help="skip excerpts with less sub-bass than this")
    p.add_argument("--val-every", type=int, default=50)
    p.add_argument("--demo-every", type=int, default=500)
    p.add_argument("--demo-at", type=float, default=7289.25, help="offset of the reference excerpt, seconds")
    p.add_argument("--demo-context", type=float, default=30.0)
    p.add_argument("--demo-seconds", type=float, default=190.0)
    p.add_argument("--demo-steps", type=int, help="default 50 for the base objective, 8 for the distilled one")
    p.add_argument("--demo-cfg", type=float, help="default 4.0 for the base objective, 1.0 for the distilled one")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--resume", type=Path)
    a = p.parse_args()

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    a.out.mkdir(parents=True, exist_ok=True)

    cache = a.out / "prompt_cond.pt"
    cfg, model = load(a.model, dev, conditioner=not cache.exists(), dit=a.dit, objective=a.objective)
    sr = cfg["sample_rate"]
    distilled = model.diffusion_objective == "rf_denoiser"
    a.demo_steps = a.demo_steps or (8 if distilled else 50)
    a.demo_cfg = a.demo_cfg if a.demo_cfg is not None else (1.0 if distilled else 4.0)
    model.pretransform.to(torch.bfloat16)

    # The prompt is fixed for a run so its embedding is cached, but the duration
    # is not: the model conditions on it and training at one value freezes that
    # pathway. The number conditioner is tiny, so it stays and runs every step.
    number = build_number_conditioner(a.model, dev)
    if cache.exists():
        prompt_cond = tuple(t.to(dev) for t in torch.load(cache))
    else:
        with torch.no_grad():
            c = model.conditioner([{"prompt": a.prompt, "seconds_total": a.seconds[0]}], dev)
        prompt_cond = c["prompt"]
        torch.save(tuple(t.cpu() for t in prompt_cond), cache)
        del model.conditioner
    torch.cuda.empty_cache()

    def conditioning(seconds: float, batch: int):
        with torch.no_grad():
            secs = number([{"seconds_total": seconds}] * batch, dev)["seconds_total"]
        return {"prompt": tuple(t.expand(batch, *t.shape[1:]) for t in prompt_cond),
                "seconds_total": secs}

    cond = conditioning(max(a.seconds), a.batch)

    dit = model.model.requires_grad_(True).train()
    # Checkpoints store the delta from these weights, not the weights themselves:
    # the update is smaller than a bfloat16 rounding step at this weight magnitude,
    # so storing the weights would quantise the finetune away.
    base = {k: v.detach().to("cpu", torch.float32).clone() for k, v in dit.state_dict().items()}
    start = 0
    if a.resume:
        delta = load_file(a.resume)
        dit.load_state_dict({k: v + delta[k].to(v.dtype).to(v.device) for k, v in dit.state_dict().items()})
        with safe_open(a.resume, framework="pt") as f:
            start = int(f.metadata()["step"])

    muon_params, adam_params, chunks = split_params(dit)
    opts = [Muon(muon_params, lr=a.muon_lr, momentum=a.momentum, weight_decay=0.0, chunks=chunks),
            bnb.optim.AdamW8bit(adam_params, lr=a.adam_lr, betas=(0.9, 0.95), weight_decay=0.01)]
    for o in opts:
        for g in o.param_groups:
            g["base_lr"] = g["lr"]
    ema = EMA(dit, a.ema, a.ema_every)

    ratio = int(model.pretransform.downsampling_ratio)
    if a.latents:
        data = LatentExcerpts(a.latents, a.seconds, sr, ratio, a.prompt,
                              epoch=a.batch * a.accum * 100,
                              index=a.index, min_rms=a.min_rms, min_sub=a.min_sub)
        encode = lambda batch: batch.to(dev, non_blocking=True).float()
    else:
        data = Excerpts([a.audio], max(a.seconds), sr, a.prompt,
                        epoch=a.batch * a.accum * 100,
                        index=a.index, min_rms=a.min_rms, min_sub=a.min_sub)

        def encode(batch):
            with torch.autocast("cuda", torch.bfloat16):
                return model.pretransform.encode(batch.to(dev, non_blocking=True)).float()
    loader = DataLoader(data, batch_size=a.batch, collate_fn=collate, drop_last=True,
                        num_workers=a.workers, persistent_workers=a.workers > 0)

    with torch.no_grad():
        ref = encode(data.at(a.demo_at, seconds=a.demo_seconds)[0][None])
    val_items = frozen_val(encode, data, dev)

    tb = SummaryWriter(a.out / "tb")
    tb.add_text("config", "\n".join(f"{k} = {v}" for k, v in vars(a).items()), 0)
    print(f"{data}; objective {model.diffusion_objective}; "
          f"{sum(q.numel() for q in muon_params) / 1e9:.2f}B on Muon, "
          f"{sum(q.numel() for q in adam_params) / 1e6:.0f}M on AdamW", flush=True)

    step, micro, t0 = start, 0, time.time()
    agg = {}
    while step < a.steps:
        for audio, meta in loader:
            cond = conditioning(meta[0]["seconds_total"], audio.shape[0])
            with torch.no_grad():
                z = encode(audio)
            masked, mask = outpaint_mask(z, data.fps, a.p_full, a.p_segments,
                                         (a.ctx_min, a.ctx_max), a.min_gen)
            cond["inpaint_mask"], cond["inpaint_masked_input"] = [mask], [masked]

            # Rectified flow: x_t = (1-t)*z + t*noise, and the model predicts noise - z.
            t = 1 - truncated_logistic_normal_rescaled(z.shape[0]).to(dev)
            t = model.dist_shift.shift(t, z.shape[-1]) if model.dist_shift else t
            noise = torch.randn_like(z)
            target = noise - z
            with torch.autocast("cuda", torch.bfloat16):
                pred = model(z * (1 - t[:, None, None]) + noise * t[:, None, None], t,
                             cond=cond, cfg_dropout_prob=0.1)

            # Two independently averaged terms, as the paper specifies: a generation
            # loss over the masked region and a context preservation loss over the
            # region handed to the model. Summed, not pooled -- pooling would weight
            # each region by its size.
            err = F.mse_loss(pred.float(), target, reduction="none")
            gen_loss, ctx_loss = masked_mean(err, 1 - mask), masked_mean(err, mask)
            loss = gen_loss + ctx_loss
            (loss / a.accum).backward()
            for k, v in (("loss", loss), ("loss_gen", gen_loss), ("loss_context", ctx_loss)):
                agg[k] = agg.get(k, 0.0) + v.item() / a.accum

            micro += 1
            if micro % a.accum:
                continue
            step += 1
            for o in opts:
                for g in o.param_groups:
                    # Warmup, then the config's inverse power-law decay, which over a
                    # run of this length is very nearly flat.
                    g["lr"] = (g["base_lr"] * min(1.0, step / max(a.warmup, 1))
                               / (1 + step / 1e6) ** 0.5)
            gnorm = torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
            for o in opts:
                o.step()
                o.zero_grad(set_to_none=True)
            ema.update(dit, step)

            if step % 10 == 0:
                dt = (time.time() - t0) / 10
                for k, v in agg.items():
                    tb.add_scalar(f"train/{k}", v / 10, step)
                tb.add_scalar("train/grad_norm", gnorm.item(), step)
                tb.add_scalar("train/muon_lr", opts[0].param_groups[0]["lr"], step)
                tb.add_scalar("train/sec_per_step", dt, step)
                tb.add_scalar("train/gpu_gb", torch.cuda.max_memory_allocated() / 2**30, step)
                print(f"step {step:6d}  loss {agg['loss'] / 10:.4f}  {dt:.2f}s/step", flush=True)
                agg, t0 = {}, time.time()

            if step % a.val_every == 0 or step == 1:
                dit.eval()
                v_gen, v_ctx = validate(model, cond, val_items)
                tb.add_scalar("val/loss", v_gen, step)
                tb.add_scalar("val/loss_context", v_ctx, step)
                dit.train()
                t0 = time.time()

            if step % a.demo_every == 0 or step == 1:
                dit.eval()
                live = {k: v.detach().clone() for k, v in dit.state_dict().items()}
                dit.load_state_dict({k: v.to(live[k].device, live[k].dtype)
                                     for k, v in ema.shadow.items()})
                torch.cuda.empty_cache()
                try:
                    demo(model, cond, ref, a.demo_steps, tb, step, a.out, sr,
                         a.demo_context, data.fps, a.demo_cfg)
                except torch.OutOfMemoryError:
                    print("demo OOM, skipped", flush=True)
                dit.load_state_dict(live)
                del live
                dit.train()
                torch.cuda.empty_cache()
                t0 = time.time()

            if step % a.save_every == 0 or step == a.steps:
                # The average is what gets shipped, so that is what is saved.
                save_file({k: (v - base[k]).to(torch.float16) for k, v in ema.shadow.items()},
                          a.out / "dit.safetensors", metadata={"step": str(step)})
                print(f"saved at step {step}", flush=True)

            if step >= a.steps:
                break
    tb.close()


if __name__ == "__main__":
    main()
