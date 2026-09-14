#!/usr/bin/env python
"""Full finetune of Stable Audio 3 Medium for outpainting one track.

Each step draws a random excerpt, encodes it with the autoencoder online, keeps
a random-length prefix and asks the model to generate the rest. That is the
model's own causal inpainting mask, which is exactly outpainting, so nothing
about the architecture changes -- only the DiT weights move.

The prompt and the excerpt length are fixed, so the text encoder runs once at
startup and is then dropped; the autoencoder stays resident but frozen.
"""

from __future__ import annotations

import argparse
import json
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
from stable_audio_3.inference.sampling import (
    build_schedule,
    sample_flow_pingpong,
    truncated_logistic_normal_rescaled,
)
from stable_audio_3.loading_utils import load_diffusion_cond
from stable_audio_3.models.inpainting import random_inpaint_mask


def load(model_dir: Path, device):
    """Build the model with the text encoder pointed at the local weights."""
    cfg = json.loads((model_dir / "model_config.json").read_text())
    for c in cfg["model"]["conditioning"]["configs"]:
        if c["type"] == "t5gemma":
            c["config"] = {k: v for k, v in c["config"].items() if k not in ("repo_id", "subfolder")}
            c["config"]["model_path"] = str(model_dir / "t5gemma-b-b-ul2")
    return cfg, load_diffusion_cond(cfg, str(model_dir / "model.safetensors"), device=device)


def outpaint_mask(latents: torch.Tensor, p_full: float, p_spans: float, p_segments: float):
    """1 = given as context, 0 = to be generated.

    Mostly a random-length prefix, which is outpainting. The remainder augments
    the shape of the context: interior spans at a controlled mask ratio so the
    model never assumes context is a clean prefix, scattered segments, and fully
    masked items so unconditional generation does not drift. Prefix length,
    span count and mask ratio are all resampled every step.
    """
    keep = torch.ones(latents.shape[0], latents.shape[-1], dtype=torch.bool, device=latents.device)
    p_causal = 1.0 - p_full - p_spans - p_segments
    if p_causal <= 0:
        raise ValueError("mask probabilities leave nothing for the causal case")
    return random_inpaint_mask(  # [segments, full, causal, spans]
        latents, padding_masks=keep,
        mask_type_probabilities=[p_segments, p_full, p_causal, p_spans],
        mask_ratio_range=(0.2, 1.0))


def masked_mean(err: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    """Mean error inside a region, per sample, then averaged over the batch.

    Pooling the whole batch in one ratio instead would weight each item by how
    much of it is masked, which quietly down-weights the long-context examples
    that outpainting is actually about.
    """
    n = region.sum(dim=(1, 2)) * err.shape[1]
    return ((err * region).sum(dim=(1, 2)) / n.clamp(min=1)).mean()


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
def demo(model, cond, z, steps: int, tb, step: int, out: Path, sr: int):
    """Continue the first half of a fixed reference excerpt and log the result."""
    mask = torch.ones_like(z[:, :1])
    mask[..., z.shape[-1] // 2:] = 0
    c = {k: tuple(t[:1] for t in v) for k, v in cond.items()}
    c["inpaint_mask"], c["inpaint_masked_input"] = [mask], [z * mask]

    sigmas = build_schedule(steps, dist_shift=model.dist_shift,
                            effective_seq_len=z.shape[-1], device=z.device)
    with torch.autocast("cuda", torch.bfloat16):
        sampled = sample_flow_pingpong(model, torch.randn_like(z), sigmas, disable_tqdm=True, cond=c)

    for tag, latent in [("outpaint", sampled)] + ([("truth", z)] if step == 1 else []):
        with torch.autocast("cuda", torch.bfloat16):
            audio = model.pretransform.decode(latent.to(torch.bfloat16))
        audio = audio.float().clamp(-1, 1)[0].cpu()
        # mp3 on disk and half-rate mono in the event file: a 380 s demo is 44 MB
        # as flac and 67 MB as raw tensorboard audio, which a long run cannot afford.
        sf.write(out / f"{tag}_{step:06d}.mp3", audio.T.numpy(), sr,
                 format="MP3", subtype="MPEG_LAYER_III")
        tb.add_audio(tag, torchaudio.functional.resample(audio.mean(0), sr, sr // 2),
                     step, sample_rate=sr // 2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, default=Path("audio.flac"))
    p.add_argument("--model", type=Path, default=Path("models/stable-audio-3-medium"))
    p.add_argument("--out", type=Path, default=Path("runs/outpaint"))
    p.add_argument("--prompt", default='TrackType: Music, VocalType: Instrumental, Genre: Electronic. Electronic dance music recorded from a live DJ set, club sound system, driving drums and synthesizer bass.')
    p.add_argument("--seconds", type=float, default=95.0)  # 1022 latent frames, mid-range for the model
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-6, help="weight matrices")
    p.add_argument("--lr-1d", type=float, default=1e-6, help="norm gains and adaLN gates")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--p-full", type=float, default=0.10, help="fully masked, keeps unconditional behaviour")
    p.add_argument("--p-spans", type=float, default=0.15, help="interior spans masked, augments context shape")
    p.add_argument("--p-segments", type=float, default=0.05, help="scattered segments masked")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--latents", type=Path, help="precomputed stream from encode_latents.py")
    p.add_argument("--index", default="energy_index.npz", help="energy index from scan_energy.py")
    p.add_argument("--min-rms", type=float, default=0.15, help="skip excerpts quieter than this")
    p.add_argument("--min-sub", type=float, default=0.0, help="skip excerpts with less sub-bass than this")
    p.add_argument("--val-every", type=int, default=50)
    p.add_argument("--demo-every", type=int, default=250)
    p.add_argument("--demo-at", type=float, default=7200.0, help="offset of the reference excerpt, seconds")
    p.add_argument("--demo-steps", type=int, default=8)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--resume", type=Path)
    a = p.parse_args()

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    a.out.mkdir(parents=True, exist_ok=True)

    cfg, model = load(a.model, dev)
    sr = cfg["sample_rate"]
    model.pretransform.to(torch.bfloat16)  # frozen, and 0.85B params is 1.7 GB saved

    # One prompt and one length for the whole run: encode the text once, then let it go.
    with torch.no_grad():
        cond = model.conditioner([{"prompt": a.prompt, "seconds_total": a.seconds}] * a.batch, dev)
    del model.conditioner
    torch.cuda.empty_cache()

    dit = model.model.requires_grad_(True).train()
    start = 0
    if a.resume:  # weights only; float32 Adam moments are 11.6 GB and not worth the disk
        dit.load_state_dict({k: v.float() for k, v in load_file(a.resume).items()})
        with safe_open(a.resume, framework="pt") as f:
            start = int(f.metadata()["step"])
    # The reference trains weight matrices with Muon at 1e-5 and 1D tensors (norm
    # gains, adaLN gates) with AdamW at 1e-6. Muon's orthogonalised update is far
    # smaller per element than Adam's, so matching Adam's lr to 1e-5 would step the
    # matrices several times too far and the gates ten times too far. Moments go
    # 8-bit because float32 master weights and grads already cost 11.6 GB.
    mats = [q for q in dit.parameters() if q.ndim >= 2]
    vecs = [q for q in dit.parameters() if q.ndim < 2]
    opt = bnb.optim.AdamW8bit(
        [{"params": mats, "lr": a.lr, "base_lr": a.lr, "weight_decay": 0.01},
         {"params": vecs, "lr": a.lr_1d, "base_lr": a.lr_1d, "weight_decay": 0.0}],
        betas=(0.9, 0.95))

    ratio = int(model.pretransform.downsampling_ratio)
    if a.latents:
        data = LatentExcerpts(a.latents, a.seconds, sr, ratio, a.prompt, epoch=a.batch * 100,
                              index=a.index, min_rms=a.min_rms, min_sub=a.min_sub)
        encode = lambda batch: batch.to(dev, non_blocking=True).float()
    else:
        data = Excerpts([a.audio], a.seconds, sr, a.prompt, epoch=a.batch * 100,
                        index=a.index, min_rms=a.min_rms, min_sub=a.min_sub)

        def encode(batch):
            with torch.autocast("cuda", torch.bfloat16):
                return model.pretransform.encode(batch.to(dev, non_blocking=True)).float()
    loader = DataLoader(data, batch_size=a.batch, collate_fn=collate, drop_last=True,
                        num_workers=a.workers, persistent_workers=a.workers > 0)

    # One fixed excerpt, so the demos are comparable step to step.
    with torch.no_grad():
        ref = encode(data.at(a.demo_at)[0][None])
    val_items = frozen_val(encode, data, dev)

    tb = SummaryWriter(a.out / "tb")
    tb.add_text("config", "\n".join(f"{k} = {v}" for k, v in vars(a).items()), 0)
    print(f"{data}; {sum(q.numel() for q in dit.parameters()) / 1e9:.2f}B trainable params", flush=True)

    step, t0 = start, time.time()
    while step < a.steps:
        for audio, _ in loader:
            step += 1
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * min(1.0, step / max(a.warmup, 1))

            with torch.no_grad():
                z = encode(audio)
            masked, mask = outpaint_mask(z, a.p_full, a.p_spans, a.p_segments)
            cond["inpaint_mask"], cond["inpaint_masked_input"] = [mask], [masked]

            # Rectified flow: x_t = (1-t)*z + t*noise, and the model predicts noise - z.
            t = 1 - truncated_logistic_normal_rescaled(z.shape[0]).to(dev)
            t = model.dist_shift.shift(t, z.shape[-1]) if model.dist_shift else t
            noise = torch.randn_like(z)
            target = noise - z
            with torch.autocast("cuda", torch.bfloat16):
                pred = model(z * (1 - t[:, None, None]) + noise * t[:, None, None], t,
                             cond=cond, cfg_dropout_prob=0.1)

            # Two terms, as the reference does: the region being generated, and the
            # context. Without the second, nothing constrains the model's output on the
            # prefix -- sampling regenerates it too -- so it drifts as the weights move.
            err = F.mse_loss(pred.float(), target, reduction="none")
            gen_loss, ctx_loss = masked_mean(err, 1 - mask), masked_mean(err, mask)
            loss = gen_loss + ctx_loss
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            if step % 10 == 0:
                dt = (time.time() - t0) / 10
                tb.add_scalar("train/loss", loss.item(), step)
                tb.add_scalar("train/loss_gen", gen_loss.item(), step)
                tb.add_scalar("train/loss_context", ctx_loss.item(), step)
                tb.add_scalar("train/grad_norm", gnorm.item(), step)
                tb.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
                tb.add_scalar("train/sec_per_step", dt, step)
                tb.add_scalar("train/gpu_gb", torch.cuda.max_memory_allocated() / 2**30, step)
                print(f"step {step:6d}  loss {loss.item():.4f}  {dt:.2f}s/step", flush=True)
                t0 = time.time()

            if step % a.val_every == 0 or step == 1:
                dit.eval()
                v_gen, v_ctx = validate(model, cond, val_items)
                tb.add_scalar("val/loss", v_gen, step)
                tb.add_scalar("val/loss_context", v_ctx, step)
                dit.train()
                t0 = time.time()

            if step % a.demo_every == 0 or step == 1:
                dit.eval()
                torch.cuda.empty_cache()
                try:
                    demo(model, cond, ref, a.demo_steps, tb, step, a.out, sr)
                except torch.OutOfMemoryError:
                    print("demo OOM, skipped", flush=True)
                dit.train()
                torch.cuda.empty_cache()
                t0 = time.time()

            if step % a.save_every == 0 or step == a.steps:
                save_file({k: v.to(torch.bfloat16) for k, v in dit.state_dict().items()},
                          a.out / "dit.safetensors", metadata={"step": str(step)})
                print(f"saved at step {step}", flush=True)

            if step >= a.steps:
                break
    tb.close()


if __name__ == "__main__":
    main()
