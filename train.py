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
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import Excerpts, collate
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


def outpaint_mask(latents: torch.Tensor, p_full: float):
    """Keep a random-length prefix, generate the rest. 1 = given, 0 = to generate.

    `p_full` of the batch is fully masked instead, which keeps the model's
    unconditional generation from drifting while it learns to continue audio.
    """
    keep = torch.ones(latents.shape[0], latents.shape[-1], dtype=torch.bool, device=latents.device)
    return random_inpaint_mask(  # probabilities are [segments, full, causal]
        latents, padding_masks=keep, mask_type_probabilities=[0.0, p_full, 1.0 - p_full])


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
        tb.add_audio(tag, audio.mean(0), step, sample_rate=sr)
        sf.write(out / f"{tag}_{step:06d}.flac", audio.T.numpy(), sr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, default=Path("/home/alessio/Desktop/2n1t3-audio.flac"))
    p.add_argument("--model", type=Path, default=Path("models/stable-audio-3-medium"))
    p.add_argument("--out", type=Path, default=Path("runs/outpaint"))
    p.add_argument("--prompt", default="")
    p.add_argument("--seconds", type=float, default=24.0)  # 258 latent frames, just over the model's 256 floor
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--p-full", type=float, default=0.1, help="fraction of fully masked items")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--demo-every", type=int, default=250)
    p.add_argument("--demo-at", type=float, default=3600.0, help="offset of the reference excerpt, seconds")
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
    if a.resume:
        dit.load_state_dict({k: v.float() for k, v in load_file(a.resume).items()})
    # float32 master weights and grads already cost 11.6 GB, so the moments go 8-bit.
    opt = bnb.optim.AdamW8bit(dit.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)

    data = Excerpts([a.audio], a.seconds, sr, a.prompt, epoch=a.batch * 100)
    loader = DataLoader(data, batch_size=a.batch, collate_fn=collate, drop_last=True,
                        num_workers=a.workers, persistent_workers=a.workers > 0)

    # One fixed excerpt, encoded once, so the demos are comparable step to step.
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        ref = model.pretransform.encode(data.at(a.demo_at)[0][None].to(dev)).float()

    tb = SummaryWriter(a.out / "tb")
    tb.add_text("config", "\n".join(f"{k} = {v}" for k, v in vars(a).items()), 0)
    print(f"{data}; {sum(q.numel() for q in dit.parameters()) / 1e9:.2f}B trainable params", flush=True)

    step, t0 = 0, time.time()
    while step < a.steps:
        for audio, _ in loader:
            step += 1
            for g in opt.param_groups:
                g["lr"] = a.lr * min(1.0, step / max(a.warmup, 1))

            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                z = model.pretransform.encode(audio.to(dev, non_blocking=True)).float()
            masked, mask = outpaint_mask(z, a.p_full)
            cond["inpaint_mask"], cond["inpaint_masked_input"] = [mask], [masked]

            # Rectified flow: x_t = (1-t)*z + t*noise, and the model predicts noise - z.
            t = 1 - truncated_logistic_normal_rescaled(z.shape[0]).to(dev)
            t = model.dist_shift.shift(t, z.shape[-1]) if model.dist_shift else t
            noise = torch.randn_like(z)
            target = noise - z
            with torch.autocast("cuda", torch.bfloat16):
                pred = model(z * (1 - t[:, None, None]) + noise * t[:, None, None], t,
                             cond=cond, cfg_dropout_prob=0.1)

            # Only the region the model is actually generating contributes to the loss.
            gen = (1 - mask).expand_as(z)
            loss = (F.mse_loss(pred.float(), target, reduction="none") * gen).sum() / gen.sum().clamp(min=1)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            if step % 10 == 0:
                dt = (time.time() - t0) / 10
                tb.add_scalar("train/loss", loss.item(), step)
                tb.add_scalar("train/grad_norm", gnorm.item(), step)
                tb.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
                tb.add_scalar("train/sec_per_step", dt, step)
                tb.add_scalar("train/gpu_gb", torch.cuda.max_memory_allocated() / 2**30, step)
                print(f"step {step:6d}  loss {loss.item():.4f}  {dt:.2f}s/step", flush=True)
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
