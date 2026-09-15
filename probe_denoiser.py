#!/usr/bin/env python
"""Measure whether the model predicts a sample or the conditional mean.

Ping-pong sampling, which is what the released Stable Audio 3 Medium checkpoint
is distilled for, rebuilds x each step as

    denoised = x - t * model(x, t)
    x        = (1 - t_next) * denoised + t_next * fresh_noise

so every step's `denoised` has to be a plausible, full-scale piece of audio on
its own. A model trained with plain mean-squared error predicts the *conditional
mean* E[x0 | x_t], which at high noise collapses toward zero. Feeding that to a
ping-pong sampler gives quiet, over-smoothed output with residual noise.

This script reports std(denoised) / std(z) across the noise level. Near 1.0 means
a sample predictor. Falling toward 0 at high t means a mean predictor.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from train import build_number_conditioner, load


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("models/stable-audio-3-medium"))
    p.add_argument("--dit", type=Path, help="transformer-only checkpoint to overlay, e.g. the base weights")
    p.add_argument("--objective", choices=["rectified_flow", "rf_denoiser"],
                   help="override the config; the base transformer is rectified_flow")
    p.add_argument("--latents", type=Path, default=Path("latents.npy"))
    p.add_argument("--resume", type=Path, help="delta checkpoint to apply")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--at", type=float, default=7289.25)
    p.add_argument("--seconds", type=float, default=47.0)
    p.add_argument("--context", type=float, default=30.0)
    p.add_argument("--fps", type=float, default=44100 / 4096)
    p.add_argument("--tag", default="pretrained")
    a = p.parse_args()

    dev = torch.device("cuda")
    cfg, model = load(a.model, dev, conditioner=False, dit=a.dit, objective=a.objective)
    model.model.to(torch.bfloat16)          # inference only; halves the resident DiT
    model.pretransform.to(torch.bfloat16)
    if a.resume:
        d = load_file(a.resume)
        model.model.load_state_dict(
            {k: v + a.alpha * d[k].to(v.dtype).to(v.device)
             for k, v in model.model.state_dict().items()})
    model.eval()

    number = build_number_conditioner(a.model, dev)
    prompt = tuple(t.to(dev) for t in torch.load("runs/base/prompt_cond.pt"))

    frames = round(a.seconds * a.fps)
    start = round(a.at * a.fps)
    arr = np.load(a.latents, mmap_mode="r")[:, start:start + frames]
    z = torch.from_numpy(np.asarray(arr)).float()[None].to(dev)

    cond = {"prompt": tuple(t[:1] for t in prompt),
            "seconds_total": number([{"seconds_total": a.seconds}], dev)["seconds_total"]}

    keep = round(a.context * a.fps)
    for label in ("full", "causal"):
        mask = torch.zeros_like(z[:, :1])
        if label == "causal":
            mask[..., :keep] = 1.0
        c = dict(cond)
        c["inpaint_mask"], c["inpaint_masked_input"] = [mask], [z * mask]
        gen = (1 - mask).bool().expand_as(z)
        zs = z[gen].std().item()

        rows = []
        for t in (0.95, 0.9, 0.8, 0.6, 0.4, 0.2):
            g = torch.Generator(device=dev).manual_seed(0)
            noise = torch.randn(z.shape, generator=g, device=dev)
            x = z * (1 - t) + noise * t
            tt = torch.full((1,), t, device=dev)
            with torch.autocast("cuda", torch.bfloat16):
                v = model(x, tt, cond=c).float()
            den = x - t * v
            ds = den[gen].std().item()
            # correlation of the estimate with the truth, inside the generated region
            dz, zz = den[gen] - den[gen].mean(), z[gen] - z[gen].mean()
            corr = (dz * zz).mean().item() / (dz.std().item() * zz.std().item() + 1e-9)
            rows.append((t, ds / zs, corr))

        print(f"\n{a.tag}  mask={label}   std(z_gen)={zs:.3f}")
        print("   t      std(denoised)/std(z)    corr(denoised, z)")
        for t, r, corr in rows:
            print(f"  {t:.2f}        {r:6.3f}              {corr:6.3f}")


if __name__ == "__main__":
    main()
