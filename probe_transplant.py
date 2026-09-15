#!/usr/bin/env python
"""Does a base-trained delta survive being added to the post-trained weights?

Stability's guidance is to train the base checkpoint and apply the result to the
post-trained one. That guidance is written for LoRA adapters, which are small by
construction. A full finetune is not, so the question has to be measured rather
than assumed.

The property at risk is the one adversarial post-training installed: that the
model's one-step estimate is a *sample*, full scale even where the model knows
little, rather than a conditional mean whose scale collapses to its correlation
with the truth. This runs that measurement on three models -- the post-trained
weights alone, the base weights alone, and the post-trained weights plus the
finetune's delta -- so the transplant can be read against both endpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from remote_weights import stream_state_dict
from train import build_number_conditioner, load

ARC = "https://huggingface.co/stabilityai/stable-audio-3-medium/resolve/main/model.safetensors"


@torch.no_grad()
def signature(model, z, cond, mask, ts):
    """Amplitude of the one-step estimate, and its correlation with the truth."""
    c = dict(cond)
    c["inpaint_mask"], c["inpaint_masked_input"] = [mask], [z * mask]
    gen = (1 - mask).bool().expand_as(z)
    zs = z[gen].std().item()
    rows = []
    for t in ts:
        g = torch.Generator(device=z.device).manual_seed(0)
        noise = torch.randn(z.shape, generator=g, device=z.device)
        x = z * (1 - t) + noise * t
        with torch.autocast("cuda", torch.bfloat16):
            v = model(x, torch.full((1,), t, device=z.device), cond=c).float()
        den = (x - t * v)[gen]
        dz, zz = den - den.mean(), z[gen] - z[gen].mean()
        corr = (dz * zz).mean().item() / (dz.std().item() * zz.std().item() + 1e-9)
        rows.append((t, den.std().item() / zs, corr))
    return rows


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("models/stable-audio-3-medium"))
    p.add_argument("--dit", type=Path,
                   default=Path("models/stable-audio-3-medium-base/dit_base.safetensors"))
    p.add_argument("--resume", type=Path, default=Path("runs/base/dit.safetensors"))
    p.add_argument("--latents", type=Path, default=Path("latents.npy"))
    p.add_argument("--token", default=str(Path.home() / ".hf_token"))
    p.add_argument("--at", type=float, default=7289.25)
    p.add_argument("--seconds", type=float, default=95.0)
    p.add_argument("--context", type=float, default=30.0)
    p.add_argument("--fps", type=float, default=44100 / 4096)
    a = p.parse_args()

    dev = torch.device("cuda")
    ts = (0.95, 0.9, 0.8, 0.6, 0.4, 0.2)

    cfg, model = load(a.model, dev, conditioner=False, dit=a.dit)
    model.model.to(torch.bfloat16)
    model.pretransform.to(torch.bfloat16)
    model.eval()

    number = build_number_conditioner(a.model, dev)
    prompt = tuple(t.to(dev) for t in torch.load("runs/base/prompt_cond.pt"))
    cond = {"prompt": tuple(t[:1] for t in prompt),
            "seconds_total": number([{"seconds_total": a.seconds}], dev)["seconds_total"]}

    frames = round(a.seconds * a.fps)
    arr = np.load(a.latents, mmap_mode="r")[:, round(a.at * a.fps):][:, :frames]
    z = torch.from_numpy(np.asarray(arr)).float()[None].to(dev)
    mask = torch.zeros_like(z[:, :1])
    mask[..., :round(a.context * a.fps)] = 1.0

    results = {}
    base_sd = {k: v.clone() for k, v in model.model.state_dict().items()}
    results["base"] = signature(model, z, cond, mask, ts)

    print("streaming the post-trained transformer from the Hub", flush=True)
    token = Path(a.token).read_text().strip()
    arc = stream_state_dict(ARC, token, prefix="model.model.", device=dev,
                            progress=lambda f: print(f"  {f:5.1%}", flush=True))
    # The file namespaces tensors under `model.model.`; the wrapper's own state
    # dict keeps one `model.` level, and the delta already matches the wrapper.
    arc = {k[len("model."):]: v for k, v in arc.items()}
    model.model.load_state_dict(arc)
    results["post-trained"] = signature(model, z, cond, mask, ts)

    delta = load_file(a.resume)
    missing = set(arc) ^ set(delta)
    if missing:
        raise SystemExit(f"{len(missing)} keys do not line up, first: {sorted(missing)[:3]}")
    model.model.load_state_dict({k: v + delta[k].to(v.dtype).to(v.device)
                                 for k, v in arc.items()})
    results["post-trained + delta"] = signature(model, z, cond, mask, ts)

    model.model.load_state_dict({k: v + delta[k].to(v.dtype).to(v.device)
                                 for k, v in base_sd.items()})
    results["base + delta (what we train)"] = signature(model, z, cond, mask, ts)

    print(f"\n{'':30s}" + "".join(f"  t={t:<12}" for t in ts))
    for name, rows in results.items():
        amp = "".join(f"  {r:5.3f}/{c:<7.3f}" for _, r, c in rows)
        print(f"{name:30s}{amp}")
    print("\nEach cell is amplitude / correlation. They coincide for a conditional-mean")
    print("predictor and separate for a sample predictor. The transplant is safe if")
    print("`post-trained + delta` keeps amplitude well above correlation at t=0.95.")


if __name__ == "__main__":
    main()
