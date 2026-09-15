#!/usr/bin/env python
"""Outpaint the tail of audio excerpts with a trained (or pretrained) checkpoint.

Writes one file per excerpt per checkpoint, plus the real continuation, so the
three can be compared directly. Pass --resume to hear a finetune; omit it to
hear the released model on the same context.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
from safetensors.torch import load_file

from dataset import Excerpts
from stable_audio_3.inference.sampling import (
    build_schedule,
    sample_discrete_euler,
    sample_flow_pingpong,
)
from train import build_number_conditioner, load


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("models/stable-audio-3-medium"))
    p.add_argument("--dit", type=Path, help="transformer-only checkpoint to overlay, e.g. the base weights")
    p.add_argument("--objective", choices=["rectified_flow", "rf_denoiser"],
                   help="override the config; the base transformer is rectified_flow")
    p.add_argument("--out", type=Path, default=Path("samples"))
    p.add_argument("--resume", type=Path, help="finetuned dit.safetensors; omit for the base model")
    p.add_argument("--tag", default="model")
    p.add_argument("--bf16-roundtrip", action="store_true",
                   help="diagnostic: quantise weights to bfloat16 and back")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="blend toward the base: 0 = pretrained, 1 = fully finetuned")
    p.add_argument("--prompt", default='TrackType: Music, VocalType: Instrumental, Genre: Electronic. Electronic dance music recorded from a live DJ set, club sound system, driving drums and synthesizer bass.')
    p.add_argument("--cond-cache", type=Path, default=Path("runs/base/prompt_cond.pt"),
                   help="cached prompt embedding; the text encoder is 1.2 GB and the "
                        "prompt is constant for a run, so it is usually not on disk")
    p.add_argument("--seconds", type=float, default=95.0)
    p.add_argument("--context", type=float, default=32.0, help="seconds of audio given as context")
    p.add_argument("--at", type=float, nargs="+", default=[600.0, 3600.0, 7200.0],
                   help="offsets in the source to take context from")
    p.add_argument("--steps", type=int, help="default 50 for the base objective, 8 for the distilled one")
    p.add_argument("--cfg", type=float, help="default 4.0 for the base objective, 1.0 for the distilled one")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    dev = torch.device("cuda")
    a.out.mkdir(parents=True, exist_ok=True)
    cached = a.cond_cache.exists()
    cfg, model = load(a.model, dev, conditioner=not cached, dit=a.dit, objective=a.objective)
    model.pretransform.to(torch.bfloat16)
    distilled = model.diffusion_objective == "rf_denoiser"
    a.steps = a.steps or (8 if distilled else 50)
    a.cfg = a.cfg if a.cfg is not None else (1.0 if distilled else 4.0)
    if a.resume:
        delta = load_file(a.resume)  # checkpoints hold the delta, not the weights
        # alpha scales the update: 0 recovers the base model, 1 the full finetune.
        model.model.load_state_dict(
            {k: v + a.alpha * delta[k].to(v.dtype).to(v.device)
             for k, v in model.model.state_dict().items()})
    if a.bf16_roundtrip:
        sd = model.model.state_dict()
        model.model.load_state_dict({k: v.to(torch.bfloat16).float() for k, v in sd.items()})
    model.eval()

    sr = cfg["sample_rate"]
    data = Excerpts([a.audio], a.seconds, sr, a.prompt)
    if cached:  # the text encoder may have been reclaimed; the tensor is a constant anyway
        prompt = tuple(t.to(dev) for t in torch.load(a.cond_cache))
        number = build_number_conditioner(a.model, dev)
        cond = {"prompt": tuple(t[:1] for t in prompt),
                "seconds_total": number([{"seconds_total": a.seconds}], dev)["seconds_total"]}
    else:
        cond = model.conditioner([{"prompt": a.prompt, "seconds_total": a.seconds}], dev)

    for off in a.at:
        with torch.autocast("cuda", torch.bfloat16):
            z = model.pretransform.encode(data.at(off)[0][None].to(dev)).float()
        keep = round(z.shape[-1] * a.context / a.seconds)
        mask = torch.ones_like(z[:, :1])
        mask[..., keep:] = 0

        c = dict(cond)
        c["inpaint_mask"], c["inpaint_masked_input"] = [mask], [z * mask]
        # Ping-pong redraws noise inside its loop from the global generator, so the
        # seed has to be set here as well for two models to be comparable.
        torch.manual_seed(a.seed)
        # Inference uses sampling_dist_shift, not the training-time dist_shift.
        # They are different objects and the model defaults the sampling one to a
        # LogSNR schedule; using the training shift here warps the whole trajectory.
        sigmas = build_schedule(a.steps, dist_shift=model.sampling_dist_shift,
                               effective_seq_len=z.shape[-1], device=dev)
        # pingpong is for the adversarially distilled rf_denoiser objective; the
        # base checkpoint is plain rectified flow and wants euler.
        sampler = (sample_flow_pingpong if model.diffusion_objective == "rf_denoiser"
                   else sample_discrete_euler)
        with torch.autocast("cuda", torch.bfloat16):
            out = sampler(model, torch.randn_like(z), sigmas, disable_tqdm=True, cond=c,
                          cfg_scale=a.cfg)

        for name, latent in ((a.tag, out), ("truth", z)):
            path = a.out / f"{int(off)}s_{name}.flac"
            if name == "truth" and path.exists():
                continue
            with torch.autocast("cuda", torch.bfloat16):
                audio = model.pretransform.decode(latent.to(torch.bfloat16))
            sf.write(path, audio.float().clamp(-1, 1)[0].T.cpu().numpy(), sr)
            print(f"wrote {path}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
