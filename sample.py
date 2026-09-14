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
from stable_audio_3.inference.sampling import build_schedule, sample_flow_pingpong
from train import load


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("models/stable-audio-3-medium"))
    p.add_argument("--out", type=Path, default=Path("samples"))
    p.add_argument("--resume", type=Path, help="finetuned dit.safetensors; omit for the base model")
    p.add_argument("--tag", default="model")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="blend toward the base: 0 = pretrained, 1 = fully finetuned")
    p.add_argument("--prompt", default='TrackType: Music, VocalType: Instrumental, Genre: Electronic. Electronic dance music recorded from a live DJ set, club sound system, driving drums and synthesizer bass.')
    p.add_argument("--seconds", type=float, default=95.0)
    p.add_argument("--context", type=float, default=32.0, help="seconds of audio given as context")
    p.add_argument("--at", type=float, nargs="+", default=[600.0, 3600.0, 7200.0],
                   help="offsets in the source to take context from")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    dev = torch.device("cuda")
    a.out.mkdir(parents=True, exist_ok=True)
    cfg, model = load(a.model, dev)
    model.pretransform.to(torch.bfloat16)
    if a.resume:
        tuned = {k: v.float() for k, v in load_file(a.resume).items()}
        if a.alpha != 1.0:  # WiSE-FT: keep the base model's behaviour, add a fraction of the finetune
            base = model.model.state_dict()
            tuned = {k: (1 - a.alpha) * base[k].float().cpu() + a.alpha * v for k, v in tuned.items()}
        model.model.load_state_dict(tuned)
    model.eval()

    sr = cfg["sample_rate"]
    data = Excerpts([a.audio], a.seconds, sr, a.prompt)
    cond = model.conditioner([{"prompt": a.prompt, "seconds_total": a.seconds}], dev)

    for off in a.at:
        with torch.autocast("cuda", torch.bfloat16):
            z = model.pretransform.encode(data.at(off)[0][None].to(dev)).float()
        keep = round(z.shape[-1] * a.context / a.seconds)
        mask = torch.ones_like(z[:, :1])
        mask[..., keep:] = 0

        c = dict(cond)
        c["inpaint_mask"], c["inpaint_masked_input"] = [mask], [z * mask]
        torch.manual_seed(a.seed)
        sigmas = build_schedule(a.steps, dist_shift=model.dist_shift,
                               effective_seq_len=z.shape[-1], device=dev)
        with torch.autocast("cuda", torch.bfloat16):
            out = sample_flow_pingpong(model, torch.randn_like(z), sigmas, disable_tqdm=True, cond=c)

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
