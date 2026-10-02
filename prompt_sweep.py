#!/usr/bin/env python
"""How much does the text prompt still steer this model?

The finetune saw one caption for its entire run, so the text pathway was trained
with a constant input while its cross-attention weights moved. Whether prompts
still work is therefore a question to measure, not assume. This holds the seed
audio, the noise and everything else fixed and varies only the caption and the
guidance scale.

Guidance is the knob that has never been touched here: the post-trained model is
sampled at 1.0 because that is what it is distilled for, which is also the setting
at which text has the least influence.
"""
from __future__ import annotations

import argparse, itertools
from pathlib import Path

import numpy as np, soundfile as sf, torch
from safetensors.torch import load_file

from dataset import Excerpts
from eval_songs import envelope, load_arc
from stable_audio_3.inference.sampling import build_schedule, sample_flow_pingpong
from train import build_number_conditioner, load

PROMPTS = {
 "trained":  "TrackType: Music, VocalType: Instrumental, Genre: Electronic. Electronic dance music recorded from a live DJ set, club sound system, driving drums and synthesizer bass.",
 "dubstep":  "TrackType: Music, VocalType: Instrumental, Genre: Dubstep. Heavy modern dubstep, half-time drums, aggressive wobble bass, deep sub, metallic growls, big drop.",
 "riddim":   "TrackType: Music, VocalType: Instrumental, Genre: Dubstep. Riddim dubstep, triplet bass stabs, mechanical growls, sparse percussion, heavy low end.",
 "hardstyle":"TrackType: Music, VocalType: Instrumental, Genre: Hardstyle. Hardstyle, distorted kick, reverse bass, euphoric synth lead, fast tempo.",
 "ambient":  "TrackType: Music, VocalType: Instrumental, Genre: Ambient. Slow ambient drone, soft pads, no drums, quiet and spacious.",
}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("models/stable-audio-3-medium"))
    p.add_argument("--arc", type=Path,
                   default=Path("models/stable-audio-3-medium/dit_arc.safetensors"))
    p.add_argument("--resume", type=Path, default=Path("runs/base/dit.safetensors"))
    p.add_argument("--out", type=Path, default=Path("prompts"))
    p.add_argument("--token", default=str(Path.home() / ".hf_token"))
    p.add_argument("--seconds", type=float, default=190.0)
    p.add_argument("--context", type=float, default=30.0)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cfg", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stock", action="store_true", help="also sweep the released model")
    a = p.parse_args()

    dev = torch.device("cuda")
    a.out.mkdir(parents=True, exist_ok=True)
    cfg, model = load(a.model, dev)          # with the text encoder this time
    model.pretransform.to(torch.bfloat16)
    sr = cfg["sample_rate"]

    print("encoding prompts", flush=True)
    conds = {}
    for name, text in PROMPTS.items():
        conds[name] = model.conditioner([{"prompt": text, "seconds_total": a.seconds}], dev)
    del model.conditioner
    torch.cuda.empty_cache()

    arc = load_arc(a.arc, a.token, dev)
    delta = load_file(a.resume)
    merged = {k: v + delta[k].to(v.dtype).to(v.device) for k, v in arc.items()}
    model.diffusion_objective = "rf_denoiser"
    model.eval()

    audio = Excerpts([a.audio], a.seconds, sr).at(0.0)[0][None].to(dev)
    with torch.autocast("cuda", torch.bfloat16):
        z = model.pretransform.encode(audio.to(torch.bfloat16)).float()
    mask = torch.zeros_like(z[:, :1])
    mask[..., :round(a.context * z.shape[-1] / a.seconds)] = 1.0
    cut = round(a.context * sr)
    seed_env = envelope(audio[0].mean(0).cpu().numpy()[:cut], sr)
    seed_rms = np.sqrt((audio[0].mean(0).cpu().numpy()[:cut] ** 2).mean())

    variants = [("yours", merged)] + ([("stock", arc)] if a.stock else [])
    print(f"\n{'model':6s} {'prompt':10s} {'cfg':>4s} {'vs seed':>8s} {'level':>7s} {'centroid':>9s}")
    for tag, w in variants:
        model.model.load_state_dict(w)
        for name, scale in itertools.product(PROMPTS, a.cfg):
            c = dict(conds[name])
            c["inpaint_mask"], c["inpaint_masked_input"] = [mask], [z * mask]
            sig = build_schedule(a.steps, dist_shift=model.sampling_dist_shift,
                                 effective_seq_len=z.shape[-1], device=dev)
            g = torch.Generator(device=dev).manual_seed(a.seed)
            noise = torch.randn(z.shape, generator=g, device=dev)
            torch.manual_seed(a.seed)
            with torch.autocast("cuda", torch.bfloat16):
                out = sample_flow_pingpong(model, noise, sig, disable_tqdm=True,
                                           cond=c, cfg_scale=scale)
                wav = model.pretransform.decode(out.to(torch.bfloat16)).float()[0].cpu()
            sf.write(a.out / f"{name}_cfg{scale:g}_{tag}.mp3",
                     wav.clamp(-1, 1).T.numpy(), sr, format="MP3", subtype="MPEG_LAYER_III")
            gen = wav.mean(0).numpy()[cut:]
            n = 2048
            fr = gen[: len(gen)//(n//2)*(n//2)]
            fr = np.lib.stride_tricks.sliding_window_view(fr, n)[::n//2]
            S = np.abs(np.fft.rfft(fr*np.hanning(n), axis=1)); f = np.fft.rfftfreq(n, 1/sr)
            cen = ((S*f).sum(1)/(S.sum(1)+1e-9)).mean()
            print(f"{tag:6s} {name:10s} {scale:4g} "
                  f"{np.abs(envelope(gen,sr)-seed_env).mean():8.4f} "
                  f"{np.sqrt((gen**2).mean())/seed_rms:6.2f}x {cen:8.0f} Hz", flush=True)


if __name__ == "__main__":
    main()
