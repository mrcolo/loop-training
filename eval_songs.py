#!/usr/bin/env python
"""The standing evaluation: outpaint every track in a folder.

Loads the model once, streams the post-trained transformer once, applies the
finetune delta once, then renders every song. Doing this per-song instead would
re-download 9.2 GB each time, which is the difference between half an hour and
most of a day.

Each render takes the first `--context` seconds of a track as the seed and
generates out to `--seconds`, at the eight steps the merged model is meant for.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from dataset import Excerpts
from remote_weights import stream_state_dict
from stable_audio_3.inference.sampling import (
    build_schedule, sample_discrete_euler, sample_flow_pingpong)
from train import build_number_conditioner, load

ARC = "https://huggingface.co/stabilityai/stable-audio-3-medium/resolve/main/model.safetensors"


def load_arc(path: Path, token_file: str, device):
    """The post-trained transformer, from disk if we have it and the Hub if not.

    It is 2.9 GB and never changes, so it is worth keeping: with it local, merging
    a new checkpoint is a read and an addition rather than a 9.2 GB download.
    """
    if path.exists():
        print(f"post-trained transformer from {path}", flush=True)
        return {k[len("model."):] if k.startswith("model.model.") else k: v.to(device)
                for k, v in load_file(str(path)).items()}
    token = Path(token_file).read_text().strip()
    print("streaming the post-trained transformer from the Hub (once)", flush=True)
    sd = stream_state_dict(ARC, token, prefix="model.model.", device=device,
                           progress=lambda f: print(f"  {f:5.1%}", flush=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.cpu() for k, v in sd.items()}, str(path))
    print(f"saved to {path}; no further downloads needed", flush=True)
    return {k[len("model."):]: v for k, v in sd.items()}


def envelope(x, sr, bands=32, n=2048):
    fr = x[: len(x) // (n // 2) * (n // 2)]
    fr = np.lib.stride_tricks.sliding_window_view(fr, n)[:: n // 2]
    S = np.abs(np.fft.rfft(fr * np.hanning(n), axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    mel = lambda h: 2595 * np.log10(1 + h / 700)
    e = np.linspace(mel(20), mel(sr / 2), bands + 1)
    hz = 700 * (10 ** (e / 2595) - 1)
    b = np.stack([S[:, (f >= hz[i]) & (f < hz[i + 1])].sum(1) for i in range(bands)], 1)
    return np.log10(b.mean(0) + 1e-10)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--songs", type=Path, default=Path("/home/stem-user/eval"))
    p.add_argument("--model", type=Path, default=Path("models/stable-audio-3-medium"))
    p.add_argument("--dit", type=Path,
                   default=Path("models/stable-audio-3-medium-base/dit_base.safetensors"))
    p.add_argument("--resume", type=Path, default=Path("runs/base/dit.safetensors"))
    p.add_argument("--out", type=Path, default=Path("songs"))
    p.add_argument("--arc", type=Path,
                   default=Path("models/stable-audio-3-medium/dit_arc.safetensors"),
                   help="local copy of the post-trained transformer. Used when present; "
                        "streamed from the Hub and saved here when not, so the 9.2 GB "
                        "download happens exactly once.")
    p.add_argument("--token", default=str(Path.home() / ".hf_token"))
    p.add_argument("--seconds", type=float, default=190.0)
    p.add_argument("--context", type=float, default=30.0)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stock", action="store_true",
                   help="also render the released model on the same noise, for A/B")
    a = p.parse_args()

    songs = sorted([q for q in a.songs.iterdir()
                    if q.suffix.lower() in (".wav", ".flac", ".mp3", ".aiff", ".m4a")])
    if not songs:
        raise SystemExit(f"no audio in {a.songs}")
    print(f"{len(songs)} tracks: " + ", ".join(q.stem.split(" - ")[0] for q in songs), flush=True)

    dev = torch.device("cuda")
    a.out.mkdir(parents=True, exist_ok=True)
    cfg, model = load(a.model, dev, conditioner=False, dit=a.dit)
    model.pretransform.to(torch.bfloat16)
    sr = cfg["sample_rate"]

    arc = load_arc(a.arc, a.token, dev)
    delta = load_file(a.resume)
    with safe_open(str(a.resume), framework="pt") as f:
        step = f.metadata()["step"]
    if set(delta) != set(arc):
        raise SystemExit("checkpoint keys do not match the transformer")
    merged = {k: v + delta[k].to(v.dtype).to(v.device) for k, v in arc.items()}
    model.diffusion_objective = "rf_denoiser"
    model.eval()

    number = build_number_conditioner(a.model, dev)
    prompt = tuple(t.to(dev) for t in torch.load("runs/base/prompt_cond.pt"))
    cond = {"prompt": tuple(t[:1] for t in prompt),
            "seconds_total": number([{"seconds_total": a.seconds}], dev)["seconds_total"]}
    cut = round(a.context * sr)

    variants = [("step" + step, merged)] + ([("stock", arc)] if a.stock else [])
    rows = []
    for tag, weights in variants:
        model.model.load_state_dict(weights)
        for q in songs:
            name = q.stem.split(" - ")[0]
            audio = Excerpts([q], a.seconds, sr).at(0.0)[0][None].to(dev)
            with torch.autocast("cuda", torch.bfloat16):
                z = model.pretransform.encode(audio.to(torch.bfloat16)).float()
            mask = torch.zeros_like(z[:, :1])
            mask[..., :round(a.context * z.shape[-1] / a.seconds)] = 1.0
            c = dict(cond)
            c["inpaint_mask"], c["inpaint_masked_input"] = [mask], [z * mask]
            sig = build_schedule(a.steps, dist_shift=model.sampling_dist_shift,
                                 effective_seq_len=z.shape[-1], device=dev)
            g = torch.Generator(device=dev).manual_seed(a.seed)
            noise = torch.randn(z.shape, generator=g, device=dev)
            torch.manual_seed(a.seed)   # ping-pong redraws from the global generator
            with torch.autocast("cuda", torch.bfloat16):
                out = sample_flow_pingpong(model, noise, sig, disable_tqdm=True,
                                           cond=c, cfg_scale=a.cfg)
                wav = model.pretransform.decode(out.to(torch.bfloat16)).float()[0].cpu()
            sf.write(a.out / f"{name}_{tag}.mp3", wav.clamp(-1, 1).T.numpy(), sr,
                     format="MP3", subtype="MPEG_LAYER_III")
            seed_a = audio[0].mean(0).cpu().numpy()[:cut]
            gen = wav.mean(0).numpy()[cut:]
            d = np.abs(envelope(gen, sr) - envelope(seed_a, sr)).mean()
            lvl = np.sqrt((gen ** 2).mean()) / (np.sqrt((seed_a ** 2).mean()) + 1e-9)
            rows.append((tag, name, d, lvl))
            print(f"  {tag:10s} {name:12s} vs seed {d:.4f}  level {lvl:.2f}x", flush=True)

    print(f"\n{'model':12s} {'mean vs seed':>13s} {'mean level':>11s}")
    for tag, _ in variants:
        r = np.array([(d, l) for t, _, d, l in rows if t == tag])
        print(f"{tag:12s} {r[:,0].mean():13.4f} {r[:,1].mean():10.2f}x")


if __name__ == "__main__":
    main()
