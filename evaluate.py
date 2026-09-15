#!/usr/bin/env python
"""Compare a finetune against its starting point over several excerpts and seeds.

One 190 s sample at one offset cannot resolve progress: the spread between seeds
is as large as the difference between checkpoints. This renders a grid of
offsets and seeds for both models in one process, so the weights load once and
every pair sees identical noise, and reports the mean and spread of each metric.

Two metrics, both over the generated region only:

  vs seed    distance between the generated audio's log-mel envelope and the
             envelope of the context it was handed. This is what outpainting is
             asked to do: continue in character.
  vs truth   the same distance against what the recording actually did next.
             Reported for reference, but on a long DJ set the real continuation
             drifts to another track, so this bottoms out at the source's own
             internal variation rather than at zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file

from stable_audio_3.inference.sampling import (
    build_schedule, sample_discrete_euler, sample_flow_pingpong)
from train import build_number_conditioner, load


def mel_envelope(x, sr, bands=32, n=2048):
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
    p.add_argument("--model", type=Path, default=Path("models/stable-audio-3-medium"))
    p.add_argument("--dit", type=Path,
                   default=Path("models/stable-audio-3-medium-base/dit_base.safetensors"))
    p.add_argument("--resume", type=Path, default=Path("runs/base/dit.safetensors"),
                   help="delta to add; pass --no-resume to evaluate the loaded model alone, "
                        "which is what you want once ship.py has already merged it in")
    p.add_argument("--no-resume", dest="resume", action="store_const", const=None)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="scale on the delta. -1 subtracts it, which recovers the model "
                        "ship.py started from without downloading it again.")
    p.add_argument("--latents", type=Path, default=Path("latents.npy"))
    p.add_argument("--out", type=Path, default=Path("eval"))
    p.add_argument("--at", type=float, nargs="+",
                   help="offsets to evaluate; by default they are drawn from the energy "
                        "index, because arbitrary offsets in a 26 h set land in dead air")
    p.add_argument("--index", default="energy_index.npz")
    p.add_argument("--min-rms", type=float, default=0.15)
    p.add_argument("--n-offsets", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--seconds", type=float, default=190.0)
    p.add_argument("--context", type=float, default=30.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--objective", choices=["rectified_flow", "rf_denoiser"],
                   default="rectified_flow",
                   help="picks the sampler. The base transformer is rectified_flow and "
                        "wants an ODE solver; leaving it at the released config's "
                        "rf_denoiser samples it with ping-pong, which is the exact "
                        "mismatch this project exists to document.")
    p.add_argument("--fps", type=float, default=44100 / 4096)
    p.add_argument("--write", action="store_true", help="also write the audio")
    a = p.parse_args()

    dev = torch.device("cuda")
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.at:
        # Same gate training uses. Spread the picks across the whole recording so
        # the excerpts are independent rather than neighbouring.
        from dataset import gated_starts
        starts, _ = gated_starts(a.index, a.seconds, a.min_rms, 0.0)
        a.at = [float(starts[i]) for i in
                np.linspace(0, len(starts) - 1, a.n_offsets).round().astype(int)]
        print("offsets from the energy gate: " + ", ".join(f"{x:.0f}s" for x in a.at))
    cfg, model = load(a.model, dev, conditioner=False, dit=a.dit, objective=a.objective)
    # The transformer stays float32 and lets autocast handle the matmuls, exactly
    # as training and sample.py do. Casting its weights to bfloat16 outright costs
    # nothing on a single forward pass but compounds over a 50-step solve: it drove
    # the untrained model's output to 0.07x the true level here before this line
    # was removed.
    model.pretransform.to(torch.bfloat16)
    model.eval()
    sr = cfg["sample_rate"]

    number = build_number_conditioner(a.model, dev)
    prompt = tuple(t.to(dev) for t in torch.load("runs/base/prompt_cond.pt"))
    cond = {"prompt": tuple(t[:1] for t in prompt),
            "seconds_total": number([{"seconds_total": a.seconds}], dev)["seconds_total"]}

    sampler = (sample_flow_pingpong if model.diffusion_objective == "rf_denoiser"
               else sample_discrete_euler)
    start = {k: v.clone() for k, v in model.model.state_dict().items()}
    delta, step = None, None
    if a.resume:
        delta = load_file(a.resume)
        if set(delta) != set(start):
            raise SystemExit(f"checkpoint keys do not match the model: "
                             f"{len(set(delta) ^ set(start))} differ")
        with __import__("safetensors").safe_open(str(a.resume), framework="pt") as f:
            step = f.metadata()["step"]

    frames = round(a.seconds * a.fps)
    keep = round(a.context * a.fps)
    stream = np.load(a.latents, mmap_mode="r")

    rows = {}
    other = f"step {step}" if a.alpha > 0 else "stock 8-step"
    tags = ("model",) if delta is None else ("start", other)
    for tag in tags:
        model.model.load_state_dict(start if tag in ("start", "model") else
                                    {k: v + a.alpha * delta[k].to(v.dtype).to(v.device)
                                     for k, v in start.items()})
        rec = []
        for off in a.at:
            arr = stream[:, round(off * a.fps):][:, :frames]
            if arr.shape[1] < frames:
                continue
            z = torch.from_numpy(np.asarray(arr)).float()[None].to(dev)
            mask = torch.zeros_like(z[:, :1])
            mask[..., :keep] = 1.0
            c = dict(cond)
            c["inpaint_mask"], c["inpaint_masked_input"] = [mask], [z * mask]
            sig = build_schedule(a.steps, dist_shift=model.sampling_dist_shift,
                                 effective_seq_len=z.shape[-1], device=dev)
            with torch.autocast("cuda", torch.bfloat16):
                truth = model.pretransform.decode(z.to(torch.bfloat16)).float()[0].mean(0).cpu().numpy()
            t_seed, t_gen = truth[:round(a.context * sr)], truth[round(a.context * sr):]
            e_seed, e_truth = mel_envelope(t_seed, sr), mel_envelope(t_gen, sr)
            for seed in a.seeds:
                g = torch.Generator(device=dev).manual_seed(seed)
                noise = torch.randn(z.shape, generator=g, device=dev)
                # Ping-pong redraws noise inside its loop from the global generator,
                # so seeding only the starting noise leaves the two models being
                # compared on different trajectories. Seed globally as well.
                torch.manual_seed(seed)
                with torch.autocast("cuda", torch.bfloat16):
                    out = sampler(model, noise, sig, disable_tqdm=True, cond=c, cfg_scale=a.cfg)
                    wav = model.pretransform.decode(out.to(torch.bfloat16)).float()[0].cpu()
                gen = wav.mean(0).numpy()[round(a.context * sr):]
                env = mel_envelope(gen, sr)
                rec.append((np.abs(env - e_seed).mean(), np.abs(env - e_truth).mean(),
                            np.sqrt((gen ** 2).mean()) / np.sqrt((t_gen ** 2).mean())))
                if a.write:
                    name = f"{int(off)}s_seed{seed}_{tag.replace(' ', '')}.mp3"
                    sf.write(a.out / name, wav.clamp(-1, 1).T.numpy(), sr,
                             format="MP3", subtype="MPEG_LAYER_III")
                print(f"  {tag:10s} {off:8.0f}s seed {seed}  "
                      f"vs seed {rec[-1][0]:.4f}  vs truth {rec[-1][1]:.4f}  "
                      f"level {rec[-1][2]:.2f}x", flush=True)
            if tag in ("start", "model"):
                rows.setdefault("reference", []).append(
                    (np.abs(e_truth - e_seed).mean(), 0.0,
                     np.sqrt((t_gen ** 2).mean()) / np.sqrt((t_seed ** 2).mean())))
        rows[tag] = rec

    print(f"\n{'model':16s} {'vs seed':>16s} {'vs truth':>16s} {'level':>14s}   n")
    for tag, rec in rows.items():
        r = np.array(rec)
        print(f"{tag:16s} {r[:,0].mean():8.4f} +-{r[:,0].std():5.4f} "
              f"{r[:,1].mean():8.4f} +-{r[:,1].std():5.4f} "
              f"{r[:,2].mean():7.2f}x +-{r[:,2].std():4.2f}  {len(r):3d}")
    print("\n`reference` is what the real recording does across the same boundary.")


if __name__ == "__main__":
    main()
