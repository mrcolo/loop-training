#!/usr/bin/env python
"""LoRA on top of loop-0, from a folder of tracks. One file, no repo imports.

    python lora.py --tracks /path/to/tracks --name loop-0-evancloud

What it builds on, and why in this order:

  base transformer  stabilityai/stable-audio-3-medium-base   the only checkpoint a
                                                             squared-error loss is
                                                             valid on
  + loop-0 delta    the 26.4 h 2n1t3 mid-train               what "loop" sounds like
  + this LoRA       trained here, on your tracks             what these tracks add

Training happens on base + loop-0. Inference happens on the *post-trained*
transformer + loop-0 delta + LoRA delta, sampled in 8 steps. Every piece is a
delta from the same base, so they add. Training the LoRA against the
post-trained weights instead would undo their adversarial post-training -- the
failure this project spent a day diagnosing.

Outputs, in runs/<name>/:
  lora.safetensors         rank-r factors (A, B) per adapted linear, plus config
  delta.safetensors        loop-0 delta + LoRA folded in, fp16; drop-in for
                           eval_songs.py / sample.py --resume
  demo_<track>.mp3         a rendered continuation, if --demo
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from stable_audio_3.factory import create_multi_conditioner_from_conditioning_config
from stable_audio_3.inference.sampling import (
    build_schedule, sample_flow_pingpong, truncated_logistic_normal_rescaled)
from stable_audio_3.loading_utils import copy_state_dict, load_diffusion_cond
from stable_audio_3.models.inpainting import MaskType, random_inpaint_mask

AUDIO = (".wav", ".flac", ".mp3", ".aiff", ".aif", ".m4a", ".ogg")
# The projections Muon trained in loop-0: attention and feed-forward. These are
# where a LoRA has the most leverage per parameter.
TARGETS = ("self_attn.to_qkv", "self_attn.to_out", "cross_attn.to_q",
           "cross_attn.to_kv", "cross_attn.to_out", "ff.ff.0.proj", "ff.ff.2")


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x, with W frozen and B initialised to zero,
    so step zero is exactly loop-0."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t().to(x.dtype)
                               @ self.B.t().to(x.dtype)) * self.scale

    def delta(self) -> torch.Tensor:
        return (self.B @ self.A) * self.scale


def add_lora(dit: nn.Module, rank: int, alpha: float, dropout: float) -> dict[str, LoRALinear]:
    found = {}
    for name, mod in list(dit.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(full.endswith(t) for t in TARGETS):
                wrapped = LoRALinear(child, rank, alpha, dropout).to(child.weight.device)
                setattr(mod, child_name, wrapped)
                found[full] = wrapped
    if not found:
        raise SystemExit("no target linears found; the model layout has changed")
    return found


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_dir: Path, base_dit: Path, loop_delta: Path, dev):
    """Autoencoder + base transformer + loop-0 delta. No text encoder: the
    prompt embedding is cached, exactly as loop-0 was trained."""
    cfg = json.loads((model_dir / "model_config.json").read_text())
    cfg["model"]["conditioning"]["configs"] = []
    model = load_diffusion_cond(cfg, str(model_dir / "model.safetensors"), device=dev)
    copy_state_dict(model, {k: v.to(dev) for k, v in load_file(str(base_dit)).items()})
    delta = load_file(str(loop_delta))
    sd = model.model.state_dict()
    missing = set(sd) ^ set(delta)
    if missing:
        raise SystemExit(f"loop-0 delta does not match the transformer ({len(missing)} keys)")
    model.model.load_state_dict({k: v + delta[k].to(v.dtype).to(v.device) for k, v in sd.items()})
    model.diffusion_objective = "rectified_flow"
    model.pretransform.to(torch.bfloat16).eval().requires_grad_(False)
    return cfg, model, delta


def number_conditioner(model_dir: Path, dev):
    cfg = json.loads((model_dir / "model_config.json").read_text())
    cc = {"configs": [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "number"],
          "cond_dim": cfg["model"]["conditioning"]["cond_dim"]}
    number = create_multi_conditioner_from_conditioning_config(cc)
    with safe_open(str(model_dir / "model.safetensors"), framework="pt") as f:
        w = {k[len("conditioner."):]: f.get_tensor(k) for k in f.keys() if k.startswith("conditioner.")}
    number.load_state_dict(w, strict=False)
    return number.to(dev).eval().requires_grad_(False)


# ---------------------------------------------------------------------------
# Data: every track encoded once, kept in memory as latents
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_tracks(paths, ae, sr: int, dev, block_s=120.0, pad_s=8.0, cache: Path | None = None):
    """Overlapping blocks, trimmed, so no encoder seam survives. The autoencoder
    must stay in eval(): its bottleneck noise is 50x larger in train mode."""
    ratio = int(ae.downsampling_ratio)
    out = []
    for i, p in enumerate(paths):
        hit = cache / (p.stem + ".npy") if cache else None
        if hit is not None and hit.exists():
            out.append((p.stem, torch.from_numpy(np.load(hit)).float()))
            continue
        x, fsr = sf.read(str(p), dtype="float32", always_2d=True)
        x = torch.from_numpy(x).T
        if fsr != sr:
            x = torchaudio.functional.resample(x, fsr, sr)
        x = x[:2] if x.shape[0] >= 2 else x.repeat(2, 1)
        block, pad, n = int(block_s * sr), int(pad_s * sr), x.shape[1]
        parts = []
        for start in range(0, n, block):
            lo, hi = max(0, start - pad), min(n, start + block + pad)
            with torch.autocast("cuda", torch.bfloat16):
                z = ae.encode(x[None, :, lo:hi].to(dev).to(torch.bfloat16))[0].float().cpu()
            skip = (start - lo) // ratio
            keep = (min(start + block, n) - start) // ratio
            parts.append(z[:, skip:skip + keep])
        z = torch.cat(parts, 1)
        if hit is not None:
            np.save(hit, z.numpy().astype(np.float16))
        out.append((p.stem, z))
        print(f"  [{i + 1}/{len(paths)}] {p.name}: {n / sr:.0f} s -> {z.shape[1]} frames", flush=True)
    return out


def draw(latents, choices_s, fps, rms_gate: float):
    """A random window from a random track, weighted by length. The window
    length is drawn from `choices_s` but never exceeds the track; windows whose
    latent energy is far below the track's are re-drawn (intros, silence)."""
    weights = [z.shape[1] for _, z in latents]
    for _ in range(20):
        name, z = random.choices(latents, weights=weights)[0]
        frames = min(round(random.choice(choices_s) * fps), z.shape[1])
        start = random.randint(0, z.shape[1] - frames)
        w = z[:, start:start + frames]
        if w.std() >= rms_gate * z.std():
            return w, frames / fps
    return w, frames / fps


def outpaint_mask(z, fps, p_full, p_segments, ctx_s=(20.0, 60.0), min_gen=15.0):
    """1 = given as context, 0 = generated. Same mix loop-0 was trained on."""
    T = z.shape[-1]
    r = random.random()
    if r < p_full:
        m = torch.zeros(1, 1, T, device=z.device)
    elif r < p_full + p_segments:
        _, m = random_inpaint_mask(z, padding_masks=torch.ones(1, T, dtype=torch.bool, device=z.device),
                                   force_mask_type=MaskType.RANDOM_SEGMENTS)
    else:
        hi = max(1.0 / fps, min(ctx_s[1], T / fps - min_gen))
        lo = min(ctx_s[0], hi)
        k = min(T - 1, max(1, round(random.uniform(lo, hi) * fps)))
        m = torch.zeros(1, 1, T, device=z.device)
        m[..., :k] = 1.0
    return z * m, m


def masked_mean(err, region):
    n = region.sum(dim=(1, 2)) * err.shape[1]
    return ((err * region).sum(dim=(1, 2)) / n.clamp(min=1)).mean()


# ---------------------------------------------------------------------------
# Demo: render on the post-trained transformer, 8 steps, as it ships
# ---------------------------------------------------------------------------

@torch.no_grad()
def demo(model, arc_path: Path, full_delta: dict, latents, prompt, number, out: Path,
         seconds=190.0, context=30.0, fps=44100 / 4096, sr=44100):
    arc = load_file(str(arc_path))
    arc = {(k[len("model."):] if k.startswith("model.model.") else k): v for k, v in arc.items()}
    sd = model.model.state_dict()
    model.model.load_state_dict({k: arc[k].to(v.device, v.dtype) + full_delta[k].to(v.device, v.dtype)
                                 for k, v in sd.items()})
    model.diffusion_objective = "rf_denoiser"
    model.eval()
    dev = next(model.parameters()).device
    for name, z in latents[:3]:
        frames = min(round(seconds * fps), z.shape[1])
        if frames < round((context + 15) * fps):
            continue
        z = z[:, :frames][None].to(dev)
        mask = torch.zeros_like(z[:, :1])
        mask[..., :round(context * fps)] = 1.0
        c = {"prompt": prompt, "seconds_total": number([{"seconds_total": frames / fps}], dev)["seconds_total"],
             "inpaint_mask": [mask], "inpaint_masked_input": [z * mask]}
        sig = build_schedule(8, dist_shift=model.sampling_dist_shift, effective_seq_len=z.shape[-1], device=dev)
        g = torch.Generator(device=dev).manual_seed(0)
        torch.manual_seed(0)
        with torch.autocast("cuda", torch.bfloat16):
            x = sample_flow_pingpong(model, torch.randn(z.shape, generator=g, device=dev), sig,
                                     disable_tqdm=True, cond=c, cfg_scale=1.0)
            wav = model.pretransform.decode(x.to(torch.bfloat16)).float()[0].cpu()
        sf.write(out / f"demo_{name[:40]}.mp3", wav.clamp(-1, 1).T.numpy(), sr,
                 format="MP3", subtype="MPEG_LAYER_III")
        print(f"  demo: {out / f'demo_{name[:40]}.mp3'}", flush=True)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracks", type=Path, required=True, help="folder of audio; searched recursively")
    p.add_argument("--name", default="loop-0-evancloud")
    p.add_argument("--model", type=Path, default=Path("models/stable-audio-3-medium"))
    p.add_argument("--base-dit", type=Path, default=Path("models/stable-audio-3-medium-base/dit_base.safetensors"))
    p.add_argument("--loop-delta", type=Path, default=Path("runs/base/dit.safetensors"),
                   help="the loop-0 delta to build on")
    p.add_argument("--arc", type=Path, default=Path("models/stable-audio-3-medium/dit_arc.safetensors"),
                   help="post-trained transformer, used only for the demo")
    p.add_argument("--prompt-cache", type=Path, default=Path("runs/base/prompt_cond.pt"))
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-4, help="Stability's LoRA default")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--seconds", type=float, nargs="+", default=[47.0, 95.0, 190.0])
    p.add_argument("--p-full", type=float, default=0.55)
    p.add_argument("--p-segments", type=float, default=0.10)
    p.add_argument("--rms-gate", type=float, default=0.5, help="reject windows quieter than this x the track")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--holdout-match", nargs="+", default=[],
                   help="hold out tracks whose file name starts with any of these strings")
    p.add_argument("--holdout", type=int, default=0,
                   help="keep N tracks out of training, listed in holdout.txt, for evaluation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--demo", action="store_true", help="render 8-step continuations at the end")
    a = p.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    out = Path("runs") / a.name
    (out / "latents").mkdir(parents=True, exist_ok=True)

    paths = sorted(q for q in a.tracks.rglob("*") if q.suffix.lower() in AUDIO and not q.name.startswith("._"))
    if not paths:
        raise SystemExit(f"no audio under {a.tracks}")
    if a.holdout_match or a.holdout:
        held = ([q for q in paths if any(q.name.lower().startswith(m.lower()) for m in a.holdout_match)]
                if a.holdout_match else random.Random(a.seed).sample(paths, a.holdout))
        if a.holdout_match and len(held) != len(a.holdout_match):
            raise SystemExit(f"--holdout-match matched {len(held)} files for {len(a.holdout_match)} patterns: "
                             + ", ".join(q.name for q in held))
        paths = [q for q in paths if q not in held]
        (out / "holdout.txt").write_text("\n".join(str(q) for q in held) + "\n")
        print(f"holding out {len(held)} tracks for evaluation", flush=True)
    print(f"{len(paths)} tracks", flush=True)

    cfg, model, loop_delta = load_model(a.model, a.base_dit, a.loop_delta, dev)
    sr = cfg["sample_rate"]
    fps = sr / int(model.pretransform.downsampling_ratio)
    latents = encode_tracks(paths, model.pretransform, sr, dev, cache=out / "latents")
    hours = sum(z.shape[1] for _, z in latents) / fps / 3600
    latents = [(n, z) for n, z in latents if z.shape[1] >= round(30 * fps)]
    print(f"{hours:.2f} h of audio, {len(latents)} tracks long enough to use", flush=True)

    number = number_conditioner(a.model, dev)
    prompt = tuple(t.to(dev)[:1] for t in torch.load(a.prompt_cache))

    dit = model.model
    dit.requires_grad_(False)
    adapters = add_lora(dit, a.rank, a.alpha, a.dropout)
    params = [q for m in adapters.values() for q in (m.A, m.B)]
    print(f"LoRA rank {a.rank} on {len(adapters)} linears, "
          f"{sum(q.numel() for q in params) / 1e6:.1f} M trainable", flush=True)
    dit.train()
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)

    def save(step):
        sd = {}
        for k, m in adapters.items():
            sd[f"{k}.lora_A"] = m.A.detach().cpu().contiguous()
            sd[f"{k}.lora_B"] = m.B.detach().cpu().contiguous()
        meta = {"step": str(step), "rank": str(a.rank), "alpha": str(a.alpha),
                "base": "stable-audio-3-medium-base", "on_top_of": str(a.loop_delta)}
        save_file(sd, str(out / "lora.safetensors"), metadata=meta)
        save_file(sd, str(out / f"lora_{step:06d}.safetensors"), metadata=meta)
        # Fold into one delta from base, so every existing tool can use it.
        full = {k: v.float().clone() for k, v in loop_delta.items()}
        for k, m in adapters.items():
            full[f"{k}.weight"] += m.delta().detach().cpu()
        save_file({k: v.to(torch.float16) for k, v in full.items()},
                  str(out / "delta.safetensors"), metadata=meta)
        return full

    t0, agg = time.time(), 0.0
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, step / max(a.warmup, 1))
        for _ in range(a.accum):
            w, secs = draw(latents, a.seconds, fps, a.rms_gate)
            z = w[None].to(dev)
            masked, mask = outpaint_mask(z, fps, a.p_full, a.p_segments)
            c = {"prompt": prompt, "seconds_total": number([{"seconds_total": secs}], dev)["seconds_total"],
                 "inpaint_mask": [mask], "inpaint_masked_input": [masked]}
            t = 1 - truncated_logistic_normal_rescaled(1).to(dev)
            t = model.dist_shift.shift(t, z.shape[-1]) if model.dist_shift else t
            noise = torch.randn_like(z)
            with torch.autocast("cuda", torch.bfloat16):
                pred = model(z * (1 - t[:, None, None]) + noise * t[:, None, None], t,
                             cond=c, cfg_dropout_prob=0.1)
            err = F.mse_loss(pred.float(), noise - z, reduction="none")
            loss = masked_mean(err, 1 - mask) + masked_mean(err, mask)
            (loss / a.accum).backward()
            agg += loss.item() / a.accum
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            print(f"step {step:5d}  loss {agg / 10:.4f}  {(time.time() - t0) / 10:.2f} s/step", flush=True)
            agg, t0 = 0.0, time.time()
        if step % a.save_every == 0 or step == a.steps:
            save(step)
            print(f"saved {out}/lora.safetensors and delta.safetensors at step {step}", flush=True)

    full = save(a.steps)
    if a.demo:
        for k, m in adapters.items():          # unwrap before loading plain weights
            parent, _, child = k.rpartition(".")
            setattr(dit.get_submodule(parent), child, m.base)
        torch.cuda.empty_cache()
        demo(model, a.arc, full, latents, prompt, number, out)
    print("done")


if __name__ == "__main__":
    main()
