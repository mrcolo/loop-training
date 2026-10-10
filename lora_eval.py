#!/usr/bin/env python
"""Score LoRA snapshots on held-out tracks, loading the model once.

    python lora_eval.py --songs <holdout dir> --out runs/<name>/eval.json \
        --lora /dev/shm/snaps/lora_000500.safetensors ... [--baselines] [--seeds 0 1]

Renders exactly as eval_songs.py does (post-trained transformer + loop-0 delta + the
adapter folded to fp16, bf16 merge, 8 ping-pong steps, the first --context seconds as
the seed of a --seconds window) and reports, per variant and averaged over tracks and
seeds:

  vs truth  32-band log-mel envelope distance to what the track really does after the
            seed. The selection metric.
  vs seed   the same distance to the seed. Rewards staying intro-like; don't select on it.
  level     generated RMS over the seed's.
  sub       share of energy under 60 Hz, generated region (the real continuation's is
            printed for reference). A checkpoint that thins the low end is a red flag.

--baselines adds the stock post-trained model and loop-0 on the same noise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from dataset import Excerpts
from eval_songs import envelope, load_arc
from stable_audio_3.inference.sampling import build_schedule, sample_flow_pingpong
from train import build_number_conditioner, load


def sub_share(x, sr, cut=60.0, n=8192):
    fr = np.lib.stride_tricks.sliding_window_view(x[: len(x) // n * n], n)[:: n // 2]
    S = np.abs(np.fft.rfft(fr * np.hanning(n), axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    return float(S[:, f < cut].sum() / S.sum())


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--songs", type=Path, required=True)
    p.add_argument("--lora", type=Path, nargs="*", default=[])
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--out", type=Path, required=True, help="results json (appended to if it exists)")
    p.add_argument("--model", type=Path, default=Path("models/stable-audio-3-medium"))
    p.add_argument("--dit", type=Path, default=Path("models/stable-audio-3-medium-base/dit_base.safetensors"))
    p.add_argument("--arc", type=Path, default=Path("models/stable-audio-3-medium/dit_arc.safetensors"))
    p.add_argument("--loop-delta", type=Path, default=Path("runs/base/dit.safetensors"))
    p.add_argument("--prompt-cache", type=Path, default=Path("runs/base/prompt_cond.pt"))
    p.add_argument("--seconds", type=float, default=190.0)
    p.add_argument("--context", type=float, default=30.0)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--render-dir", type=Path, default=None, help="also write each render here as wav")
    p.add_argument("--lean", action="store_true", help="bf16 matrices, ~4 GB of GPU: fits beside a trainer")
    a = p.parse_args()

    songs = sorted(q for q in a.songs.iterdir() if q.suffix.lower() in (".wav", ".flac", ".mp3"))
    dev = torch.device("cuda")
    where = torch.device("cpu") if a.lean else dev
    cfg, model = load(a.model, where, conditioner=False, dit=a.dit)
    model.pretransform.to(torch.bfloat16)
    sr = cfg["sample_rate"]
    arc = load_arc(a.arc, "", where)
    loop = load_file(str(a.loop_delta))
    model.diffusion_objective = "rf_denoiser"
    model.eval()
    if a.lean:
        for m in model.model.modules():
            if isinstance(m, torch.nn.Linear) and m.weight.numel() >= 1 << 20:
                m.to(torch.bfloat16)
        model.to(dev)
    number = build_number_conditioner(a.model, dev)
    prompt = tuple(t.to(dev) for t in torch.load(a.prompt_cache))
    cond = {"prompt": tuple(t[:1] for t in prompt),
            "seconds_total": number([{"seconds_total": a.seconds}], dev)["seconds_total"]}

    # Seeds and truths, encoded once.
    items = []
    for q in songs:
        audio = Excerpts([q], a.seconds, sr).at(0.0)[0][None].to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            z = model.pretransform.encode(audio.to(torch.bfloat16)).float()
        cut = round(a.context * sr)
        mono = audio[0].mean(0).cpu().numpy()
        items.append((q.stem, z, mono[:cut], mono[cut:]))

    def weights_for(lora_path):
        """eval_songs' merge: post-trained (bf16) + fp16 delta, added in the post-trained dtype."""
        if lora_path == "stock":
            return arc
        if lora_path == "loop-0":
            full = {k: v.to(torch.float16) for k, v in loop.items()}
        else:
            with safe_open(str(lora_path), framework="pt") as f:
                meta = f.metadata()
            scale = float(meta["alpha"]) / float(meta["rank"])
            lsd = load_file(str(lora_path))
            full = {k: v.float() for k, v in loop.items()}
            for k in lsd:
                if k.endswith(".lora_A"):
                    base = k[: -len(".lora_A")]
                    full[base + ".weight"] += (lsd[base + ".lora_B"] @ lsd[k]) * scale
            full = {k: v.to(torch.float16) for k, v in full.items()}
        return {k: v + full[k].to(v.dtype).to(v.device) for k, v in arc.items()}

    variants = (["stock", "loop-0"] if a.baselines else []) + [str(x) for x in a.lora]
    results = json.loads(a.out.read_text()) if a.out.exists() else {}
    for var in variants:
        w = weights_for(var)
        own = model.model.state_dict()
        model.model.load_state_dict({k: w[k].to(own[k].dtype) for k in own})
        del w
        rows = []
        for name, z, seed_a, true_a in items:
            mask = torch.zeros_like(z[:, :1])
            mask[..., :round(a.context * z.shape[-1] / a.seconds)] = 1.0
            c = dict(cond)
            c["inpaint_mask"], c["inpaint_masked_input"] = [mask], [z * mask]
            sig = build_schedule(8, dist_shift=model.sampling_dist_shift, effective_seq_len=z.shape[-1], device=dev)
            for s in a.seeds:
                g = torch.Generator(device=dev).manual_seed(s)
                noise = torch.randn(z.shape, generator=g, device=dev)
                torch.manual_seed(s)
                with torch.autocast("cuda", torch.bfloat16):
                    out = sample_flow_pingpong(model, noise, sig, disable_tqdm=True, cond=c, cfg_scale=a.cfg)
                    wav = model.pretransform.decode(out.to(torch.bfloat16)).float()[0].cpu()
                cut = len(seed_a)
                gen = wav.mean(0).numpy()[cut:]
                n = min(len(gen), len(true_a))
                rows.append({
                    "track": name, "seed": s,
                    "vs_truth": float(np.abs(envelope(gen[:n], sr) - envelope(true_a[:n], sr)).mean()),
                    "vs_seed": float(np.abs(envelope(gen, sr) - envelope(seed_a, sr)).mean()),
                    "level": float(np.sqrt((gen ** 2).mean()) / (np.sqrt((seed_a ** 2).mean()) + 1e-9)),
                    "sub": sub_share(gen[:n], sr), "sub_true": sub_share(true_a[:n], sr),
                    "peak": float(wav.abs().max())})
                if a.render_dir:
                    import soundfile as sf
                    a.render_dir.mkdir(parents=True, exist_ok=True)
                    tag = Path(var).stem if var not in ("stock", "loop-0") else var
                    sf.write(a.render_dir / f"{name}_{tag}_s{s}.wav", wav.T.numpy(), sr, subtype="FLOAT")
        key = Path(var).stem if var not in ("stock", "loop-0") else var
        mean = {k: float(np.mean([r[k] for r in rows])) for k in ("vs_truth", "vs_seed", "level", "sub", "sub_true", "peak")}
        results[key] = {"mean": mean, "rows": rows}
        a.out.write_text(json.dumps(results, indent=1))
        print(f"{key:28s} vs truth {mean['vs_truth']:.4f}  vs seed {mean['vs_seed']:.4f}  level {mean['level']:.2f}x  "
              f"sub {mean['sub']:.2f} (real {mean['sub_true']:.2f})  peak {mean['peak']:.2f}", flush=True)


if __name__ == "__main__":
    main()
