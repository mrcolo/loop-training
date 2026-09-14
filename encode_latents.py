#!/usr/bin/env python
"""Encode a long recording to one continuous latent stream.

The autoencoder is 89% of a training step, so encoding once and training from
the result buys roughly seven times the optimiser updates for a fixed time
budget. Blocks overlap and are trimmed so no seam survives into the stream.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from train import load

BLOCK_S, PAD_S = 190.0, 8.0


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", type=Path)
    p.add_argument("--model", type=Path, default=Path("models/stable-audio-3-medium"))
    p.add_argument("--out", type=Path, default=Path("latents.npy"))
    a = p.parse_args()

    dev = torch.device("cuda")
    cfg, model = load(a.model, dev)
    ae = model.pretransform.to(torch.bfloat16).eval().requires_grad_(False)
    del model.model, model.conditioner
    torch.cuda.empty_cache()

    sr, ratio, ch = cfg["sample_rate"], int(ae.downsampling_ratio), ae.io_channels
    with sf.SoundFile(a.audio) as f:
        total, file_sr = f.frames, f.samplerate
    n_out = int(total / file_sr * sr) // ratio
    dim = model.io_channels if hasattr(model, "io_channels") else 256
    out = np.lib.format.open_memmap(a.out, mode="w+", dtype=np.float16, shape=(dim, n_out))
    print(f"{total / file_sr / 3600:.2f} h -> {n_out} latent frames, {out.nbytes / 1e9:.2f} GB", flush=True)

    block, pad = int(BLOCK_S * file_sr), int(PAD_S * file_sr)
    t0, written = time.time(), 0
    with sf.SoundFile(a.audio) as f:
        for start in range(0, total, block):
            lo = max(0, start - pad)
            f.seek(lo)
            want = min(start + block + pad, total) - lo
            x = torch.from_numpy(f.read(want, dtype="float32", always_2d=True)).T
            if file_sr != sr:
                x = torchaudio.functional.resample(x, file_sr, sr)
            x = x[:2] if x.shape[0] >= 2 else x.repeat(2, 1)
            with torch.autocast("cuda", torch.bfloat16):
                z = ae.encode(x[None].to(dev))[0].float().cpu().numpy()
            skip = (start - lo) // ratio
            keep = min(block, total - start) // ratio
            chunk = z[:, skip:skip + keep]
            end = min(written + chunk.shape[1], n_out)
            out[:, written:end] = chunk[:, : end - written].astype(np.float16)
            written = end
            if (start // block) % 20 == 0:
                done = written / n_out
                rate = (time.time() - t0) / max(done, 1e-9)
                print(f"  {done:5.1%}  eta {(rate - (time.time() - t0)) / 60:5.1f} min", flush=True)
    out.flush()
    print(f"wrote {a.out} ({written} frames) in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
