#!/usr/bin/env python
"""Index a long recording by loudness and sub-bass content.

A multi-hour DJ set is not uniform: it contains dead air, ambient intros and
transitions alongside the loud material. Sampling excerpts uniformly teaches the
model the average of all of it. This writes a coarse index that `dataset.Excerpts`
uses to skip the parts you do not want to train on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SR, N = 44100, 4096


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", type=Path)
    p.add_argument("--out", type=Path, default=Path("energy_index.npz"))
    p.add_argument("--hop", type=float, default=8.0, help="seconds between probes")
    p.add_argument("--win", type=float, default=2.0, help="seconds per probe")
    a = p.parse_args()

    with sf.SoundFile(a.audio) as f:
        total, sr = f.frames, f.samplerate
    steps = int((total / sr - a.win) // a.hop)
    rms, sub = np.zeros(steps, np.float32), np.zeros(steps, np.float32)
    freqs = np.fft.rfftfreq(N, 1 / sr)
    low, window = freqs < 120, np.hanning(N)

    with sf.SoundFile(a.audio) as f:
        for i in range(steps):
            f.seek(int(i * a.hop * sr))
            x = f.read(int(a.win * sr), dtype="float32", always_2d=True).mean(1)
            rms[i] = np.sqrt((x**2).mean())
            spectrum = np.abs(np.fft.rfft(x[:N] * window)) + 1e-9
            sub[i] = spectrum[low].sum() / spectrum.sum()

    np.savez(a.out, rms=rms, sub=sub, hop=a.hop, win=a.win)
    print(f"{a.out}: {steps} probes over {total / sr / 3600:.1f} h")
    for name, v in (("rms", rms), ("sub", sub)):
        print(f"  {name}: p10 {np.percentile(v, 10):.3f}  p50 {np.percentile(v, 50):.3f}  "
              f"p90 {np.percentile(v, 90):.3f}")


if __name__ == "__main__":
    main()
