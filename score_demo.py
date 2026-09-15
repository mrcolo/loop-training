#!/usr/bin/env python
"""Score logged demos against the true continuation.

The demos and the ground truth are both written to the TensorBoard event file as
audio, so a run can be scored after the fact with no GPU and no re-rendering.
Every demo uses the same reference excerpt, the same context length and the same
noise seed, so the only thing varying across steps is the model.

Two numbers per demo, both over the generated region only:

  envelope   mean absolute difference between the demo's log-mel envelope and
             the truth's, in log units. Lower is closer to the real continuation.
  level      generated RMS over the true continuation's RMS. This is the number
             that collapsed when the objective was wrong, so it is reported
             separately rather than folded into the first.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import soundfile as sf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load(tb: Path):
    ea = EventAccumulator(str(tb), size_guidance={"audio": 1000})
    ea.Reload()
    out = {}
    for tag in ea.Tags()["audio"]:
        for a in ea.Audio(tag):
            wav, sr = sf.read(io.BytesIO(a.encoded_audio_string), always_2d=True)
            out[(tag, a.step)] = (wav.mean(1), sr)
    return out


def mel_envelope(x: np.ndarray, sr: int, bands: int = 32, n: int = 2048):
    """Log energy in `bands` mel-spaced bins, averaged over time."""
    frames = x[: len(x) // (n // 2) * (n // 2)]
    frames = np.lib.stride_tricks.sliding_window_view(frames, n)[:: n // 2]
    S = np.abs(np.fft.rfft(frames * np.hanning(n), axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    mel = lambda h: 2595 * np.log10(1 + h / 700)
    edges = np.linspace(mel(20), mel(sr / 2), bands + 1)
    hz = 700 * (10 ** (edges / 2595) - 1)
    band = np.stack([S[:, (f >= hz[i]) & (f < hz[i + 1])].sum(1) for i in range(bands)], 1)
    return np.log10(band.mean(0) + 1e-10)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tb", type=Path, default=Path("runs/base/tb"))
    p.add_argument("--context", type=float, default=30.0, help="seconds of real seed in each demo")
    a = p.parse_args()

    clips = load(a.tb)
    truth = next((v for (tag, _), v in clips.items() if tag == "truth"), None)
    if truth is None:
        raise SystemExit(f"no `truth` audio in {a.tb}; it is logged once, at step 1")
    t_wav, sr = truth
    cut = round(a.context * sr)
    t_gen = t_wav[cut:]
    t_env = mel_envelope(t_gen, sr)
    t_rms = np.sqrt((t_gen ** 2).mean())

    steps = sorted(s for (tag, s) in clips if tag == "outpaint")
    print(f"reference: {len(t_wav) / sr:.0f} s, {a.context:.0f} s context, "
          f"true continuation RMS {t_rms:.4f}\n")
    print(f"{'step':>7}  {'envelope':>9}  {'level':>7}  {'centroid':>9}")
    for s in steps:
        wav, _ = clips[("outpaint", s)]
        gen = wav[cut:]
        env = mel_envelope(gen, sr)
        n = 2048
        fr = gen[: len(gen) // (n // 2) * (n // 2)]
        fr = np.lib.stride_tricks.sliding_window_view(fr, n)[:: n // 2]
        S = np.abs(np.fft.rfft(fr * np.hanning(n), axis=1))
        f = np.fft.rfftfreq(n, 1 / sr)
        cen = ((S * f).sum(1) / (S.sum(1) + 1e-9)).mean()
        rms = np.sqrt((gen ** 2).mean())
        label = "  (untrained base)" if s <= 1 else ""
        print(f"{s:7d}  {np.abs(env - t_env).mean():9.4f}  "
              f"{rms / t_rms:6.2f}x  {cen:8.0f} Hz{label}")


if __name__ == "__main__":
    main()
