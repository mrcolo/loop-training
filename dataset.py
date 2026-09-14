"""Random excerpts from audio files, decoded on demand.

Nothing is pre-decoded and nothing is cached: each item seeks straight to a
random offset and reads only the frames it needs. A 26-hour source file
therefore costs no memory, which is what lets the autoencoder stay online.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio


class Excerpts(torch.utils.data.Dataset):
    """Fixed-length stereo excerpts drawn uniformly from a set of tracks.

    Args:
        paths: audio files to draw from.
        seconds: excerpt length; shorter tracks are zero-padded.
        sample_rate: rate the model expects; files are resampled per item.
        prompt: text condition attached to every excerpt.
        epoch: excerpts per pass. This is a stream rather than a finite set, so
            the length is simply how often you want the training loop to wrap.
        index: optional .npz from `scan_energy`, holding per-hop rms and sub-bass
            fraction for the first track. A heterogeneous source (a long set, a
            radio rip) contains quiet passages that drag the model toward a quiet
            average; gating on these keeps training on the material you want.
        min_rms, min_sub: thresholds an excerpt's mean must clear to be sampled.
    """

    def __init__(self, paths, seconds: float = 24.0, sample_rate: int = 44100,
                 prompt: str = "", epoch: int = 1000, index=None,
                 min_rms: float = 0.0, min_sub: float = 0.0):
        self.tracks = []
        for path in map(Path, paths):
            with sf.SoundFile(path) as f:
                self.tracks.append((path, f.samplerate, f.frames))
        if not self.tracks:
            raise ValueError("no audio files given")
        self.weights = [frames / sr for _, sr, frames in self.tracks]
        self.seconds, self.sample_rate, self.prompt, self.epoch = seconds, sample_rate, prompt, epoch
        self.frames = round(seconds * sample_rate)

        self.starts = None
        if index is not None and (min_rms > 0 or min_sub > 0):
            d = np.load(index)
            hop, k = float(d["hop"]), max(1, round(seconds / float(d["hop"])))
            box = np.ones(k) / k
            keep = ((np.convolve(d["rms"], box, "valid") >= min_rms)
                    & (np.convolve(d["sub"], box, "valid") >= min_sub))
            self.starts = np.flatnonzero(keep) * hop
            self.kept_hours = len(self.starts) * hop / 3600  # starts are hop-spaced, not window-spaced
            if not len(self.starts):
                raise ValueError("energy gate rejected the entire source; lower the thresholds")

    def __len__(self) -> int:
        return self.epoch

    def __getitem__(self, _):
        if self.starts is not None:  # gated: draw only from qualifying windows
            path, sr, _ = self.tracks[0]
            return self._read(path, sr, round(random.choice(self.starts) * sr))
        path, sr, total = random.choices(self.tracks, weights=self.weights)[0]
        return self._read(path, sr, random.randint(0, max(0, total - round(self.seconds * sr))))

    def at(self, offset_seconds: float, track: int = 0):
        """One specific excerpt, for a demo you want to compare across steps."""
        path, sr, total = self.tracks[track]
        start = min(round(offset_seconds * sr), max(0, total - round(self.seconds * sr)))
        return self._read(path, sr, start)

    def _read(self, path: Path, sr: int, start: int):
        with sf.SoundFile(path) as f:
            f.seek(start)
            block = f.read(round(self.seconds * sr), dtype="float32", always_2d=True)

        audio = torch.from_numpy(block).T
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        audio = audio[:2] if audio.shape[0] >= 2 else audio.repeat(2, 1)
        audio = F.pad(audio[:, : self.frames], (0, max(0, self.frames - audio.shape[1])))
        return audio, {"prompt": self.prompt, "seconds_total": self.seconds}

    def __repr__(self) -> str:
        gate = "" if self.starts is None else \
            f", gated to {self.kept_hours:.1f} h ({self.kept_hours / (sum(self.weights) / 3600):.0%}) of the source"
        return (f"Excerpts({len(self.tracks)} tracks, {sum(self.weights) / 3600:.1f} h total, "
                f"{self.seconds:.1f}s @ {self.sample_rate} Hz{gate})")


def collate(batch):
    """Stack the audio, keep the metadata as the list of dicts the conditioner wants."""
    audio, metadata = zip(*batch)
    return torch.stack(audio), list(metadata)
