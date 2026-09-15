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


def gated_starts(index, seconds: float, min_rms: float, min_sub: float):
    """Window start offsets, in seconds, that clear the energy thresholds."""
    d = np.load(index)
    hop = float(d["hop"])
    box = np.ones(max(1, round(seconds / hop)))
    box /= box.size
    keep = ((np.convolve(d["rms"], box, "valid") >= min_rms)
            & (np.convolve(d["sub"], box, "valid") >= min_sub))
    starts = np.flatnonzero(keep) * hop
    if not len(starts):
        raise ValueError("energy gate rejected the entire source; lower the thresholds")
    return starts, hop


class LatentExcerpts(torch.utils.data.Dataset):
    """Windows of a latent stream precomputed by `encode_latents.py`.

    The autoencoder is 89% of a training step, so a long run is far better spent
    training from latents written once. Crops still land on arbitrary latent
    frames, so the augmentation given up is only waveform-domain (gain, channel
    swap); mask shape carries the rest.
    """

    def __init__(self, path, seconds=190.0, sample_rate: int = 44100,
                 ratio: int = 4096, prompt: str = "", epoch: int = 1000,
                 index=None, min_rms: float = 0.0, min_sub: float = 0.0):
        # `seconds` may be one length or several. The model supports 256-4096 latent
        # frames and conditions on the duration, so training at a single length
        # freezes that conditioning and narrows the model to that one size.
        self.choices = [float(seconds)] if np.isscalar(seconds) else [float(x) for x in seconds]
        self.z = np.load(path, mmap_mode="r")
        self.fps = sample_rate / ratio
        seconds = max(self.choices)
        self.frames = round(seconds * self.fps)
        self.seconds, self.prompt, self.epoch = seconds, prompt, epoch
        self.starts = None
        if index is not None and (min_rms > 0 or min_sub > 0):
            secs, _ = gated_starts(index, seconds, min_rms, min_sub)
            self.starts = (secs * self.fps).astype(np.int64)
            self.starts = self.starts[self.starts + self.frames <= self.z.shape[1]]
        self.hours = self.z.shape[1] / self.fps / 3600

    def __len__(self) -> int:
        return self.epoch

    def _window(self, frame: int, seconds: float | None = None):
        seconds = seconds if seconds is not None else random.choice(self.choices)
        frames = round(seconds * self.fps)
        frame = int(min(max(frame, 0), self.z.shape[1] - frames))
        z = torch.from_numpy(np.asarray(self.z[:, frame:frame + frames])).float()
        return z, {"prompt": self.prompt, "seconds_total": seconds}

    def __getitem__(self, _):
        if self.starts is not None:
            return self._window(random.choice(self.starts))
        return self._window(random.randint(0, self.z.shape[1] - self.frames))

    def at(self, offset_seconds: float, track: int = 0, seconds: float | None = None):
        return self._window(round(offset_seconds * self.fps), seconds or max(self.choices))

    def __repr__(self) -> str:
        gate = "" if self.starts is None else f", gated to {len(self.starts)} windows"
        lens = "/".join(f"{s:.0f}" for s in self.choices)
        return f"LatentExcerpts({self.hours:.1f} h of latents, lengths {lens}s{gate})"


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
            self.starts, hop = gated_starts(index, seconds, min_rms, min_sub)
            self.kept_hours = len(self.starts) * hop / 3600  # starts are hop-spaced

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
