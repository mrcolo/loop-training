# Brief: encoding a large audio corpus into training latents

You have ~80 GB of audio to add to this project's training set. This describes
exactly what to produce, why the format is what it is, and the mistakes that
will silently corrupt the result.

## What you are producing

One float16 `.npy` array per source file, shape `(256, frames)`, plus a manifest
and an energy index. The training script memory-maps these and crops windows from
them; it never touches the original audio again.

The autoencoder is 89% of a training step when run online, so precomputing is
worth roughly **7x more optimiser steps for a fixed time budget**. That is the
whole reason this job exists.

## The numbers

| quantity | value |
| --- | --- |
| latent frame rate | 10.7666 fps (44100 / 4096) |
| channels | 256 |
| storage | float16 |
| size | 5.4 kB per audio-second, **19.8 MB per audio-hour** |
| encode throughput | **23x realtime** on an RTX 3090 (measured: 26.4 h in 69 min) |

80 GB is roughly 200 hours if the source is FLAC, or ~126 hours if it is 16-bit
WAV. Check before you plan.

| corpus | latents on disk | GPU time |
| --- | --- | --- |
| 126 h | 2.5 GB | 5.5 hours |
| 200 h | 4.0 GB | 8.7 hours |
| 300 h | 6.0 GB | 13 hours |

The latents are ~200x smaller than the audio. Encoding 80 GB yields about 4 GB.

## Non-negotiables

These are the ways to produce latents that look fine and train badly.

**Use the same autoencoder the model will train with.** Latents are meaningless
across models. Load the pretransform from `stabilityai/stable-audio-3-medium`
(its `pretransform` config is byte-identical to `-base`, so either works). Do not
substitute SAME-S or SAME-L standalone weights without checking the config matches.

**Keep the autoencoder in `eval()`.** The SoftNorm bottleneck applies noise
regularisation whose scale differs by **50x** between train and eval mode
(1e-3 against 5e-2). Calling `.train()` anywhere on the model, rather than only
on the DiT, injects 50x more noise into every latent you write. This is the single
most damaging mistake available here and it is invisible until sample quality is
bad. `load_diffusion_cond` returns the model in `eval()`; leave it there.

**Resample to exactly 44100 Hz and exactly 2 channels before encoding.** Mono
sources get duplicated, >2 channels get truncated to the first two.

**Encode in overlapping blocks and trim.** The encoder is convolutional plus
transformer, so block boundaries leave seams. Use ~190 s blocks with ~8 s of
padding on each side, encode the padded block, then keep only the middle. The
reference implementation is `encode_latents.py` in this repo.

**Use `pretransform.encode()`, not the raw autoencoder.** It applies the scale
factor (1.0 for this model, but do not hardcode that assumption).

**float16, not bfloat16.** Latent values sit around unit scale with meaningful
fine structure; float16's 10-bit mantissa preserves it, bfloat16's 7-bit does not.
This project already lost a day to a bfloat16 rounding bug in a different file.

## Structure for many files

`encode_latents.py` here handles a single recording and writes one stream. For a
corpus, produce one array per source file plus a manifest:

```json
{
  "sample_rate": 44100,
  "downsampling_ratio": 4096,
  "channels": 256,
  "dtype": "float16",
  "files": [
    {"source": "sets/2024-03-19.flac", "latents": "latents/2024-03-19.npy",
     "frames": 1025162, "seconds": 95217.4, "sha256": "..."}
  ]
}
```

Keep the source path and a hash so a latent file can always be traced back and
re-verified. Do not concatenate everything into one array: a single corrupt or
truncated write then poisons the whole corpus, and per-file arrays let you drop
or re-encode one source without redoing the rest.

`dataset.LatentExcerpts` currently opens one array. Extending it to a manifest
means sampling a file weighted by its frame count, then an offset within it —
about ten lines.

## Energy index

Run `scan_energy.py` per source file as well. It probes RMS and sub-bass fraction
every 8 s and takes 45 seconds per 26 hours, so it is free relative to encoding.
Training uses it to skip dead air; on a 26-hour DJ set, gating at `rms >= 0.15`
kept 95% of the material and dropped the silence between sets. Without it, a
corpus with quiet passages teaches the model a quiet average.

## Verify before you hand it over

Do not trust a job that ran to completion. For a handful of random windows:

1. Decode the latent window back to audio with the same pretransform.
2. Compare against the source audio at the same offset: RMS within a few percent,
   spectral centroid within ~5%.
3. Confirm `frames == floor(seconds * 44100 / 4096)` for every file in the manifest.
4. Check for all-zero regions, which mean a block failed to write.

A round-trip that comes back quiet or dull is the `eval()` mistake above.

## Practical notes

- Encoding is GPU-bound and single-stream; batching short blocks helps little
  because the encoder already processes internally chunked.
- Write with `numpy.lib.format.open_memmap` so the file is preallocated and
  crash-resumable rather than accumulating in RAM.
- Peak VRAM for encoding alone is under 8 GB, so it co-exists with nothing else
  on a 24 GB card but needs no more than that.
- Expect ~1 GB/hour of source if the corpus is FLAC; confirm the real ratio on a
  sample before committing to a disk plan.
