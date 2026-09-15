#!/usr/bin/env python
"""Turn a finetune checkpoint into the model you actually run.

Training happens against the base transformer, because that is the checkpoint
the squared-error objective is valid for. Inference should happen on the
post-trained transformer, because that is the one that samples in eight steps.
The finetune moves between them as a float16 delta, which is the format the
checkpoints are already written in.

This streams the post-trained transformer from the Hub, adds the delta, and
writes one bfloat16 file alongside the autoencoder, ready for `sample.py`.
Nothing intermediate is written, because there is rarely room for it.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from remote_weights import stream_state_dict

ARC = "https://huggingface.co/stabilityai/stable-audio-3-medium/resolve/main/model.safetensors"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", type=Path, default=Path("runs/base/dit.safetensors"))
    p.add_argument("--out", type=Path, default=Path("models/stable-audio-3-medium/dit_shipped.safetensors"))
    p.add_argument("--token", default=str(Path.home() / ".hf_token"))
    p.add_argument("--alpha", type=float, default=1.0,
                   help="scale on the delta; 1.0 is the finetune, 0.0 the released model")
    p.add_argument("--drop-base", type=Path,
                   help="delete this file first, e.g. the base transformer once training "
                        "is finished. Freeing 2.9 GB is often the only way this fits.")
    a = p.parse_args()

    if a.drop_base and a.drop_base.exists():
        print(f"removing {a.drop_base} ({a.drop_base.stat().st_size / 1e9:.1f} GB)")
        a.drop_base.unlink()

    free = shutil.disk_usage(a.out.parent).free / 1e9
    if free < 3.2:
        raise SystemExit(f"{free:.1f} GB free at {a.out.parent}; this needs about 3.2. "
                         f"Pass --drop-base models/stable-audio-3-medium-base/dit_base.safetensors "
                         f"if training is done, or free space elsewhere.")

    delta = load_file(a.resume)
    with safe_open(str(a.resume), framework="pt") as f:
        step = f.metadata()["step"]
    print(f"delta from step {step}, {len(delta)} tensors, alpha {a.alpha}")

    token = Path(a.token).read_text().strip()
    print("streaming the post-trained transformer from the Hub")
    arc = stream_state_dict(ARC, token, prefix="model.model.", device="cpu",
                            progress=lambda f: print(f"  {f:5.1%}", flush=True))

    # The file namespaces under `model.model.`; checkpoints keep one `model.` level.
    merged, missing = {}, []
    for k, v in arc.items():
        dk = k[len("model."):]
        if dk not in delta:
            missing.append(dk)
            continue
        merged[k] = (v.float() + a.alpha * delta[dk].float()).to(torch.bfloat16)
    if missing:
        raise SystemExit(f"{len(missing)} tensors have no delta, first: {missing[:3]}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(a.out), metadata={"step": step, "alpha": str(a.alpha),
                                            "base": "stable-audio-3-medium"})
    print(f"\nwrote {a.out} ({a.out.stat().st_size / 1e9:.2f} GB)\n")
    print("sample with the post-trained objective and its eight-step sampler:\n")
    print(f"  python sample.py --audio YOUR.flac --dit {a.out} \\\n"
          f"      --objective rf_denoiser --steps 8 --cfg 1.0 --context 30 --seconds 190")


if __name__ == "__main__":
    main()
