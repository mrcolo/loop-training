#!/usr/bin/env python
"""How large is the finetune's update next to the post-training offset?

The delta can be added to the post-trained weights only while it stays small
enough not to swamp what post-training installed. Measuring that needs the
post-trained weights, but only one layer of them: a byte-ranged request for a
single transformer block is about 100 MB against 9.2 GB for the file, and the
ratio it gives tracks the whole model closely enough to watch over a run.
"""
from __future__ import annotations

import argparse, json, struct
from pathlib import Path

import numpy as np
import requests
import torch
from safetensors import safe_open
from safetensors.torch import load_file

URL = "https://huggingface.co/stabilityai/stable-audio-3-medium/resolve/main/model.safetensors"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=11)
    p.add_argument("--base", type=Path,
                   default=Path("models/stable-audio-3-medium-base/dit_base.safetensors"))
    p.add_argument("--resume", type=Path, default=Path("runs/base/dit.safetensors"))
    p.add_argument("--token", default=str(Path.home() / ".hf_token"))
    a = p.parse_args()

    tok = Path(a.token).read_text().strip()
    h = {"Authorization": f"Bearer {tok}"}
    rng = lambda x, y: requests.get(URL, headers={**h, "Range": f"bytes={x}-{y}"},
                                    timeout=120).content
    n = struct.unpack("<Q", rng(0, 7))[0]
    hdr = json.loads(rng(8, 7 + n)); hdr.pop("__metadata__", None)
    data = 8 + n

    want = [k for k in hdr if f".layers.{a.layer}." in k
            and k.endswith("weight") and len(hdr[k]["shape"]) == 2]
    delta = load_file(a.resume)
    with safe_open(str(a.resume), framework="pt") as f:
        step = f.metadata()["step"]

    rows = []
    with safe_open(str(a.base), framework="pt") as f:
        for k in want:
            x, y = hdr[k]["data_offsets"]
            arc = torch.frombuffer(bytearray(rng(data + x, data + y - 1)),
                                   dtype=torch.float32).reshape(hdr[k]["shape"])
            w = f.get_tensor(k).float()
            d_arc = ((arc - w).norm() / w.norm()).item()
            d_ft = (delta[k[len("model."):]].float().norm() / w.norm()).item()
            rows.append((k.split(f"layers.{a.layer}.")[1], d_arc, d_ft))

    print(f"step {step}, layer {a.layer}\n")
    print(f"  {'tensor':38s} {'post-training':>13s} {'finetune':>10s} {'ratio':>7s}")
    for name, d_arc, d_ft in rows:
        print(f"  {name:38s} {d_arc:12.3%} {d_ft:9.3%} {d_ft / d_arc:6.2f}x")
    r = np.median([f / a_ for _, a_, f in rows])
    print(f"\n  median ratio {r:.2f}x. The transplant held at 0.85x and at this value;"
          f"\n  re-run probe_transplant.py if it climbs much past ~3x.")


if __name__ == "__main__":
    main()
