#!/usr/bin/env python
"""Fold a LoRA checkpoint into a full delta from base: loop-0 delta + (alpha/r) B A.
The result is a drop-in --resume for eval_songs.py, sample.py and ship.py."""
import argparse
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("lora", type=Path)
p.add_argument("--loop-delta", type=Path, default=Path("runs/base/dit.safetensors"))
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
with safe_open(str(a.lora), framework="pt") as f:
    meta = f.metadata()
scale = float(meta["alpha"]) / float(meta["rank"])
lora = load_file(str(a.lora))
full = {k: v.float() for k, v in load_file(str(a.loop_delta)).items()}
n = 0
for k in lora:
    if k.endswith(".lora_A"):
        base = k[: -len(".lora_A")]
        full[base + ".weight"] += (lora[base + ".lora_B"] @ lora[k]) * scale
        n += 1
save_file({k: v.half() for k, v in full.items()}, str(a.out), metadata=meta)
print(f"folded {n} adapters (step {meta['step']}) into {a.out}")
