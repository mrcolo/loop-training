#!/usr/bin/env python
"""Split a delta checkpoint into upload-sized shards.

A 2.9 GB single file needs an unbroken two-hour upload on this link, and it has
not once survived that. Shards fail independently: a dropped connection costs one
piece, and the next attempt resumes at that piece rather than at zero. The layout
is the standard sharded-safetensors one, so anything that reads Hub checkpoints
reads this.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", type=Path, default=Path("runs/base/dit.safetensors"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard-mb", type=int, default=300)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    with safe_open(str(a.resume), framework="pt") as f:
        step = f.metadata()["step"]
        keys = list(f.keys())
        shards, cur, size = [], {}, 0
        for k in keys:
            t = f.get_tensor(k)
            cur[k] = t
            size += t.numel() * t.element_size()
            if size >= a.shard_mb * 1024 * 1024:
                shards.append(cur); cur, size = {}, 0
        if cur:
            shards.append(cur)

    n = len(shards)
    index = {"metadata": {"total_size": 0, "step": step}, "weight_map": {}}
    for i, sd in enumerate(shards, 1):
        name = f"delta-{i:05d}-of-{n:05d}.safetensors"
        save_file(sd, str(a.out / name), metadata={"step": step})
        index["metadata"]["total_size"] += sum(t.numel() * t.element_size() for t in sd.values())
        for k in sd:
            index["weight_map"][k] = name
        print(f"  {name}  {(a.out / name).stat().st_size/1e6:.0f} MB", flush=True)
    (a.out / "delta.safetensors.index.json").write_text(json.dumps(index, indent=1))
    print(f"{n} shards, step {step}, {index['metadata']['total_size']/1e9:.2f} GB total")


if __name__ == "__main__":
    main()
