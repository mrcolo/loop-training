#!/usr/bin/env python
"""Fetch Stable Audio 3 Medium, converting it to bfloat16 as it streams.

The published checkpoint is 9.2 GB of float32. Training keeps its master copy in
float32 in memory regardless, so storing bfloat16 halves the disk cost for the
price of one rounding at load. Tensors are converted in flight, so the float32
file never lands on disk, and an interrupted run resumes at the last whole
tensor it wrote.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import requests
import torch

REPO = "https://huggingface.co/stabilityai/stable-audio-3-medium/resolve/main"
CHUNK = 1 << 20


def read_exact(stream, n: int) -> bytes:
    """urllib3 hands back short reads; a tensor has to arrive whole."""
    parts, got = [], 0
    while got < n:
        chunk = stream.read(min(CHUNK, n - got))
        if not chunk:
            raise IOError(f"stream ended {n - got} bytes early")
        parts.append(chunk)
        got += len(chunk)
    return b"".join(parts)


def get(url: str, token: str, start: int | None = None, end: int | None = None):
    headers = {"Authorization": f"Bearer {token}"}
    if start is not None:
        headers["Range"] = f"bytes={start}-" + ("" if end is None else str(end))
    r = requests.get(url, headers=headers, stream=True, timeout=60)
    r.raise_for_status()
    return r


def read_header(url: str, token: str):
    n = struct.unpack("<Q", get(url, token, 0, 7).content)[0]
    header = json.loads(get(url, token, 8, 7 + n).content)
    header.pop("__metadata__", None)
    return 8 + n, header


def plan(header: dict):
    """Order tensors by position in the source and lay out the bfloat16 copy."""
    order = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
    out, cursor, tensors = {}, 0, []
    for name, spec in order:
        src_a, src_b = spec["data_offsets"]
        cast = spec["dtype"] == "F32"
        size = (src_b - src_a) // 2 if cast else src_b - src_a
        out[name] = {"dtype": "BF16" if cast else spec["dtype"], "shape": spec["shape"],
                     "data_offsets": [cursor, cursor + size]}
        tensors.append((name, src_a, src_b, cursor, cursor + size, cast))
        cursor += size
    blob = json.dumps(out).encode()
    blob += b" " * (-len(blob) % 8)
    return struct.pack("<Q", len(blob)) + blob, tensors, cursor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("models/stable-audio-3-medium"))
    p.add_argument("--token", default=str(Path.home() / ".hf_token"))
    a = p.parse_args()
    token = Path(a.token).read_text().strip()
    a.out.mkdir(parents=True, exist_ok=True)

    for name in ["model_config.json", "t5gemma-b-b-ul2/config.json", "t5gemma-b-b-ul2/tokenizer.json",
                 "t5gemma-b-b-ul2/tokenizer.model", "t5gemma-b-b-ul2/tokenizer_config.json",
                 "t5gemma-b-b-ul2/special_tokens_map.json", "t5gemma-b-b-ul2/generation_config.json",
                 "t5gemma-b-b-ul2/model.safetensors"]:
        dst = a.out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        want = int(requests.head(f"{REPO}/{name}", headers={"Authorization": f"Bearer {token}"},
                                 allow_redirects=True, timeout=60).headers["content-length"])
        for _ in range(200):
            have = dst.stat().st_size if dst.exists() else 0
            if have >= want:
                break
            try:
                with get(f"{REPO}/{name}", token, have) as r, open(dst, "ab") as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
            except requests.RequestException as e:
                print(f"  retrying {name}: {e}", flush=True)
        print(f"{name}: {dst.stat().st_size / 1e6:.1f} MB", flush=True)

    url, dst = f"{REPO}/model.safetensors", a.out / "model.safetensors"
    src_data, header = read_header(url, token)
    out_header, tensors, total = plan(header)
    print(f"{len(tensors)} tensors, {total / 1e9:.2f} GB bfloat16 "
          f"(from {tensors[-1][2] / 1e9:.2f} GB float32)", flush=True)

    done = max(0, dst.stat().st_size - len(out_header)) if dst.exists() else 0
    first = next((i for i, t in enumerate(tensors) if t[4] > done), len(tensors))
    if first == len(tensors):
        print("already complete", flush=True)
        return
    if not dst.exists():
        dst.write_bytes(out_header)
    with open(dst, "r+b") as f:
        f.truncate(len(out_header) + tensors[first][3])
        f.seek(0, 2)
        stream = get(url, token, src_data + tensors[first][1]).raw
        stream.decode_content = True
        for name, src_a, src_b, _, _, cast in tensors[first:]:
            buf = read_exact(stream, src_b - src_a)
            if cast:
                buf = torch.frombuffer(bytearray(buf), dtype=torch.float32).to(
                    torch.bfloat16).view(torch.int16).numpy().tobytes()
            f.write(buf)
            if f.tell() % (1 << 28) < CHUNK:
                print(f"  {(f.tell() - len(out_header)) / total:6.1%}", flush=True)
    print(f"wrote {dst} ({dst.stat().st_size / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
