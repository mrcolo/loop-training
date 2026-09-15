"""Load a checkpoint's tensors straight from the Hub into memory.

The post-trained weights are needed only to answer one question -- whether a
base-trained delta survives being added to them -- and there is no room on disk
for a second 2.9 GB transformer. Tensors are read in file order over a single
ranged request, converted to bfloat16 as they arrive, and handed back as a state
dict. Nothing is written.
"""

from __future__ import annotations

import json
import struct

import requests
import torch

CHUNK = 1 << 20


def _get(url: str, token: str, start: int | None = None, end: int | None = None):
    h = {"Authorization": f"Bearer {token}"}
    if start is not None:
        h["Range"] = f"bytes={start}-" + ("" if end is None else str(end))
    r = requests.get(url, headers=h, stream=True, timeout=120)
    r.raise_for_status()
    return r


def _read_exact(stream, n: int) -> bytes:
    parts, got = [], 0
    while got < n:
        c = stream.read(min(CHUNK, n - got))
        if not c:
            raise IOError(f"stream ended {n - got} bytes early")
        parts.append(c)
        got += len(c)
    return b"".join(parts)


def stream_state_dict(url: str, token: str, prefix: str = "", device="cpu",
                      dtype=torch.bfloat16, progress=None):
    """State dict of every tensor whose name starts with `prefix`.

    Reading stops once the last wanted tensor has arrived, so fetching only the
    transformer out of a file that also holds an autoencoder costs the bytes up
    to the transformer's end rather than the whole file.
    """
    n = struct.unpack("<Q", _get(url, token, 0, 7).content)[0]
    header = json.loads(_get(url, token, 8, 7 + n).content)
    header.pop("__metadata__", None)
    data = 8 + n

    order = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
    wanted = [i for i, (k, _) in enumerate(order) if k.startswith(prefix)]
    if not wanted:
        raise ValueError(f"no tensors under prefix {prefix!r}")
    first, last = wanted[0], wanted[-1]

    out, done = {}, 0
    total = sum(order[i][1]["data_offsets"][1] - order[i][1]["data_offsets"][0] for i in wanted)
    stream = _get(url, token, data + order[first][1]["data_offsets"][0]).raw
    stream.decode_content = True
    for i in range(first, last + 1):
        name, spec = order[i]
        a, b = spec["data_offsets"]
        buf = _read_exact(stream, b - a)
        if not name.startswith(prefix) or not buf:
            continue
        src = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}[spec["dtype"]]
        t = torch.frombuffer(bytearray(buf), dtype=src).reshape(spec["shape"])
        out[name] = t.to(device=device, dtype=dtype).clone()
        done += b - a
        if progress and done % (1 << 28) < CHUNK:
            progress(done / total)
    return out
