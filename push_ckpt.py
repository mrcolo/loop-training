#!/usr/bin/env python
"""Upload a checkpoint to the private Hugging Face repo.

Checkpoints hold the float16 delta from the base weights, so a file is only
meaningful alongside the base checkpoint it was trained from; the model card
records that.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi
from safetensors import safe_open


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("runs/outpaint/dit.safetensors"))
    p.add_argument("--repo", default="fcolooo/stems-loop")
    p.add_argument("--token", default=str(Path.home() / ".hf_token"))
    p.add_argument("--name", default=None, help="filename in the repo")
    a = p.parse_args()

    with safe_open(a.ckpt, framework="pt") as f:
        step = f.metadata().get("step", "unknown")
    name = a.name or f"checkpoints/dit-delta-step{step}.safetensors"
    api = HfApi(token=Path(a.token).read_text().strip())
    api.upload_file(path_or_fileobj=str(a.ckpt), path_in_repo=name,
                    repo_id=a.repo, repo_type="model",
                    commit_message=f"outpainting finetune delta, step {step}")
    print(f"pushed {a.ckpt} -> {a.repo}/{name} ({a.ckpt.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
