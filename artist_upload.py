#!/usr/bin/env python
"""Upload a loop-0 artist adapter to a private Hugging Face repo: the adapter, its card, the
training/eval code and a few sample renders.

    python artist_upload.py --repo fcolooo/loop-0-summitcloud --card runs/loop-0-summitcloud/README.md \
        --adapter runs/loop-0-summitcloud/loop-0-summitcloud-20261010.safetensors \
        --extra runs/loop-0-summitcloud/split.json runs/loop-0-summitcloud/eval.json \
        --samples runs/loop-0-summitcloud/samples
"""
import argparse
import os
import time
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"          # the Xet path stalls on this connection
from huggingface_hub import HfApi

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--repo", required=True)
p.add_argument("--card", type=Path, required=True)
p.add_argument("--adapter", type=Path, required=True)
p.add_argument("--extra", type=Path, nargs="*", default=[])
p.add_argument("--samples", type=Path, default=None, help="folder of .mp3 renders, uploaded to samples/")
a = p.parse_args()

api = HfApi(token=Path.home().joinpath(".hf_token").read_text().strip())
api.create_repo(a.repo, private=True, exist_ok=True)
files = [(a.card, "README.md"), (a.adapter, a.adapter.name),
         (Path("lora.py"), "lora.py"), (Path("lora_fold.py"), "lora_fold.py"), (Path("lora_eval.py"), "lora_eval.py")]
files += [(f, f.name) for f in a.extra]
if a.samples:
    files += [(f, "samples/" + f.name) for f in sorted(a.samples.glob("*.mp3"))]
for src, dst in files:
    for t in range(10):
        try:
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=a.repo)
            print("ok", dst, flush=True)
            break
        except Exception as e:
            print("retry", dst, type(e).__name__, str(e)[:120], flush=True)
            time.sleep(20)
    else:
        raise SystemExit(f"could not upload {dst}")
info = api.model_info(a.repo, files_metadata=True)
print(f"https://huggingface.co/{a.repo}  ({len(info.siblings)} files, private={info.private})")
