"""Upload the loop-0-evancloud adapter and a card to fcolooo/loop-0-evancloud."""
import os, sys, time, re, glob
os.environ["HF_HUB_DISABLE_XET"] = "1"
from pathlib import Path
from huggingface_hub import HfApi

out, best = Path(sys.argv[1]), sys.argv[2]
api = HfApi(token=Path.home().joinpath(".hf_token").read_text().strip())
repo = "fcolooo/loop-0-evancloud"
api.create_repo(repo, private=True, exist_ok=True)

def row(log):
    try:
        m = [l for l in open(log) if re.match(r"^(step|stock)", l)]
        return {l.split()[0]: l.split()[1] for l in m}
    except FileNotFoundError:
        return {}
base = row(out / "eval_loop0.log")
lines = []
for f in sorted(glob.glob(str(out / "eval_[0-9]*.log")), key=lambda x: int(re.findall(r"(\d+)\.log", x)[0])):
    s = re.findall(r"(\d+)\.log", f)[0]
    v = row(f).get(f"step{s}")
    if v: lines.append(f"| loop-0-evancloud @ {s}{' (shipped)' if s == best else ''} | {v} |")
held = (out / "holdout.txt").read_text().strip().splitlines() if (out / "holdout.txt").exists() else []

card = f"""---
base_model: fcolooo/loop-0
tags: [audio, outpainting, stable-audio, lora]
---
# loop-0-evancloud

A LoRA adapter on top of [loop-0](https://huggingface.co/fcolooo/loop-0), trained on the
EvanCloud tracks. Rank 32 on the attention and feed-forward projections, 36.6 M parameters.
Shipped checkpoint: step {best}, chosen by audio on held-out tracks, not by loss.

## Held-out evaluation

{len(held)} tracks never seen in training; first 30 s as seed, 160 s generated at 8 steps.
Distance between the generated audio's log-mel envelope and the seed's (lower = continues
the track's character more closely).

| model | distance |
| --- | --- |
| stock post-trained | {base.get('stock', '-')} |
| loop-0 | {next((v for k, v in base.items() if k.startswith('step')), '-')} |
{chr(10).join(lines)}

## Use

Everything is a delta from `stable-audio-3-medium-base`, so they add:
post-trained transformer + loop-0 delta + this adapter, sampled in 8 steps.

```bash
python lora_fold.py loop-0-evancloud.safetensors --loop-delta <loop-0 delta> --out delta.safetensors
python eval_songs.py --songs YOUR_TRACKS --resume delta.safetensors --out renders
```

`lora.py` and `lora_fold.py` are included; training code is in
[github.com/mrcolo/loop-training](https://github.com/mrcolo/loop-training).
"""
(out / "README.md").write_text(card)
files = [(out / "README.md", "README.md"), (out / "loop-0-evancloud.safetensors", "loop-0-evancloud.safetensors"),
         (Path("lora.py"), "lora.py"), (Path("lora_fold.py"), "lora_fold.py")]
for d in sorted(glob.glob(str(out / f"eval_{best}" / "*.mp3")))[:6]:
    files.append((Path(d), "samples/" + Path(d).name))
for src, dst in files:
    for t in range(10):
        try:
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=repo); print("ok", dst, flush=True); break
        except Exception as e:
            print("retry", dst, type(e).__name__, flush=True); time.sleep(20)
print("https://huggingface.co/" + repo)
