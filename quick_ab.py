"""Render held-out tracks with loop-0 and an adapter delta, same noise, bf16 transformer
so it fits beside a running trainer."""
import sys, numpy as np, soundfile as sf, torch
from pathlib import Path
from safetensors.torch import load_file
from dataset import Excerpts
from eval_songs import envelope, load_arc
from stable_audio_3.inference.sampling import build_schedule, sample_flow_pingpong
from train import build_number_conditioner, load

dev = torch.device("cuda"); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
cfg, model = load(Path("models/stable-audio-3-medium"), "cpu", conditioner=False)
model.to(torch.bfloat16).to(dev); model.eval()   # cast on the CPU, so the GPU never holds fp32
sr = cfg["sample_rate"]
arc = {k: v.to(torch.bfloat16) for k, v in load_arc(Path("models/stable-audio-3-medium/dit_arc.safetensors"), "", "cpu").items()}
variants = {"loop0": load_file("runs/base/dit.safetensors"), "evancloud1500": load_file(sys.argv[1])}
number = build_number_conditioner(Path("models/stable-audio-3-medium"), dev)
prompt = tuple(t.to(dev)[:1] for t in torch.load("runs/base/prompt_cond.pt"))
cond = {"prompt": prompt, "seconds_total": number([{"seconds_total": 190.0}], dev)["seconds_total"]}
model.diffusion_objective = "rf_denoiser"
songs = [Path(l.strip()) for l in open("runs/loop-0-evancloud/holdout.txt") if l.strip()]
for tag, d in variants.items():
    model.model.load_state_dict({k: (v.float() + d[k].float()).to(torch.bfloat16) for k, v in arc.items()})
    del d
    for q in songs:
        name = q.stem.split(" - ", 1)[1][:40]
        with torch.no_grad():
            audio = Excerpts([q], 190.0, sr).at(0.0)[0][None].to(dev)
            with torch.autocast("cuda", torch.bfloat16):
                z = model.pretransform.encode(audio.to(torch.bfloat16)).float()
            mask = torch.zeros_like(z[:, :1]); mask[..., :round(30 * z.shape[-1] / 190)] = 1
            c = dict(cond); c["inpaint_mask"], c["inpaint_masked_input"] = [mask], [z * mask]
            sig = build_schedule(8, dist_shift=model.sampling_dist_shift, effective_seq_len=z.shape[-1], device=dev)
            g = torch.Generator(device=dev).manual_seed(0); torch.manual_seed(0)
            with torch.autocast("cuda", torch.bfloat16):
                x = sample_flow_pingpong(model, torch.randn(z.shape, generator=g, device=dev), sig,
                                         disable_tqdm=True, cond=c, cfg_scale=1.0)
                wav = model.pretransform.decode(x.to(torch.bfloat16)).float()[0].cpu()
        sf.write(out / f"{name}_{tag}.mp3", wav.clamp(-1, 1).T.numpy(), sr, format="MP3", subtype="MPEG_LAYER_III")
        seed = audio[0].mean(0).float().cpu().numpy()[:30 * sr]; gen = wav.mean(0).numpy()[30 * sr:]
        print(f"{tag:14s} {name:42s} vs seed {np.abs(envelope(gen, sr) - envelope(seed, sr)).mean():.4f}  "
              f"level {np.sqrt((gen**2).mean()) / np.sqrt((seed**2).mean()):.2f}x", flush=True)
        torch.cuda.empty_cache()
