#!/bin/bash
# Overnight: download -> unzip -> train loop-0-evancloud -> pick the best kept
# checkpoint on held-out tracks, scored on audio -> demos -> upload adapter.
set -u
cd /home/stem-user/loop
NAME=loop-0-evancloud; OUT=runs/$NAME; EC=/home/stem-user/evancloud
log(){ echo "=== $(date -Is) $*"; }

while pgrep -f "rclone cop[y]" >/dev/null; do sleep 30; done
[ -s "$EC/EvanCloud.zip" ] || { log "zip missing, stopping"; exit 1; }
log "download done: $(du -h $EC/EvanCloud.zip | cut -f1)"

mkdir -p "$EC/raw" "$EC/tracks"
unzip -q -o "$EC/EvanCloud.zip" -d "$EC/raw" && log "unzipped"
# normalise everything to flac so one reader handles it all
find "$EC/raw" -type f ! -name "._*" | while read -r f; do
  ext="${f##*.}"; ext="${ext,,}"
  case "$ext" in wav|flac|mp3|aif|aiff|m4a|aac|ogg|opus|alac|wma)
    rel="${f#$EC/raw/}"; dst="$EC/tracks/${rel%.*}.flac"; mkdir -p "$(dirname "$dst")"
    [ -s "$dst" ] || { ffmpeg -nostdin -loglevel error -y -i "$f" -vn -map_metadata -1 -ac 2 -ar 44100 -c:a flac -f flac "$dst.part" && mv "$dst.part" "$dst"; } ;;
  esac
done
# drop anything unreadable or under 30 s, so one bad file cannot kill the run hours in
find "$EC/tracks" -name "*.part" -delete
.venv/bin/python - "$EC/tracks" <<'PYV'
import sys, glob, os, soundfile as sf
bad = 0
for f in glob.glob(sys.argv[1] + "/**/*.flac", recursive=True):
    try:
        i = sf.info(f); ok = i.duration >= 30 and i.channels == 2
    except Exception:
        ok = False
    if not ok:
        os.remove(f); bad += 1
print(f"removed {bad} unreadable or too-short files")
PYV
N=$(find "$EC/tracks" -name "*.flac" | wc -l)
SECS=$(.venv/bin/python -c "
import soundfile as sf, glob
print(int(sum(sf.info(f).duration for f in glob.glob('$EC/tracks/**/*.flac', recursive=True))))")
log "$N tracks, $((SECS/3600)) h $(((SECS%3600)/60)) min"
[ "$N" -ge 20 ] || { log "only $N usable tracks, stopping"; exit 1; }

# ~25 passes over the training audio. One optimiser step sees about 4 x 111 s.
HOLD=4   # the four validation tracks named below
STEPS=$(.venv/bin/python -c "print(max(1500, min(6000, round(25 * $SECS / 444 / 100) * 100)))")
SAVE=$(( STEPS / 4 ))
log "training $STEPS steps, saving every $SAVE, holding out $HOLD tracks"
mkdir -p "$OUT"
.venv/bin/python -u lora.py --tracks "$EC/tracks" --name "$NAME" --steps "$STEPS" \
   --save-every "$SAVE" --seed 0 \
   --holdout-match "15476 - " "2876 - " "2672 - " "2598 - " > "$OUT.log" 2>&1
RC=$?; log "training exit $RC"
[ "$RC" -eq 0 ] && [ -s "$OUT/lora.safetensors" ] || { log "training failed, stopping before eval and upload"; exit 1; }

# Score every kept checkpoint on the held-out tracks, on audio.
if [ "$HOLD" -gt 0 ]; then
  mkdir -p "$EC/holdout"; rm -f "$EC/holdout"/*
  while read -r f; do [ -n "$f" ] && ln -sf "$f" "$EC/holdout/"; done < "$OUT/holdout.txt"
  .venv/bin/python -u eval_songs.py --songs "$EC/holdout" --resume runs/base/dit.safetensors \
     --out "$OUT/eval_loop0" --stock > "$OUT/eval_loop0.log" 2>&1
  for ck in "$OUT"/lora_0*.safetensors; do
    s=$(basename "$ck" .safetensors | sed 's/lora_0*//')
    .venv/bin/python lora_fold.py "$ck" --out "$OUT/tmp_delta.safetensors"
    .venv/bin/python -u eval_songs.py --songs "$EC/holdout" --resume "$OUT/tmp_delta.safetensors" \
       --out "$OUT/eval_$s" > "$OUT/eval_$s.log" 2>&1
    rm -f "$OUT/tmp_delta.safetensors"
    log "scored step $s: $(grep -E '^step' "$OUT/eval_$s.log" | tail -1)"
  done
  BEST=$(for f in "$OUT"/eval_[0-9]*.log; do
           v=$(grep -E '^step' "$f" | tail -1 | awk '{print $2}'); echo "$v $f"; done | sort -n | head -1 | awk '{print $2}')
  BS=$(basename "$BEST" .log | sed 's/eval_//')
else
  BS=$STEPS
fi
log "best checkpoint: step $BS"
cp "$OUT/lora_$(printf %06d $BS).safetensors" "$OUT/$NAME.safetensors"
.venv/bin/python lora_fold.py "$OUT/$NAME.safetensors" --out "$OUT/delta.safetensors"
log "final adapter $OUT/$NAME.safetensors, folded delta $OUT/delta.safetensors"

# Upload the adapter (small) and its card. The 2.9 GB delta is derivable.
.venv/bin/python -u evancloud_upload.py "$OUT" "$BS" > "$OUT/upload.log" 2>&1
log "upload exit $?"
log "overnight complete"
