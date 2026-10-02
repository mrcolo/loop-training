# Upload status — fcolooo/loop-0

## 2026-09-15: finetune delta (step 12000) uploaded — DONE

All nine shards plus `delta.safetensors.index.json` are in `fcolooo/loop-0`.
Verified, not just listed: every shard's sha256 was recomputed locally and
matches the LFS sha256 the Hub reports; the index matches on size.
`README.md` and `stems-loop.pdf` were left untouched.

### What worked

Per-shard `huggingface_hub.upload_file` with `HF_HUB_DISABLE_XET=1`, one file at
a time, 10 attempts each, skipping anything already in `list_repo_files`. This
was the job already running from the other session
(`scratchpad/push_shards.py`, log at `hf_shards.log`); it needed no changes and
was left alone to finish.

- Wall clock: 19:21 → 22:38, **3h17m** for 2.91 GB.
- Effective **246 kB/s** end to end including retries; individual shards ran
  290–413 kB/s.
- 12 upload attempts for 10 files. Shards 3, 5 and 7 each failed once mid-file
  and succeeded on attempt 2; the other seven went first try.

### What to know about the failure modes

Two different layers of failure, and only one of them is cheap:

1. **Part-level.** Within a shard, `hf_hub`'s multipart uploader retries the
   individual failed S3 part. Shard 2 hit an `SSLEOFError` plus three
   `NameResolutionError`s on `hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com`
   and still completed on attempt 1. These cost seconds.
2. **File-level.** Some drops kill the whole `upload_file` call, and it restarts
   the shard from zero — shard 5 died at 208/319 MB and re-uploaded from the
   start. This is why shard size matters: ~320 MB is about 15–18 min, so a late
   failure costs that much. Anything materially larger is a bad bet on this
   link.

The DNS failures are worth flagging on their own — `Temporary failure in name
resolution` means the resolver, not just bandwidth, is flaky here. Retry loops
need to treat DNS errors as transient rather than fatal.

### What did not work (from the earlier sessions, not re-tried)

- `upload_file` on the single 2.9 GB file: connection reset during Xet's auth
  token refresh, then a `BadRequestError`. Three attempts.
- `git push` with git-lfs on the single file: i/o timeout to S3 at 265 MB.
- zstd: only 7% on this data, so compression is not a route.

Not needed in the end, but staged and ready if a future upload stalls:
`preupload_lfs_files` + `create_commit` per shard (separates byte transfer from
the commit). `hf_transfer` was never installed — deliberately, to avoid touching
the venv the live training run is using.

### For the other session

- `push_shards.py` (PID 1897533) exited on its own after printing `final:` with
  all files listed. Nothing is still uploading. Nothing needs rerunning.
- The training run was not touched: no GPU use, no processes killed, no files
  deleted outside scratch.
- The staged shards are still on disk at
  `/tmp/claude-1001/-home-stem-user-loop/d54a759c-.../scratchpad/shards/`
  (2.9 GB). Safe to delete now that the upload is verified — left in place
  because they belong to the other session.

---

## 2026-09-16: finetune delta (step 20000) uploaded under `step20000/` — DONE

All nine shards plus `delta.safetensors.index.json` are in `fcolooo/loop-0`
under the `step20000/` prefix. Every shard's sha256 was recomputed locally and
matches the LFS sha256 the Hub reports; the index matches on size.

The step-12000 files were **not** touched: nine `delta-*.safetensors` plus
`delta.safetensors.index.json` remain at the repo root, alongside `README.md`
and `stems-loop.pdf`. The two checkpoints now sit side by side.

### What worked

Same method as step 12000 — per-shard `upload_file` with `HF_HUB_DISABLE_XET=1`,
retry per shard, skip anything already present — with `path_in_repo` set to
`step20000/<name>`. Script at `scratchpad/push_shards20k.py` (session 3314808e),
log at `hf_shards20k.log`.

- Wall clock: 09:52 → 12:32, **2h40m** for 2.91 GB.
- Effective **303 kB/s** end to end including retries.
- 14 upload attempts for 10 files: 10 succeeded, 4 failed mid-transfer.
  Shards 1, 2, 3, 6, 7, 8 and the index went first try; shard 4 needed 2 real
  attempts, shard 9 needed 2, shard 5 needed 3.
- Logged errors: 4 `SSLEOFError`, 3 `NameResolutionError`.

### The link behaved differently this time

Per-shard rates ranged **295 kB/s to 1907 kB/s** — a 6x spread within one run.
Shards 2 and 3 went at ~1.9 MB/s (175s and 172s each), then it fell back to
300-600 kB/s for the rest. Worth knowing that the ~400 kB/s ceiling assumed in
the first upload is not a hard cap; the link is sometimes much faster, so a
stalled-looking transfer is not necessarily the best this connection can do.

When shard 2 came in 6x faster than expected I checked whether the bytes had
actually moved rather than being deduplicated server-side: the stored sha256
matched the local step-20000 file, and no step-20000 shard shares a hash with
any step-12000 file. The speed was real.

Also confirmed before starting: the step-20000 shards are byte-for-byte
*different* from the step-12000 ones despite having identical file sizes.
Identical sizes are just the architecture — shard boundaries come from tensor
shapes, not values. Worth re-checking on any future re-stage, since a silently
duplicated staging directory would look exactly like a correct one from `ls`.

### Known flaw in push_shards20k.py

A failed `list_repo_files` call hits `continue`, which advances the retry
counter — so a listing failure silently spends one of a shard's 15 attempts.
Cosmetically this inflates the "attempt N" numbers in the log (shard 4 shows
"attempt 3" having made only 2 real upload attempts). Harmless at a budget of
15, but fix the loop to not consume an attempt on listing errors if this is
reused.

### For the other session

- `push_shards20k.py` (PID 1955632) exited on its own after printing `final:`.
  Nothing is still uploading.
- The GPU was not touched at any point — this was network and disk only. The
  eval/training process holding ~20 GB was left alone.
- Staged shards remain at `scratchpad/shards20k/` (2.9 GB). Safe to delete now
  that the upload is verified; left in place because they belong to the other
  session.
