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
