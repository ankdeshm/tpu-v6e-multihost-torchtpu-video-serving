#!/usr/bin/env python3
"""Zero-local-disk, idempotent, multi-threaded HuggingFace -> GCS model weight streamer."""

import argparse
import concurrent.futures
import os
import sys
import time
from google.cloud import storage
from huggingface_hub import HfApi, hf_hub_url
import requests


def transfer_one_file(
    repo_id: str,
    rfilename: str,
    expected_size: int,
    bucket_name: str,
    prefix: str,
    hf_token: str | None,
) -> tuple[str, str, int]:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_path = f"{prefix.rstrip('/')}/{repo_id}/{rfilename}"
    blob = bucket.blob(blob_path)
    if blob.exists():
        blob.reload()
        if expected_size > 0 and blob.size == expected_size:
            return (rfilename, "SKIPPED_EXISTS", blob.size)

    url = hf_hub_url(repo_id=repo_id, filename=rfilename)
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    t0 = time.time()
    written = 0
    for attempt in range(1, 4):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                # 64 MiB resumable upload chunks directly to GCS
                with blob.open("wb", chunk_size=64 * 1024 * 1024) as gcs_f:
                    for chunk in resp.iter_content(chunk_size=16 * 1024 * 1024):
                        if chunk:
                            gcs_f.write(chunk)
                            written += len(chunk)
            dt = max(time.time() - t0, 1e-3)
            mbps = (written / (1024 * 1024)) / dt
            print(f"[stage-gcs] DONE {repo_id}/{rfilename} ({written/1e9:.2f} GB in {dt:.1f}s = {mbps:.1f} MiB/s)", flush=True)
            return (rfilename, "UPLOADED", written)
        except Exception as exc:
            print(f"[stage-gcs] WARN attempt {attempt}/3 failed for {repo_id}/{rfilename}: {exc}", flush=True)
            written = 0
            if attempt == 3:
                raise
            time.sleep(5 * attempt)
    return (rfilename, "FAILED", 0)


def stage_repo(repo_id: str, bucket_name: str, prefix: str, workers: int, hf_token: str | None):
    api = HfApi(token=hf_token)
    info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
    siblings = [s for s in info.siblings if not s.rfilename.startswith(".git")]
    total_bytes = sum((s.size or 0) for s in siblings)
    print(
        f"[stage-gcs] === Starting {repo_id} -> gs://{bucket_name}/{prefix}/{repo_id}/ "
        f"({len(siblings)} files, {total_bytes/1e9:.2f} GB, workers={workers}) ===",
        flush=True,
    )
    t_repo = time.time()
    uploaded_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                transfer_one_file,
                repo_id,
                s.rfilename,
                s.size or 0,
                bucket_name,
                prefix,
                hf_token,
            ): s.rfilename
            for s in siblings
        }
        done_cnt = 0
        for fut in concurrent.futures.as_completed(futs):
            fname, status, nbytes = fut.result()
            done_cnt += 1
            uploaded_bytes += nbytes
            if done_cnt % 5 == 0 or done_cnt == len(siblings):
                elapsed = max(time.time() - t_repo, 1.0)
                print(
                    f"[stage-gcs] Progress {repo_id}: {done_cnt}/{len(siblings)} files "
                    f"({uploaded_bytes/1e9:.1f}/{total_bytes/1e9:.1f} GB, elapsed={elapsed:.0f}s)",
                    flush=True,
                )
    # Write a completion marker
    client = storage.Client()
    marker = client.bucket(bucket_name).blob(f"{prefix.rstrip('/')}/{repo_id}/_STAGE_COMPLETE.txt")
    marker.upload_from_string(f"completed_at={time.time()}\nfiles={len(siblings)}\nbytes={total_bytes}\n")
    print(f"[stage-gcs] === COMPLETED {repo_id} ({total_bytes/1e9:.2f} GB) ===", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", required=True, help="Comma-separated HF repo IDs in priority order")
    parser.add_argument("--bucket", default="YOUR_GCS_BUCKET")
    parser.add_argument("--prefix", default="models")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    for repo in [r.strip() for r in args.repos.split(",") if r.strip()]:
        stage_repo(repo, args.bucket, args.prefix, args.workers, hf_token)


if __name__ == "__main__":
    main()
