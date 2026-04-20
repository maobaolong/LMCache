#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Chunk hash log uploader agent.

Periodically scans a directory for completed chunk hash JSONL files,
uploads them to Tencent Mirrors generic repository, and moves
successfully uploaded files to an ``uploaded/`` subdirectory to
avoid re-uploading.

Usage::

    python upload_chunk_hash_logs.py \
        --log-dir /data1/here_I_test_chunk_hash \
        --interval 60 \
        --username xxx \
        --token  xxx_token \
        --repo LMCache-MP-Chunk-Hashes

Files are uploaded under the path::

    {LOCAL_IP}_{MODEL_NAME}/{YYYY-MM-DD}/{filename}

The script skips files that are still being actively written to
(i.e. not modified in the last ``--stable-seconds`` window).
"""

# Future
from __future__ import annotations

# Standard
from datetime import datetime, timezone
from pathlib import Path
import argparse
import logging
import os
import platform
import shutil
import time

# Third Party
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("chunk_hash_uploader")

DEFAULT_BASE_URL = "https://mirrors.tencent.com/repository/generic"


def _is_stable(path: Path, stable_seconds: float) -> bool:
    """Return True if the file has not been modified recently."""
    try:
        mtime = path.stat().st_mtime
        return (time.time() - mtime) >= stable_seconds
    except OSError:
        return False


def upload_file(
    file_path: Path,
    *,
    base_url: str,
    repo: str,
    remote_path: str,
    username: str,
    token: str,
    timeout: int = 120,
) -> bool:
    """Upload a single file via HTTP PUT with basic auth.

    Returns True on success, False on failure.
    """
    url = f"{base_url}/{repo}/{remote_path}"
    logger.info("Uploading %s -> %s", file_path.name, url)
    try:
        with open(file_path, "rb") as f:
            resp = requests.put(
                url,
                data=f,
                auth=(username, token),
                timeout=timeout,
            )
        if resp.status_code in (200, 201):
            logger.info("Uploaded %s (HTTP %d)", file_path.name, resp.status_code)
            return True
        else:
            logger.error(
                "Upload failed for %s: HTTP %d — %s",
                file_path.name,
                resp.status_code,
                resp.text[:200],
            )
            return False
    except requests.RequestException as e:
        logger.error("Upload error for %s: %s", file_path.name, e)
        return False


def process_directory(
    log_dir: Path,
    *,
    base_url: str,
    repo: str,
    hostname: str,
    username: str,
    token: str,
    stable_seconds: float,
) -> int:
    """Scan log_dir for uploadable JSONL files.

    Returns the number of files successfully uploaded.
    """
    uploaded_dir = log_dir / "uploaded"
    uploaded_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(log_dir.glob("chunk_hashes_*.jsonl"))
    if not files:
        logger.debug("No files found in %s", log_dir)
        return 0

    uploaded_count = 0
    for file_path in files:
        if not _is_stable(file_path, stable_seconds):
            logger.debug("Skipping %s (still being written)", file_path.name)
            continue

        # Build remote path: {LOCAL_IP}_{MODEL_NAME}/{YYYY-MM-DD}/{filename}
        file_mtime = file_path.stat().st_mtime
        date_str = datetime.fromtimestamp(file_mtime, tz=timezone.utc).strftime(
            "%Y-%m-%d"
        )

        # Get environment variables with defaults
        local_ip = os.environ.get("LOCAL_IP", "0.0.0.0")
        model_name = os.environ.get("MODEL_NAME", "xxx")

        remote_path = f"{local_ip}_{model_name}/{date_str}/{file_path.name}"

        ok = upload_file(
            file_path,
            base_url=base_url,
            repo=repo,
            remote_path=remote_path,
            username=username,
            token=token,
        )
        if ok:
            dest = uploaded_dir / file_path.name
            shutil.move(str(file_path), str(dest))
            logger.info("Moved %s -> uploaded/", file_path.name)
            uploaded_count += 1

    return uploaded_count


def run_loop(args: argparse.Namespace) -> None:
    """Main polling loop."""
    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        logger.error("Log directory does not exist: %s", log_dir)
        return

    hostname = args.hostname or platform.node()
    logger.info(
        "Starting chunk hash uploader: log_dir=%s, interval=%ds, hostname=%s, repo=%s",
        log_dir,
        args.interval,
        hostname,
        args.repo,
    )

    while True:
        try:
            count = process_directory(
                log_dir,
                base_url=args.base_url,
                repo=args.repo,
                hostname=hostname,
                username=args.username,
                token=args.token,
                stable_seconds=args.stable_seconds,
            )
            if count:
                logger.info("Uploaded %d file(s) this cycle", count)
        except Exception:
            logger.exception("Error during upload cycle")

        time.sleep(args.interval)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Periodically upload chunk hash JSONL files to Tencent Mirrors.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Directory containing chunk hash JSONL files.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=600,
        help="Seconds between upload cycles (default: 600 = 10 min).",
    )
    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="Username for Tencent Mirrors authentication.",
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Token for Tencent Mirrors authentication.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="LMCache-MP-Chunk-Hashes",
        help="Generic repository name (default: LMCache-MP-Chunk-Hashes).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="Base URL for the generic repository API.",
    )
    parser.add_argument(
        "--hostname",
        type=str,
        default="",
        help="Hostname for upload path. Defaults to platform.node().",
    )
    parser.add_argument(
        "--stable-seconds",
        type=float,
        default=60.0,
        help="Skip files modified within this many seconds "
        "(default: 60). Avoids uploading files still being written.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single upload cycle and exit (no polling loop).",
    )

    args = parser.parse_args()

    if args.once:
        log_dir = Path(args.log_dir)
        if not log_dir.is_dir():
            logger.error("Log directory does not exist: %s", log_dir)
            return
        hostname = args.hostname or platform.node()
        count = process_directory(
            log_dir,
            base_url=args.base_url,
            repo=args.repo,
            hostname=hostname,
            username=args.username,
            token=args.token,
            stable_seconds=args.stable_seconds,
        )
        logger.info("Uploaded %d file(s)", count)
    else:
        run_loop(args)


if __name__ == "__main__":
    main()
