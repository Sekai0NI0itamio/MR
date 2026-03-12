#!/usr/bin/env python3
"""
Download the AcoustID database dumps (fingerprints + track metadata).

This script:
  1. Crawls https://data.acoustid.org via index.json to find the latest
     year → month → day of available data.
  2. Downloads the compressed JSONL files for that day into ``db/``.
  3. Optionally truncates to ``--max-rows`` lines (for CI / testing).

Usage:
    python scripts/download_db.py [--max-rows N]
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import shutil
import sys
import urllib.request
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mr.config import (  # noqa: E402
    ACOUSTID_BASE_URL,
    DB_DIR,
    DB_FINGERPRINT_JSONL,
    DB_META_JSONL,
    DB_TRACK_FP_JSONL,
    DB_TRACK_JSONL,
    DB_TRACK_META_JSONL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_db")

_HEADERS = {"User-Agent": "MR/1.0"}


# ── Directory crawling ───────────────────────────────────────────────────


def _fetch_index(url: str) -> list[dict]:
    """Fetch and parse an ``index.json`` directory listing."""
    req = urllib.request.Request(url, headers=_HEADERS)  # noqa: S310
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        return json.loads(resp.read())


def _pick_latest_dir(entries: list[dict]) -> str:
    """Return the name of the last directory entry (sorted lexicographically)."""
    dirs = sorted(e["name"] for e in entries if e["name"].endswith("/"))
    if not dirs:
        raise RuntimeError(f"No sub-directories found in listing: {entries!r}")
    return dirs[-1].rstrip("/")


def _discover_latest_day_files(base_url: str) -> dict[str, str]:
    """
    Walk the AcoustID data directory to find the latest day's dump files.

    Returns a mapping ``{file_type: full_url}`` for the relevant files
    (e.g. ``"fingerprint-update"`` → ``"https://…/2026-03-11-fingerprint-update.jsonl.gz"``).
    """
    # 1. Latest year
    root_entries = _fetch_index(f"{base_url}/index.json")
    latest_year = _pick_latest_dir(root_entries)
    logger.info("Latest year: %s", latest_year)

    # 2. Latest month
    year_entries = _fetch_index(f"{base_url}/{latest_year}/index.json")
    latest_month = _pick_latest_dir(year_entries)
    logger.info("Latest month: %s", latest_month)

    # 3. Find latest day's files inside the month
    month_entries = _fetch_index(f"{base_url}/{latest_year}/{latest_month}/index.json")
    files = [e["name"] for e in month_entries if e["name"].endswith(".jsonl.gz")]

    # Extract dates from filenames like "2026-03-11-fingerprint-update.jsonl.gz"
    date_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})-")
    dates = sorted({m.group(1) for f in files if (m := date_pattern.match(f))})
    if not dates:
        raise RuntimeError(f"No dated JSONL files found in {latest_month}")
    latest_date = dates[-1]
    logger.info("Latest date: %s", latest_date)

    # Collect the files for that date
    prefix = f"{latest_date}-"
    day_files: dict[str, str] = {}
    for f in files:
        if f.startswith(prefix):
            # e.g. "fingerprint-update" from "2026-03-11-fingerprint-update.jsonl.gz"
            file_type = f[len(prefix):].removesuffix(".jsonl.gz")
            day_files[file_type] = f"{base_url}/{latest_year}/{latest_month}/{f}"

    logger.info("Found %d files for %s: %s", len(day_files), latest_date, list(day_files))
    return day_files


# ── Download / decompress ────────────────────────────────────────────────


def _download(url: str, dest: Path) -> Path:
    """Download *url* to *dest*."""
    logger.info("Downloading %s → %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=_HEADERS)  # noqa: S310
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f_out:  # noqa: S310
        shutil.copyfileobj(resp, f_out)
    logger.info("Download complete (%d bytes)", dest.stat().st_size)
    return dest


def _truncate_jsonl_gz(path: Path, max_rows: int) -> None:
    """Keep only the first *max_rows* lines in a gzipped JSONL file."""
    logger.info("Truncating %s to %d rows", path.name, max_rows)
    tmp = path.with_suffix(".tmp.gz")
    with gzip.open(path, "rt", encoding="utf-8") as fin, \
         gzip.open(tmp, "wt", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= max_rows:
                break
            fout.write(line)
    tmp.replace(path)
    logger.info("Truncated to %d rows", min(max_rows, i + 1) if 'i' in dir() else 0)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AcoustID database dumps.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Truncate each file to this many data rows (for testing).",
    )
    args = parser.parse_args()

    DB_DIR.mkdir(parents=True, exist_ok=True)

    # Discover latest dump URLs
    day_files = _discover_latest_day_files(ACOUSTID_BASE_URL)

    # ── Fingerprints ─────────────────────────────────────────────────
    if "fingerprint-update" not in day_files:
        logger.error("No fingerprint-update file found")
        sys.exit(1)
    _download(day_files["fingerprint-update"], DB_FINGERPRINT_JSONL)
    if args.max_rows:
        _truncate_jsonl_gz(DB_FINGERPRINT_JSONL, args.max_rows)

    # ── Track-fingerprint mapping ─────────────────────────────────────
    if "track_fingerprint-update" not in day_files:
        logger.error("No track_fingerprint-update file found")
        sys.exit(1)
    _download(day_files["track_fingerprint-update"], DB_TRACK_FP_JSONL)
    if args.max_rows:
        _truncate_jsonl_gz(DB_TRACK_FP_JSONL, args.max_rows)

    # ── Track metadata mapping (track_id → meta_id) ──────────────────
    if "track_meta-update" in day_files:
        _download(day_files["track_meta-update"], DB_TRACK_META_JSONL)
        if args.max_rows:
            _truncate_jsonl_gz(DB_TRACK_META_JSONL, args.max_rows)

    # ── Meta (title / artist / album) ────────────────────────────────
    if "meta-update" not in day_files:
        logger.error("No meta-update file found")
        sys.exit(1)
    _download(day_files["meta-update"], DB_META_JSONL)
    if args.max_rows:
        _truncate_jsonl_gz(DB_META_JSONL, args.max_rows)

    # ── Track records ────────────────────────────────────────────────
    if "track-update" in day_files:
        _download(day_files["track-update"], DB_TRACK_JSONL)
        if args.max_rows:
            _truncate_jsonl_gz(DB_TRACK_JSONL, args.max_rows)
        if args.max_rows:
            _truncate_jsonl_gz(DB_TRACK_META_JSONL, args.max_rows)

    logger.info("Database download complete.  Files in %s", DB_DIR)


if __name__ == "__main__":
    main()
