#!/usr/bin/env python3
"""
Download the AcoustID database dumps (fingerprints + track metadata).

This script:
  1. Crawls https://data.acoustid.org via ``index.json`` to find the
     latest year → month → day of available data.
  2. Downloads the latest day's **fingerprint** file (large, ~50 MB/day).
  3. Downloads **all days** in the latest month for the smaller metadata
     files (``meta-update``, ``track_meta-update``,
     ``track_fingerprint-update``) and concatenates them so that every
     track referenced in the fingerprint index has its title / artist /
     album resolved.
  4. Optionally truncates to ``--max-rows`` lines (for CI / testing).

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

# File types that must be collected across the *entire* month so that
# every track has metadata.  These are small (~1–3 MB each per day).
_MONTH_WIDE_TYPES = {"meta-update", "track_meta-update", "track_fingerprint-update"}


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


def _discover_month(base_url: str) -> tuple[str, str, list[dict]]:
    """Return ``(latest_year, latest_month, month_file_entries)``."""
    root_entries = _fetch_index(f"{base_url}/index.json")
    latest_year = _pick_latest_dir(root_entries)
    logger.info("Latest year: %s", latest_year)

    year_entries = _fetch_index(f"{base_url}/{latest_year}/index.json")
    latest_month = _pick_latest_dir(year_entries)
    logger.info("Latest month: %s", latest_month)

    month_entries = _fetch_index(
        f"{base_url}/{latest_year}/{latest_month}/index.json"
    )
    return latest_year, latest_month, month_entries


def _collect_all_metadata_urls(base_url: str) -> dict[str, list[str]]:
    """
    Walk **every** year / month directory and collect URLs for the small
    metadata file types (``meta-update``, ``track_meta-update``,
    ``track_fingerprint-update``).

    Returns ``{file_type: [url, …]}`` across *all* available history.
    """
    date_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})-")
    all_urls: dict[str, list[str]] = {}

    root_entries = _fetch_index(f"{base_url}/index.json")
    year_dirs = sorted(
        e["name"].rstrip("/") for e in root_entries if e["name"].endswith("/")
    )
    logger.info("Scanning %d years for metadata: %s", len(year_dirs), year_dirs)

    for year in year_dirs:
        year_entries = _fetch_index(f"{base_url}/{year}/index.json")
        month_dirs = sorted(
            e["name"].rstrip("/") for e in year_entries if e["name"].endswith("/")
        )
        for month in month_dirs:
            logger.info("  Indexing %s/%s …", year, month)
            month_entries = _fetch_index(
                f"{base_url}/{year}/{month}/index.json"
            )
            month_base = f"{base_url}/{year}/{month}"
            for entry in month_entries:
                fname = entry["name"]
                if not fname.endswith(".jsonl.gz"):
                    continue
                m = date_pattern.match(fname)
                if not m:
                    continue
                file_type = fname[len(m.group(0)):].removesuffix(".jsonl.gz")
                if file_type in _MONTH_WIDE_TYPES:
                    all_urls.setdefault(file_type, []).append(
                        f"{month_base}/{fname}"
                    )

    for ft, urls in all_urls.items():
        logger.info("  %s: %d files", ft, len(urls))
    return all_urls


def _classify_month_files(
    base_url: str,
    year: str,
    month: str,
    entries: list[dict],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Classify the files in a month directory.

    Returns:
      - ``latest_day_files``: ``{file_type: url}`` for the *latest day*
        (used for the large fingerprint dump).
      - ``all_month_files``: ``{file_type: [url, …]}`` for *every day*
        (used for the smaller metadata files).
    """
    files = [e["name"] for e in entries if e["name"].endswith(".jsonl.gz")]
    date_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})-")

    dates = sorted({m.group(1) for f in files if (m := date_pattern.match(f))})
    if not dates:
        raise RuntimeError(f"No dated JSONL files found in {month}")
    latest_date = dates[-1]
    logger.info("Latest date: %s  (%d days in month)", latest_date, len(dates))

    month_base = f"{base_url}/{year}/{month}"
    latest_day_files: dict[str, str] = {}
    all_month_files: dict[str, list[str]] = {}

    for f in sorted(files):
        m = date_pattern.match(f)
        if not m:
            continue
        file_type = f[len(m.group(0)):].removesuffix(".jsonl.gz")
        url = f"{month_base}/{f}"

        if m.group(1) == latest_date:
            latest_day_files[file_type] = url

        if file_type in _MONTH_WIDE_TYPES:
            all_month_files.setdefault(file_type, []).append(url)

    return latest_day_files, all_month_files


# ── Download helpers ─────────────────────────────────────────────────────


def _download(url: str, dest: Path) -> Path:
    """Download *url* to *dest*."""
    logger.info("Downloading %s → %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=_HEADERS)  # noqa: S310
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f_out:  # noqa: S310
        shutil.copyfileobj(resp, f_out)
    logger.info("Download complete (%d bytes)", dest.stat().st_size)
    return dest


def _download_and_concat(urls: list[str], dest: Path) -> Path:
    """
    Download multiple gzipped JSONL files and concatenate their contents
    into a single gzipped JSONL file at *dest*.
    """
    logger.info(
        "Downloading & concatenating %d files → %s", len(urls), dest
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest, "wt", encoding="utf-8") as out:
        for url in urls:
            logger.info("  ← %s", url.rsplit("/", 1)[-1])
            req = urllib.request.Request(url, headers=_HEADERS)  # noqa: S310
            with urllib.request.urlopen(req) as resp:  # noqa: S310
                with gzip.open(resp, "rt", encoding="utf-8") as gz:
                    shutil.copyfileobj(gz, out)
    logger.info("Concat complete (%d bytes)", dest.stat().st_size)
    return dest


def _truncate_jsonl_gz(path: Path, max_rows: int) -> None:
    """Keep only the first *max_rows* lines in a gzipped JSONL file."""
    logger.info("Truncating %s to %d rows", path.name, max_rows)
    tmp = path.with_suffix(".tmp.gz")
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as fin, \
         gzip.open(tmp, "wt", encoding="utf-8") as fout:
        for count, line in enumerate(fin):
            if count >= max_rows:
                break
            fout.write(line)
    tmp.replace(path)
    logger.info("Truncated to %d rows", min(max_rows, count))


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

    # ── Discover latest month for fingerprints ───────────────────────
    year, month, entries = _discover_month(ACOUSTID_BASE_URL)
    day_files, _ = _classify_month_files(
        ACOUSTID_BASE_URL, year, month, entries,
    )

    # ── Collect metadata URLs across ALL years/months ────────────────
    all_meta_urls = _collect_all_metadata_urls(ACOUSTID_BASE_URL)

    # ── Fingerprints (latest day only — large files) ─────────────────
    if "fingerprint-update" not in day_files:
        logger.error("No fingerprint-update file found")
        sys.exit(1)
    _download(day_files["fingerprint-update"], DB_FINGERPRINT_JSONL)
    if args.max_rows:
        _truncate_jsonl_gz(DB_FINGERPRINT_JSONL, args.max_rows)

    # ── Track-fingerprint mapping (all history) ──────────────────────
    if "track_fingerprint-update" not in all_meta_urls:
        logger.error("No track_fingerprint-update files found")
        sys.exit(1)
    _download_and_concat(all_meta_urls["track_fingerprint-update"], DB_TRACK_FP_JSONL)
    if args.max_rows:
        _truncate_jsonl_gz(DB_TRACK_FP_JSONL, args.max_rows)

    # ── Track→meta mapping (all history) ─────────────────────────────
    if "track_meta-update" in all_meta_urls:
        _download_and_concat(all_meta_urls["track_meta-update"], DB_TRACK_META_JSONL)
        if args.max_rows:
            _truncate_jsonl_gz(DB_TRACK_META_JSONL, args.max_rows)

    # ── Meta: title / artist / album (all history) ───────────────────
    if "meta-update" not in all_meta_urls:
        logger.error("No meta-update files found")
        sys.exit(1)
    _download_and_concat(all_meta_urls["meta-update"], DB_META_JSONL)
    if args.max_rows:
        _truncate_jsonl_gz(DB_META_JSONL, args.max_rows)

    # ── Track records (latest day only, optional) ────────────────────
    if "track-update" in day_files:
        _download(day_files["track-update"], DB_TRACK_JSONL)
        if args.max_rows:
            _truncate_jsonl_gz(DB_TRACK_JSONL, args.max_rows)

    logger.info("Database download complete.  Files in %s", DB_DIR)


if __name__ == "__main__":
    main()
