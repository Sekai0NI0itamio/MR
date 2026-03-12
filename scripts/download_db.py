#!/usr/bin/env python3
"""
Download the AcoustID database dumps (fingerprints + track metadata).

This script:
  1. Crawls https://data.acoustid.org via ``index.json`` for the
     configured year range (default: last 5 years).
  2. Downloads metadata files (``meta-update``, ``track_meta-update``,
     ``track_fingerprint-update``) for every day, **caching** past days
     locally so they are only downloaded once.  The current day is always
     re-downloaded to pick up intra-day updates.
  3. Downloads the latest day's **fingerprint** file.
  4. Concatenates all cached + freshly downloaded metadata into merged
     output files for the index builder.

Concurrency:
  - All years are processed in parallel (unbounded).
  - Within each year, up to 50 file downloads run concurrently.

Usage:
    python scripts/download_db.py [--max-rows N] [--year-start Y] [--year-end Y]
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import logging
import re
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mr.config import (  # noqa: E402
    ACOUSTID_BASE_URL,
    DB_CACHE_DIR,
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

# Metadata file types that are collected across the full history.
_META_TYPES = {"meta-update", "track_meta-update", "track_fingerprint-update"}

# Max concurrent downloads inside a single year.
_MAX_CONCURRENT_DOWNLOADS = 50

_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-")


# ── Helpers ──────────────────────────────────────────────────────────────


def _fetch_index(url: str) -> list[dict]:
    """Fetch and parse an ``index.json`` directory listing."""
    req = urllib.request.Request(url, headers=_HEADERS)  # noqa: S310
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        return json.loads(resp.read())


def _pick_latest_dir(entries: list[dict]) -> str:
    dirs = sorted(e["name"] for e in entries if e["name"].endswith("/"))
    if not dirs:
        raise RuntimeError(f"No sub-directories found in listing: {entries!r}")
    return dirs[-1].rstrip("/")


def _cache_path_for(filename: str) -> Path:
    """Return the local cache path for a remote filename."""
    return DB_CACHE_DIR / filename


def _is_today(date_str: str) -> bool:
    """True if *date_str* (``YYYY-MM-DD``) is today (UTC)."""
    return date_str == datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def _is_valid_gz(path: Path) -> bool:
    """Return True if *path* is a valid gzip file we can fully read."""
    try:
        with gzip.open(path, "rb") as f:
            while f.read(65536):
                pass
        return True
    except (EOFError, gzip.BadGzipFile, OSError):
        return False


def _download_to(url: str, dest: Path, *, validate: bool = False) -> Path:
    """Download *url* → *dest* (raw bytes, no decompression)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=_HEADERS)  # noqa: S310
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f_out:  # noqa: S310
        shutil.copyfileobj(resp, f_out)
    if validate and not _is_valid_gz(dest):
        dest.unlink(missing_ok=True)
        raise OSError(f"Downloaded file is corrupt: {dest}")
    return dest


# ── Per-year download logic ─────────────────────────────────────────────


def _download_one_file(url: str, filename: str, date_str: str) -> Path:
    """
    Download a single metadata file, using the cache for past days.

    - Past days: return cached file if it exists, otherwise download & cache.
    - Today: always re-download (don't cache).
    """
    cached = _cache_path_for(filename)
    today = _is_today(date_str)

    if not today and cached.exists():
        if _is_valid_gz(cached):
            return cached
        logger.warning("Corrupt cache file %s — re-downloading", cached.name)
        cached.unlink(missing_ok=True)

    dest = cached if not today else cached.with_suffix(".today.gz")
    try:
        _download_to(url, dest, validate=True)
    except OSError:
        logger.warning("Download corrupt, retrying once: %s", filename)
        _download_to(url, dest, validate=True)

    if today:
        # Don't persist; return temp path that will be used then discarded.
        return dest

    logger.debug("Cached %s (%d bytes)", cached.name, cached.stat().st_size)
    return cached


def _process_year(base_url: str, year: str) -> list[tuple[str, Path]]:
    """
    Process all months in *year*: discover files, download (or read from
    cache) every metadata file.  Returns ``[(file_type, local_path), …]``.

    Downloads within the year run with up to ``_MAX_CONCURRENT_DOWNLOADS``
    threads.
    """
    year_entries = _fetch_index(f"{base_url}/{year}/index.json")
    month_dirs = sorted(
        e["name"].rstrip("/") for e in year_entries if e["name"].endswith("/")
    )

    # Collect (file_type, url, filename, date_str) tuples to download
    tasks: list[tuple[str, str, str, str]] = []
    for month in month_dirs:
        month_entries = _fetch_index(f"{base_url}/{year}/{month}/index.json")
        month_base = f"{base_url}/{year}/{month}"
        for entry in month_entries:
            fname = entry["name"]
            if not fname.endswith(".jsonl.gz"):
                continue
            m = _DATE_RE.match(fname)
            if not m:
                continue
            file_type = fname[len(m.group(0)):].removesuffix(".jsonl.gz")
            if file_type in _META_TYPES:
                tasks.append((file_type, f"{month_base}/{fname}", fname, m.group(1)))

    # Separate already-cached (past, file exists) from needing download
    results: list[tuple[str, Path]] = []
    to_download: list[tuple[str, str, str, str]] = []
    for ft, url, fname, date_str in tasks:
        cached = _cache_path_for(fname)
        if not _is_today(date_str) and cached.exists():
            results.append((ft, cached))
        else:
            to_download.append((ft, url, fname, date_str))

    if to_download:
        logger.info(
            "  %s: %d cached, %d to download",
            year, len(results), len(to_download),
        )
        with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_DOWNLOADS) as pool:
            futures = {
                pool.submit(_download_one_file, url, fname, ds): ft
                for ft, url, fname, ds in to_download
            }
            for future in as_completed(futures):
                ft = futures[future]
                path = future.result()
                results.append((ft, path))
    else:
        logger.info("  %s: %d cached, nothing to download", year, len(results))

    return results


# ── Merge cached files into final output ─────────────────────────────────


def _merge_files(file_paths: list[Path], dest: Path) -> Path:
    """
    Concatenate many gzipped JSONL files by appending raw bytes.

    The gzip format natively supports concatenated streams, so we skip
    the expensive decompress→recompress cycle entirely.
    """
    logger.info("Merging %d files → %s", len(file_paths), dest.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # A tiny gzip stream that decompresses to a single newline – appended
    # after each file to prevent line-fusion at stream boundaries.
    gz_newline = gzip.compress(b"\n")
    skipped = 0
    with open(dest, "wb") as out:
        for p in sorted(file_paths):
            try:
                with open(p, "rb") as f_in:
                    shutil.copyfileobj(f_in, out)
                out.write(gz_newline)
            except OSError as exc:
                logger.warning("Skipping unreadable file %s: %s", p.name, exc)
                skipped += 1
    if skipped:
        logger.warning("Skipped %d unreadable files during merge", skipped)
    logger.info("Merged %s (%d bytes)", dest.name, dest.stat().st_size)
    return dest


# ── Fingerprint + latest-day helpers ─────────────────────────────────────


def _discover_latest_fingerprint(base_url: str) -> tuple[str, dict[str, str]]:
    """
    Find the latest year → month → day and return
    ``(latest_date, {file_type: url})`` for that day.
    """
    root_entries = _fetch_index(f"{base_url}/index.json")
    latest_year = _pick_latest_dir(root_entries)
    year_entries = _fetch_index(f"{base_url}/{latest_year}/index.json")
    latest_month = _pick_latest_dir(year_entries)
    month_entries = _fetch_index(
        f"{base_url}/{latest_year}/{latest_month}/index.json"
    )
    files = [e["name"] for e in month_entries if e["name"].endswith(".jsonl.gz")]
    dates = sorted({m.group(1) for f in files if (m := _DATE_RE.match(f))})
    if not dates:
        raise RuntimeError("No dated JSONL files found")
    latest_date = dates[-1]

    month_base = f"{base_url}/{latest_year}/{latest_month}"
    prefix = f"{latest_date}-"
    day_files = {}
    for f in files:
        if f.startswith(prefix):
            ft = f[len(prefix):].removesuffix(".jsonl.gz")
            day_files[ft] = f"{month_base}/{f}"

    logger.info("Latest dump: %s (%d file types)", latest_date, len(day_files))
    return latest_date, day_files


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


def _default_year_start() -> int:
    return datetime.datetime.now(datetime.timezone.utc).year - 5


def _default_year_end() -> int:
    return datetime.datetime.now(datetime.timezone.utc).year


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AcoustID database dumps.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Truncate each file to this many data rows (for testing).",
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=_default_year_start(),
        help="First year of metadata to download (default: current year - 5).",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=_default_year_end(),
        help="Last year of metadata to download (default: current year).",
    )
    args = parser.parse_args()

    DB_DIR.mkdir(parents=True, exist_ok=True)
    DB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Fingerprints (latest day only) ────────────────────────────
    _, day_files = _discover_latest_fingerprint(ACOUSTID_BASE_URL)
    if "fingerprint-update" not in day_files:
        logger.error("No fingerprint-update file found")
        sys.exit(1)
    _download_to(day_files["fingerprint-update"], DB_FINGERPRINT_JSONL)
    logger.info("Fingerprints downloaded (%d bytes)", DB_FINGERPRINT_JSONL.stat().st_size)
    if args.max_rows:
        _truncate_jsonl_gz(DB_FINGERPRINT_JSONL, args.max_rows)

    # Optional track-update (latest day)
    if "track-update" in day_files:
        _download_to(day_files["track-update"], DB_TRACK_JSONL)
        if args.max_rows:
            _truncate_jsonl_gz(DB_TRACK_JSONL, args.max_rows)

    # ── 2. Metadata (year range, cached, concurrent) ────────────────
    root_entries = _fetch_index(f"{ACOUSTID_BASE_URL}/index.json")
    year_dirs = sorted(
        e["name"].rstrip("/") for e in root_entries if e["name"].endswith("/")
    )
    year_dirs = [
        y for y in year_dirs
        if y.isdigit() and args.year_start <= int(y) <= args.year_end
    ]
    logger.info(
        "Processing %d years (%d–%d): %s",
        len(year_dirs), args.year_start, args.year_end, year_dirs,
    )

    # Process all years concurrently (no limit on year threads)
    all_results: list[tuple[str, Path]] = []
    with ThreadPoolExecutor(max_workers=len(year_dirs)) as pool:
        year_futures = {
            pool.submit(_process_year, ACOUSTID_BASE_URL, y): y
            for y in year_dirs
        }
        for future in as_completed(year_futures):
            y = year_futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                logger.info("Year %s complete: %d files", y, len(results))
            except Exception:
                logger.exception("Failed to process year %s", y)

    # Group by file type
    by_type: dict[str, list[Path]] = {}
    for ft, path in all_results:
        by_type.setdefault(ft, []).append(path)

    for ft, paths in by_type.items():
        logger.info("  %s: %d files total", ft, len(paths))

    # ── 3. Merge into final output files ─────────────────────────────
    if "track_fingerprint-update" not in by_type:
        logger.error("No track_fingerprint-update files found")
        sys.exit(1)
    _merge_files(by_type["track_fingerprint-update"], DB_TRACK_FP_JSONL)
    if args.max_rows:
        _truncate_jsonl_gz(DB_TRACK_FP_JSONL, args.max_rows)

    if "track_meta-update" in by_type:
        _merge_files(by_type["track_meta-update"], DB_TRACK_META_JSONL)
        if args.max_rows:
            _truncate_jsonl_gz(DB_TRACK_META_JSONL, args.max_rows)

    if "meta-update" not in by_type:
        logger.error("No meta-update files found")
        sys.exit(1)
    _merge_files(by_type["meta-update"], DB_META_JSONL)
    if args.max_rows:
        _truncate_jsonl_gz(DB_META_JSONL, args.max_rows)

    # ── Clean up today's temp files ──────────────────────────────────
    for p in DB_CACHE_DIR.glob("*.today.gz"):
        p.unlink(missing_ok=True)

    logger.info("Database download complete.  Files in %s", DB_DIR)


if __name__ == "__main__":
    main()
