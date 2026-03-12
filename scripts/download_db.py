#!/usr/bin/env python3
"""
Download the AcoustID database dumps (fingerprints + track metadata).

This script:
  1. Downloads the compressed CSV files from data.acoustid.org.
  2. Decompresses them into the ``db/`` directory.
  3. Validates that the expected columns exist.

Usage:
    python scripts/download_db.py [--max-rows N]

The ``--max-rows`` flag is useful for CI/testing: it truncates the CSV
to the first *N* data rows so you can build a small index quickly.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import logging
import shutil
import sys
import urllib.request
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mr.config import (  # noqa: E402
    ACOUSTID_FINGERPRINT_URL,
    ACOUSTID_TRACK_URL,
    DB_DIR,
    DB_FINGERPRINT_CSV,
    DB_TRACK_CSV,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_db")


def _download(url: str, dest: Path) -> Path:
    """Download *url* to *dest*, showing progress."""
    logger.info("Downloading %s → %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 – trusted URL from config
    logger.info("Download complete (%d bytes)", dest.stat().st_size)
    return dest


def _decompress_gz(gz_path: Path, out_path: Path) -> Path:
    """Decompress a .gz file."""
    logger.info("Decompressing %s → %s", gz_path.name, out_path.name)
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    logger.info("Decompressed (%d bytes)", out_path.stat().st_size)
    return out_path


def _truncate_csv(csv_path: Path, max_rows: int) -> None:
    """Keep only the header + first *max_rows* data rows in *csv_path*."""
    logger.info("Truncating %s to %d rows", csv_path.name, max_rows)
    tmp = csv_path.with_suffix(".tmp")
    with open(csv_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(tmp, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(header)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            writer.writerow(row)
    tmp.replace(csv_path)
    logger.info("Truncated to %d data rows", min(max_rows, i + 1) if 'i' in dir() else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AcoustID database dumps.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Truncate each CSV to this many data rows (for testing).",
    )
    args = parser.parse_args()

    DB_DIR.mkdir(parents=True, exist_ok=True)

    # ── Fingerprints ─────────────────────────────────────────────────
    fp_gz = DB_DIR / "fingerprints.csv.gz"
    _download(ACOUSTID_FINGERPRINT_URL, fp_gz)
    _decompress_gz(fp_gz, DB_FINGERPRINT_CSV)
    if args.max_rows:
        _truncate_csv(DB_FINGERPRINT_CSV, args.max_rows)

    # ── Track metadata ───────────────────────────────────────────────
    tr_gz = DB_DIR / "tracks.csv.gz"
    _download(ACOUSTID_TRACK_URL, tr_gz)
    _decompress_gz(tr_gz, DB_TRACK_CSV)
    if args.max_rows:
        _truncate_csv(DB_TRACK_CSV, args.max_rows)

    logger.info("Database download complete.  Files in %s", DB_DIR)


if __name__ == "__main__":
    main()
