#!/usr/bin/env python3
"""
Build a FAISS index from the downloaded AcoustID CSV dumps.

This script:
  1. Parses ``db/fingerprints.csv`` into float32 vectors.
  2. Parses ``db/tracks.csv`` into a metadata dictionary.
  3. Builds a FAISS IVF-Flat (or Flat) index.
  4. Persists the index and metadata to disk for the pipeline to load.

Usage:
    python scripts/build_index.py [--max-rows N] [--nlist 256]

The ``--max-rows`` flag limits rows loaded from the fingerprint CSV
(useful for testing or memory-limited runners).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mr.config import FAISS_INDEX_PATH, FAISS_NLIST, FINGERPRINT_DIM, TRACK_METADATA_PATH  # noqa: E402
from mr.db import build_faiss_index, load_fingerprints_csv, parse_tracks_csv  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_index")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from AcoustID data.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Only load the first N fingerprint rows (for testing).",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=FAISS_NLIST,
        help="Number of IVF clusters (default: %(default)s).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=FINGERPRINT_DIM,
        help="Fingerprint vector dimension (default: %(default)s).",
    )
    args = parser.parse_args()

    # 1. Load track metadata
    track_metadata = parse_tracks_csv()

    # 2. Load fingerprint vectors
    vectors, track_ids = load_fingerprints_csv(dim=args.dim, max_rows=args.max_rows)
    if vectors.shape[0] == 0:
        logger.error("No fingerprint vectors loaded – aborting.")
        sys.exit(1)

    # 3. Build & save index
    build_faiss_index(
        vectors=vectors,
        track_ids=track_ids,
        track_metadata=track_metadata,
        index_path=FAISS_INDEX_PATH,
        metadata_path=TRACK_METADATA_PATH,
        nlist=args.nlist,
    )

    logger.info("Index built successfully.")
    logger.info("  Index : %s (%d bytes)", FAISS_INDEX_PATH, FAISS_INDEX_PATH.stat().st_size)
    logger.info("  Meta  : %s (%d bytes)", TRACK_METADATA_PATH, TRACK_METADATA_PATH.stat().st_size)


if __name__ == "__main__":
    main()
