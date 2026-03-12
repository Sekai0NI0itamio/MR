"""
Configuration constants and helpers for the MR pipeline.

All paths are resolved relative to the repository root (PROJECT_ROOT).
Thresholds, parallelism knobs, and database URLs live here so they
are easy to override from environment variables or workflow inputs.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.environ.get("MR_PROJECT_ROOT", Path(__file__).resolve().parent.parent))

INCOMING_DIR = PROJECT_ROOT / "incoming"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
DEBUG_DIR = PROJECT_ROOT / "debug"

DB_DIR = PROJECT_ROOT / "db"
DB_FINGERPRINT_CSV = DB_DIR / "fingerprints.csv"
DB_TRACK_CSV = DB_DIR / "tracks.csv"
FAISS_INDEX_PATH = DB_DIR / "fingerprints.index"
TRACK_METADATA_PATH = DB_DIR / "track_metadata.pkl"

# ── Audio normalisation ──────────────────────────────────────────────────
NORMALIZED_SAMPLE_RATE = 44100
NORMALIZED_CHANNELS = 1       # mono
NORMALIZED_BIT_DEPTH = 16
SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".ogg"}

# ── Recognition knobs ────────────────────────────────────────────────────
TOP_K = int(os.environ.get("MR_TOP_K", "20"))
HIGH_CONFIDENCE_THRESHOLD = float(os.environ.get("MR_HIGH_CONF", "0.5"))

# ── Parallelism ──────────────────────────────────────────────────────────
MAX_WORKERS = int(os.environ.get("MR_MAX_WORKERS", "4"))

# ── AcoustID data directory ──────────────────────────────────────────────
ACOUSTID_BASE_URL = os.environ.get(
    "MR_ACOUSTID_BASE_URL",
    "https://data.acoustid.org",
)

# Local paths for downloaded JSONL dump files
DB_FINGERPRINT_JSONL = DB_DIR / "fingerprints.jsonl.gz"
DB_TRACK_FP_JSONL = DB_DIR / "track_fingerprint.jsonl.gz"
DB_TRACK_JSONL = DB_DIR / "tracks.jsonl.gz"
DB_TRACK_META_JSONL = DB_DIR / "track_meta.jsonl.gz"
DB_META_JSONL = DB_DIR / "meta.jsonl.gz"

# ── FAISS index parameters ───────────────────────────────────────────────
FINGERPRINT_DIM = 120          # number of int32 sub-fingerprints kept per track
FAISS_NPROBE = 10              # clusters to visit during search
FAISS_NLIST = 256              # number of IVF clusters (tuned at build time)

# ── Helpers ───────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create all output directories if they don't already exist."""
    for d in (INCOMING_DIR, RESULTS_DIR, LOGS_DIR, DEBUG_DIR, DB_DIR):
        d.mkdir(parents=True, exist_ok=True)
