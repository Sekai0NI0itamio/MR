"""
Database loading, index building, and fingerprint querying.

This module handles:
  1. Parsing the AcoustID CSV dumps into usable structures.
  2. Building a FAISS IVF index from raw sub-fingerprint vectors.
  3. Loading a pre-built index + metadata from disk.
  4. Querying the index for the *k* nearest neighbours.

The fingerprint vectors stored in the index are derived from the raw
Chromaprint sub-fingerprint arrays.  Each track's fingerprint is
truncated or zero-padded to ``FINGERPRINT_DIM`` 32-bit integers, then
stored as a float32 vector so FAISS can work with it.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from mr.config import (
    DB_DIR,
    DB_FINGERPRINT_CSV,
    DB_FINGERPRINT_JSONL,
    DB_TRACK_CSV,
    DB_TRACK_JSONL,
    DB_TRACK_META_JSONL,
    FAISS_INDEX_PATH,
    FAISS_NLIST,
    FINGERPRINT_DIM,
    TRACK_METADATA_PATH,
)

logger = logging.getLogger("mr.db")

# We import faiss lazily so the module can be imported even when faiss
# is not installed (e.g. during lightweight unit tests).
_faiss = None


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss  # type: ignore[import-untyped]
        _faiss = faiss
    return _faiss


# ── Fingerprint vector utilities ──────────────────────────────────────────


def fingerprint_to_vector(raw_fp: List[int], dim: int = FINGERPRINT_DIM) -> np.ndarray:
    """
    Convert a variable-length list of 32-bit sub-fingerprint ints into a
    fixed-length float32 vector suitable for FAISS.

    Values are cast from int32 → float32 to preserve bit-level information
    while allowing FAISS L2 distance computations.
    """
    arr = np.array(raw_fp[:dim], dtype=np.int32).astype(np.float32)
    if len(arr) < dim:
        arr = np.pad(arr, (0, dim - len(arr)), mode="constant", constant_values=0)
    return arr


# ── Data parsing (CSV + JSONL) ────────────────────────────────────────────


def _detect_fingerprint_path() -> Path:
    """Return the first existing fingerprint file (JSONL preferred)."""
    if DB_FINGERPRINT_JSONL.exists():
        return DB_FINGERPRINT_JSONL
    return DB_FINGERPRINT_CSV


def _detect_track_path() -> Path:
    """Return the first existing track-metadata file (JSONL preferred)."""
    if DB_TRACK_META_JSONL.exists():
        return DB_TRACK_META_JSONL
    if DB_TRACK_JSONL.exists():
        return DB_TRACK_JSONL
    return DB_TRACK_CSV


def _is_jsonl(path: Path) -> bool:
    name = path.name
    return name.endswith(".jsonl") or name.endswith(".jsonl.gz")


def _open_file(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_tracks_csv(path: Path | None = None) -> Dict[str, dict]:
    """
    Parse an AcoustID track-metadata file (CSV or JSONL, optionally gzipped)
    into a mapping ``{track_id: {"title": ..., "artist": ..., "album": ...}}``.
    """
    path = path or _detect_track_path()
    logger.info("Parsing track metadata from %s", path)

    tracks: Dict[str, dict] = {}

    if _is_jsonl(path):
        with _open_file(path) as fh:
            for line in fh:
                row = json.loads(line)
                tid = str(row.get("id") or row.get("track_id", ""))
                tracks[tid] = {
                    "title": row.get("title", "Unknown"),
                    "artist": row.get("artist", "Unknown"),
                    "album": row.get("album", "Unknown"),
                }
    else:
        with _open_file(path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                tid = row.get("id") or row.get("track_id", "")
                tracks[tid] = {
                    "title": row.get("title", "Unknown"),
                    "artist": row.get("artist", "Unknown"),
                    "album": row.get("album", "Unknown"),
                }
    logger.info("Loaded %d track metadata entries", len(tracks))
    return tracks


def load_fingerprints_csv(
    path: Path | None = None,
    dim: int = FINGERPRINT_DIM,
    max_rows: int | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Parse an AcoustID fingerprint file (CSV or JSONL, optionally gzipped)
    and return:
      - ``vectors``: float32 array of shape ``(N, dim)``
      - ``track_ids``: list of corresponding track IDs (length N)

    If *max_rows* is set, only the first *max_rows* entries are loaded
    (useful for testing or memory-constrained environments).
    """
    path = path or _detect_fingerprint_path()
    logger.info("Loading fingerprints from %s (dim=%d)", path, dim)

    vectors: list[np.ndarray] = []
    track_ids: list[str] = []

    if _is_jsonl(path):
        with _open_file(path) as fh:
            for i, line in enumerate(fh):
                if max_rows is not None and i >= max_rows:
                    break
                row = json.loads(line)
                tid = str(row.get("track_id") or row.get("id", ""))
                raw = row.get("fingerprint", [])
                if isinstance(raw, str):
                    raw = [int(x) for x in raw.split(",") if x.strip()]
                if not raw:
                    continue
                vec = fingerprint_to_vector(raw, dim)
                vectors.append(vec)
                track_ids.append(tid)
    else:
        with _open_file(path) as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                if max_rows is not None and i >= max_rows:
                    break
                tid = row.get("track_id") or row.get("id", "")
                raw_str = row.get("fingerprint", "")
                if not raw_str:
                    continue
                raw_ints = [int(x) for x in raw_str.split(",") if x.strip()]
                vec = fingerprint_to_vector(raw_ints, dim)
                vectors.append(vec)
                track_ids.append(tid)

    mat = np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, dim), dtype=np.float32)
    logger.info("Loaded %d fingerprint vectors (%s)", mat.shape[0], mat.shape)
    return mat, track_ids


# ── Index building ────────────────────────────────────────────────────────


def build_faiss_index(
    vectors: np.ndarray,
    track_ids: List[str],
    track_metadata: Dict[str, dict],
    index_path: Path | None = None,
    metadata_path: Path | None = None,
    nlist: int = FAISS_NLIST,
) -> None:
    """
    Build a FAISS IVF-Flat index from *vectors* and persist it together
    with the metadata mapping to disk.
    """
    faiss = _get_faiss()
    index_path = index_path or FAISS_INDEX_PATH
    metadata_path = metadata_path or TRACK_METADATA_PATH
    index_path.parent.mkdir(parents=True, exist_ok=True)

    dim = vectors.shape[1]
    n = vectors.shape[0]
    logger.info("Building FAISS IVF-Flat index: %d vectors, dim=%d, nlist=%d", n, dim, nlist)

    # Use a flat index for small datasets, IVF for larger ones
    if n < nlist * 40:
        logger.info("Dataset too small for IVF; falling back to IndexFlatL2")
        index = faiss.IndexFlatL2(dim)
    else:
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(vectors)

    index.add(vectors)
    faiss.write_index(index, str(index_path))
    logger.info("FAISS index written to %s", index_path)

    combined_meta = {
        "track_ids": track_ids,
        "track_metadata": track_metadata,
    }
    with open(metadata_path, "wb") as fh:
        pickle.dump(combined_meta, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Metadata written to %s", metadata_path)


# ── Index loading ─────────────────────────────────────────────────────────


class FingerprintDB:
    """
    Encapsulates a loaded FAISS index + track metadata for querying.
    """

    def __init__(
        self,
        index_path: Path | None = None,
        metadata_path: Path | None = None,
    ):
        faiss = _get_faiss()
        index_path = index_path or FAISS_INDEX_PATH
        metadata_path = metadata_path or TRACK_METADATA_PATH

        logger.info("Loading FAISS index from %s", index_path)
        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, "rb") as fh:
            meta = pickle.load(fh)  # noqa: S301 – trusted local file
        self.track_ids: List[str] = meta["track_ids"]
        self.track_metadata: Dict[str, dict] = meta["track_metadata"]
        logger.info("Index loaded: %d vectors", self.index.ntotal)

    # ── Query ─────────────────────────────────────────────────────────

    def query(
        self,
        fingerprint_vector: np.ndarray,
        k: int = 20,
        nprobe: int = 10,
    ) -> List[Tuple[str, dict, float]]:
        """
        Search for the *k* nearest neighbours of *fingerprint_vector*.

        Returns a list of ``(track_id, metadata_dict, distance)`` sorted
        by ascending distance (lower = more similar).
        """
        faiss = _get_faiss()
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe

        vec = fingerprint_vector.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(vec, k)

        results: List[Tuple[str, dict, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            tid = self.track_ids[idx]
            meta = self.track_metadata.get(tid, {"title": "Unknown", "artist": "Unknown", "album": "Unknown"})
            results.append((tid, meta, float(dist)))
        return results
