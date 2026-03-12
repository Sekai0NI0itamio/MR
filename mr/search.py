"""
High-level search logic: take a raw fingerprint, query the database,
and return a ``RecognitionResult`` with the top-K matches.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from mr.config import FINGERPRINT_DIM, TOP_K
from mr.db import FingerprintDB, fingerprint_to_vector
from mr.models import RecognitionResult, TrackMatch

logger = logging.getLogger("mr.search")


def _distance_to_confidence(distance: float, max_distance: float = 1e6) -> float:
    """
    Convert a FAISS L2 distance into a 0–1 confidence score.

    The mapping uses an inverse relationship: closer vectors (lower distance)
    yield higher confidence.  ``max_distance`` caps the denominator so that
    very large distances still yield a small positive number rather than ≈0.
    """
    if distance < 0:
        distance = 0.0
    return 1.0 / (1.0 + distance / max_distance)


def search(
    raw_fingerprint: List[int],
    db: FingerprintDB,
    *,
    top_k: int = TOP_K,
    dim: int = FINGERPRINT_DIM,
) -> List[TrackMatch]:
    """
    Query *db* with *raw_fingerprint* and return up to *top_k*
    ``TrackMatch`` objects sorted by descending confidence.
    """
    vec = fingerprint_to_vector(raw_fingerprint, dim)
    raw_results = db.query(vec, k=top_k)

    # Determine a normalisation ceiling from the worst distance in this
    # result set.  This makes confidence relative within the batch.
    if raw_results:
        max_dist = max(r[2] for r in raw_results) or 1.0
    else:
        max_dist = 1.0

    matches: List[TrackMatch] = []
    for track_id, meta, dist in raw_results:
        confidence = _distance_to_confidence(dist, max_distance=max_dist)
        matches.append(
            TrackMatch(
                track_id=track_id,
                title=meta.get("title", "Unknown"),
                artist=meta.get("artist", "Unknown"),
                album=meta.get("album", "Unknown"),
                confidence=round(confidence, 4),
            )
        )

    # Sort descending by confidence
    matches.sort(key=lambda m: m.confidence, reverse=True)
    return matches[:top_k]


def recognize_file(
    input_file: str,
    duration: float,
    raw_fingerprint: List[int],
    db: FingerprintDB,
) -> RecognitionResult:
    """
    Full recognition for a single file.  Wraps ``search()`` and returns
    a ``RecognitionResult``.
    """
    try:
        matches = search(raw_fingerprint, db)
        return RecognitionResult(
            input_file=input_file,
            duration_seconds=duration,
            matches=matches,
            status="success",
        )
    except Exception as exc:
        logger.exception("Search failed for %s", input_file)
        return RecognitionResult(
            input_file=input_file,
            duration_seconds=duration,
            matches=[],
            status="failed",
            error=str(exc),
        )
