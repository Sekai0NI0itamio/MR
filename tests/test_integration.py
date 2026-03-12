"""
Integration tests – exercise the pipeline end-to-end with mocked
external tools (ffmpeg, fpcalc) and a tiny in-memory FAISS index.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import mr.config as cfg
from mr.db import FingerprintDB, build_faiss_index, fingerprint_to_vector
from mr.models import RecognitionResult
from mr.search import recognize_file


@pytest.fixture
def tiny_db(tmp_path: Path):
    """
    Build a small FAISS index with 5 dummy tracks and return a
    ``FingerprintDB`` instance.
    """
    dim = cfg.FINGERPRINT_DIM
    n = 5
    track_ids = [f"track_{i}" for i in range(n)]
    track_metadata = {
        tid: {"title": f"Song {i}", "artist": f"Artist {i}", "album": f"Album {i}"}
        for i, tid in enumerate(track_ids)
    }

    rng = np.random.RandomState(42)
    vectors = rng.randn(n, dim).astype(np.float32)

    index_path = cfg.FAISS_INDEX_PATH
    meta_path = cfg.TRACK_METADATA_PATH

    build_faiss_index(
        vectors=vectors,
        track_ids=track_ids,
        track_metadata=track_metadata,
        index_path=index_path,
        metadata_path=meta_path,
    )

    return FingerprintDB(index_path=index_path, metadata_path=meta_path)


class TestIntegrationSearch:
    def test_query_returns_results(self, tiny_db: FingerprintDB):
        vec = np.zeros(cfg.FINGERPRINT_DIM, dtype=np.float32)
        results = tiny_db.query(vec, k=3)
        assert len(results) == 3
        for tid, meta, dist in results:
            assert tid.startswith("track_")
            assert "title" in meta

    def test_recognize_file_success(self, tiny_db: FingerprintDB):
        result = recognize_file(
            input_file="demo.mp3",
            duration=60.0,
            raw_fingerprint=list(range(cfg.FINGERPRINT_DIM)),
            db=tiny_db,
        )
        assert result.status == "success"
        assert len(result.matches) > 0
        assert result.matches[0].confidence > 0.0

    def test_full_roundtrip_json(self, tiny_db: FingerprintDB):
        result = recognize_file(
            input_file="roundtrip.ogg",
            duration=120.0,
            raw_fingerprint=[42] * cfg.FINGERPRINT_DIM,
            db=tiny_db,
        )
        j = result.to_json()
        data = json.loads(j)
        restored = RecognitionResult.from_dict(data)
        assert restored.input_file == "roundtrip.ogg"
        assert restored.duration_seconds == 120.0
        assert len(restored.matches) == len(result.matches)
