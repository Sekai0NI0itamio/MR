"""Tests for mr.search."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from mr.search import _distance_to_confidence, search, recognize_file
from mr.models import TrackMatch


class TestDistanceToConfidence:
    def test_zero_distance_returns_one(self):
        assert _distance_to_confidence(0.0) == 1.0

    def test_large_distance_returns_low_confidence(self):
        conf = _distance_to_confidence(1e12)
        assert 0.0 < conf < 0.01

    def test_negative_distance_clamped(self):
        assert _distance_to_confidence(-5.0) == 1.0

    def test_moderate_distance(self):
        conf = _distance_to_confidence(500_000, max_distance=1_000_000)
        assert 0.5 < conf < 0.8


class TestSearch:
    def _mock_db(self, results):
        """Create a mock FingerprintDB that returns *results* from query()."""
        db = MagicMock()
        db.query.return_value = results
        return db

    def test_returns_sorted_matches(self):
        raw_results = [
            ("t1", {"title": "Far", "artist": "A1", "album": "Al1"}, 100.0),
            ("t2", {"title": "Close", "artist": "A2", "album": "Al2"}, 10.0),
            ("t3", {"title": "Mid", "artist": "A3", "album": "Al3"}, 50.0),
        ]
        db = self._mock_db(raw_results)
        matches = search([1, 2, 3], db, top_k=3)

        assert len(matches) == 3
        # Closest (lowest distance) should have highest confidence
        assert matches[0].title == "Close"
        assert matches[-1].title == "Far"
        # All confidences between 0 and 1
        assert all(0.0 <= m.confidence <= 1.0 for m in matches)

    def test_empty_db_returns_empty(self):
        db = self._mock_db([])
        matches = search([1, 2, 3], db)
        assert matches == []

    def test_top_k_limits_results(self):
        raw_results = [
            (f"t{i}", {"title": f"S{i}", "artist": "A", "album": "Al"}, float(i))
            for i in range(10)
        ]
        db = self._mock_db(raw_results)
        matches = search([1], db, top_k=5)
        assert len(matches) == 5


class TestRecognizeFile:
    def test_success(self):
        db = MagicMock()
        db.query.return_value = [
            ("t1", {"title": "Hit", "artist": "Star", "album": "Best"}, 5.0),
        ]
        result = recognize_file("song.mp3", 200.0, [10, 20, 30], db)
        assert result.status == "success"
        assert len(result.matches) == 1
        assert result.matches[0].title == "Hit"

    def test_db_error_returns_failed(self):
        db = MagicMock()
        db.query.side_effect = RuntimeError("index corrupt")
        result = recognize_file("bad.mp3", 0.0, [1], db)
        assert result.status == "failed"
        assert "index corrupt" in result.error
