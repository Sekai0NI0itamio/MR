"""Tests for mr.models."""

from __future__ import annotations

import json

from mr.models import RecognitionResult, TrackMatch


class TestTrackMatch:
    def test_to_dict(self):
        tm = TrackMatch(
            track_id="abc123",
            title="Song Title",
            artist="Artist Name",
            album="Album Name",
            confidence=0.95,
        )
        d = tm.to_dict()
        assert d["track_id"] == "abc123"
        assert d["confidence"] == 0.95

    def test_fields(self):
        tm = TrackMatch(
            track_id="x",
            title="T",
            artist="A",
            album="Al",
            confidence=0.0,
        )
        assert tm.title == "T"
        assert tm.confidence == 0.0


class TestRecognitionResult:
    def _make_result(self, n_matches: int = 3, status: str = "success") -> RecognitionResult:
        matches = [
            TrackMatch(
                track_id=f"id{i}",
                title=f"Song {i}",
                artist=f"Artist {i}",
                album=f"Album {i}",
                confidence=round(0.9 - i * 0.1, 2),
            )
            for i in range(n_matches)
        ]
        return RecognitionResult(
            input_file="test.mp3",
            duration_seconds=200.5,
            matches=matches,
            status=status,
        )

    def test_to_json_roundtrip(self):
        result = self._make_result()
        j = result.to_json()
        data = json.loads(j)
        restored = RecognitionResult.from_dict(data)
        assert restored.input_file == "test.mp3"
        assert len(restored.matches) == 3
        assert restored.matches[0].confidence == 0.9

    def test_top_match(self):
        result = self._make_result()
        assert result.top_match is not None
        assert result.top_match.confidence == 0.9

    def test_top_match_empty(self):
        result = self._make_result(n_matches=0)
        assert result.top_match is None

    def test_high_confidence_count(self):
        result = self._make_result(n_matches=5)
        # Matches: 0.9, 0.8, 0.7, 0.6, 0.5 → all ≥ 0.5
        assert result.high_confidence_count == 5

    def test_failed_result(self):
        result = RecognitionResult(
            input_file="bad.mp3",
            duration_seconds=0.0,
            matches=[],
            status="failed",
            error="Corrupted file",
        )
        d = result.to_dict()
        assert d["status"] == "failed"
        assert d["error"] == "Corrupted file"
