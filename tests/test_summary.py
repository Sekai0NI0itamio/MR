"""Tests for mr.summary."""

from __future__ import annotations

from mr.models import RecognitionResult, TrackMatch
from mr.summary import build_summary_table


class TestBuildSummaryTable:
    def _make_result(self, filename="test.mp3", status="success", n=2):
        matches = [
            TrackMatch(
                track_id=f"id{i}",
                title=f"Song {i}",
                artist=f"Artist {i}",
                album=f"Album {i}",
                confidence=round(0.9 - i * 0.15, 2),
            )
            for i in range(n)
        ]
        return RecognitionResult(
            input_file=filename,
            duration_seconds=123.4,
            matches=matches,
            status=status,
            error="bad data" if status == "failed" else None,
        )

    def test_contains_header(self):
        md = build_summary_table([self._make_result()])
        assert "Filename" in md
        assert "Status" in md

    def test_contains_filename(self):
        md = build_summary_table([self._make_result("hello.mp3")])
        assert "hello.mp3" in md

    def test_failed_includes_error(self):
        md = build_summary_table([self._make_result(status="failed")])
        assert "bad data" in md
        assert "❌" in md

    def test_success_icon(self):
        md = build_summary_table([self._make_result()])
        assert "✅" in md

    def test_multiple_files(self):
        results = [
            self._make_result("a.mp3"),
            self._make_result("b.ogg", status="failed"),
        ]
        md = build_summary_table(results)
        assert "a.mp3" in md
        assert "b.ogg" in md

    def test_no_matches(self):
        r = self._make_result(n=0)
        md = build_summary_table([r])
        assert "—" in md  # em-dash placeholder for empty top match

    def test_pipe_in_title_escaped(self):
        r = RecognitionResult(
            input_file="pipe.mp3",
            duration_seconds=10.0,
            matches=[
                TrackMatch(
                    track_id="x",
                    title="A | B",
                    artist="C | D",
                    album="E",
                    confidence=0.8,
                )
            ],
            status="success",
        )
        md = build_summary_table([r])
        # Pipes in content must be escaped so they don't break the table
        assert "A \\| B" in md
