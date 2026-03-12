"""
Data models used throughout the MR pipeline.
"""

from __future__ import annotations

import dataclasses
import json
from typing import List, Optional


@dataclasses.dataclass
class TrackMatch:
    """A single candidate match returned by the search engine."""

    track_id: str
    title: str
    artist: str
    album: str
    confidence: float  # 0.0 – 1.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class RecognitionResult:
    """Complete recognition result for one input file."""

    input_file: str
    duration_seconds: float
    matches: List[TrackMatch]
    status: str            # "success" | "failed"
    error: Optional[str] = None

    # ── Derived helpers ──────────────────────────────────────────────────

    @property
    def high_confidence_count(self) -> int:
        from mr.config import HIGH_CONFIDENCE_THRESHOLD
        return sum(1 for m in self.matches if m.confidence >= HIGH_CONFIDENCE_THRESHOLD)

    @property
    def top_match(self) -> Optional[TrackMatch]:
        return self.matches[0] if self.matches else None

    # ── Serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "input_file": self.input_file,
            "duration_seconds": self.duration_seconds,
            "matches": [m.to_dict() for m in self.matches],
            "status": self.status,
            "error": self.error,
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, **kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "RecognitionResult":
        matches = [TrackMatch(**m) for m in data.get("matches", [])]
        return cls(
            input_file=data["input_file"],
            duration_seconds=data["duration_seconds"],
            matches=matches,
            status=data["status"],
            error=data.get("error"),
        )
