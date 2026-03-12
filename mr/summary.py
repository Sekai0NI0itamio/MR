"""
GitHub Actions summary generation.

Writes a Markdown table to ``$GITHUB_STEP_SUMMARY`` (or stdout if
running locally) with per-file recognition results.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from mr.models import RecognitionResult


def _escape_md(text: str) -> str:
    """Escape pipe characters for Markdown tables."""
    return text.replace("|", "\\|")


def build_summary_table(results: List[RecognitionResult]) -> str:
    """
    Build a Markdown table summarising all recognition results.
    """
    lines = [
        "## 🎵 Music Recognizer – Run Summary\n",
        "| Filename | Duration (s) | Matches (≥0.5) | Top Match | Confidence | Status | Error |",
        "|----------|-------------|----------------|-----------|------------|--------|-------|",
    ]

    for r in sorted(results, key=lambda x: x.input_file):
        top = r.top_match
        top_str = f"{_escape_md(top.title)} – {_escape_md(top.artist)}" if top else "—"
        conf_str = f"{top.confidence:.4f}" if top else "—"
        err_str = _escape_md(r.error[:80]) if r.error else ""
        status_icon = "✅" if r.status == "success" else "❌"

        lines.append(
            f"| {_escape_md(r.input_file)} "
            f"| {r.duration_seconds:.1f} "
            f"| {r.high_confidence_count} "
            f"| {top_str} "
            f"| {conf_str} "
            f"| {status_icon} {r.status} "
            f"| {err_str} |"
        )

    lines.append("")
    return "\n".join(lines)


def write_github_summary(results: List[RecognitionResult]) -> None:
    """
    Append the summary Markdown to ``$GITHUB_STEP_SUMMARY`` if available,
    otherwise print to stdout.
    """
    md = build_summary_table(results)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(md)
    else:
        print(md)
