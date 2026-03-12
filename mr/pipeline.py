"""
Pipeline orchestrator – coordinates the end-to-end recognition flow.

Called from the GitHub Actions workflow via ``python -m mr.pipeline``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from mr.audio import AudioProcessingError, process_audio_file
from mr.config import (
    LOGS_DIR,
    MAX_WORKERS,
    RESULTS_DIR,
    ensure_dirs,
)
from mr.db import FingerprintDB
from mr.models import RecognitionResult
from mr.search import recognize_file
from mr.summary import write_github_summary
from mr.utils import discover_audio_files, get_file_logger, setup_root_logger

logger = logging.getLogger("mr.pipeline")


# ── Per-file processing ──────────────────────────────────────────────────


def _process_one(audio_path: Path, db: FingerprintDB) -> RecognitionResult:
    """
    Process a single audio file through the full pipeline.

    Never raises – returns a ``RecognitionResult`` with ``status="failed"``
    on error.
    """
    file_logger = get_file_logger(audio_path.stem)
    file_logger.info("Processing %s", audio_path.name)

    try:
        _wav_path, duration, raw_fp = process_audio_file(audio_path)
        result = recognize_file(
            input_file=audio_path.name,
            duration=duration,
            raw_fingerprint=raw_fp,
            db=db,
        )
        file_logger.info(
            "Recognition complete: %d matches, top=%.4f",
            len(result.matches),
            result.matches[0].confidence if result.matches else 0.0,
        )
        return result

    except AudioProcessingError as exc:
        file_logger.error("Audio processing failed: %s", exc)
        return RecognitionResult(
            input_file=audio_path.name,
            duration_seconds=0.0,
            matches=[],
            status="failed",
            error=str(exc),
        )
    except Exception as exc:
        file_logger.error("Unexpected error: %s\n%s", exc, traceback.format_exc())
        return RecognitionResult(
            input_file=audio_path.name,
            duration_seconds=0.0,
            matches=[],
            status="failed",
            error=str(exc),
        )


# ── Main pipeline ────────────────────────────────────────────────────────


def run(max_workers: int | None = None) -> List[RecognitionResult]:
    """
    Discover audio files, process them concurrently, persist results,
    and generate the GitHub Actions summary.

    Returns the list of ``RecognitionResult`` objects (one per file).
    """
    setup_root_logger()
    ensure_dirs()
    max_workers = max_workers or MAX_WORKERS

    audio_files = discover_audio_files()
    if not audio_files:
        logger.warning("No audio files found in incoming/. Nothing to do.")
        return []

    logger.info("Found %d audio file(s) to process", len(audio_files))

    # Load the FAISS database once (shared across threads)
    try:
        db = FingerprintDB()
    except Exception as exc:
        logger.error("Failed to load fingerprint database: %s", exc)
        logger.error("Run scripts/download_db.py and scripts/build_index.py first.")
        sys.exit(1)

    results: List[RecognitionResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_path = {
            pool.submit(_process_one, path, db): path
            for path in audio_files
        }
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.error("Executor error for %s: %s", path.name, exc)
                result = RecognitionResult(
                    input_file=path.name,
                    duration_seconds=0.0,
                    matches=[],
                    status="failed",
                    error=str(exc),
                )
            results.append(result)

    # ── Persist results ──────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for r in results:
        stem = Path(r.input_file).stem
        out_path = RESULTS_DIR / f"{stem}.json"
        out_path.write_text(r.to_json(), encoding="utf-8")
        logger.info("Result written → %s", out_path)

    # ── Summary ──────────────────────────────────────────────────────
    write_github_summary(results)

    ok = sum(1 for r in results if r.status == "success")
    fail = sum(1 for r in results if r.status == "failed")
    logger.info("Pipeline complete: %d succeeded, %d failed", ok, fail)

    return results


# ── CLI entry point ──────────────────────────────────────────────────────

def main() -> None:
    workers = int(os.environ.get("MR_MAX_WORKERS", str(MAX_WORKERS)))
    results = run(max_workers=workers)
    if any(r.status == "failed" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
