"""
Audio conversion and fingerprint generation.

External dependencies:
  - **ffmpeg** (system binary) – used for normalising audio to 16-bit mono WAV.
  - **fpcalc** (Chromaprint CLI, bundled with pyacoustid) – used for
    generating compressed fingerprints from normalised audio.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import mr.config as cfg

logger = logging.getLogger("mr.audio")


class AudioProcessingError(Exception):
    """Raised when normalisation or fingerprinting fails."""


# ── Normalisation ─────────────────────────────────────────────────────────


def normalize_audio(input_path: Path, output_dir: Path | None = None) -> Path:
    """
    Convert *input_path* to a normalised WAV file using ``ffmpeg``.

    Returns the path to the normalised WAV.  The file is written into
    *output_dir* (defaults to ``DEBUG_DIR``).
    """
    output_dir = output_dir or cfg.DEBUG_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_normalised.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ac", str(cfg.NORMALIZED_CHANNELS),
        "-ar", str(cfg.NORMALIZED_SAMPLE_RATE),
        "-sample_fmt", f"s{cfg.NORMALIZED_BIT_DEPTH}",
        "-f", "wav",
        str(output_path),
    ]

    logger.info("Normalising %s → %s", input_path.name, output_path.name)
    logger.debug("ffmpeg command: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise AudioProcessingError(
            f"ffmpeg failed for {input_path.name}: {result.stderr[:500]}"
        )

    logger.debug("Normalisation complete: %s (%d bytes)", output_path.name, output_path.stat().st_size)
    return output_path


# ── Fingerprint generation ────────────────────────────────────────────────


def generate_fingerprint(audio_path: Path) -> tuple[float, list[int]]:
    """
    Run ``fpcalc`` on *audio_path* and return ``(duration, raw_fingerprint)``.

    The raw fingerprint is a list of 32-bit integer sub-fingerprints.
    """
    cmd = [
        "fpcalc",
        "-raw",          # output raw integers instead of compressed base64
        "-json",
        str(audio_path),
    ]

    logger.info("Generating fingerprint for %s", audio_path.name)
    logger.debug("fpcalc command: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise AudioProcessingError(
            f"fpcalc failed for {audio_path.name}: {result.stderr[:500]}"
        )

    data = json.loads(result.stdout)
    duration: float = float(data["duration"])
    fingerprint: list[int] = data["fingerprint"]

    logger.debug("Fingerprint: duration=%.1fs, %d sub-fingerprints", duration, len(fingerprint))

    # Optionally dump fingerprint text for debugging
    debug_fp_path = cfg.DEBUG_DIR / f"{audio_path.stem}_fingerprint.json"
    debug_fp_path.parent.mkdir(parents=True, exist_ok=True)
    debug_fp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return duration, fingerprint


# ── Convenience ───────────────────────────────────────────────────────────


def process_audio_file(input_path: Path) -> tuple[Path, float, list[int]]:
    """
    Full audio pipeline: normalise → fingerprint.

    Returns ``(normalised_wav_path, duration, raw_fingerprint)``.
    """
    wav_path = normalize_audio(input_path)
    duration, fingerprint = generate_fingerprint(wav_path)
    return wav_path, duration, fingerprint
