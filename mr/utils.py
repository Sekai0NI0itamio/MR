"""
Logging helpers and general-purpose file utilities.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

from mr.config import INCOMING_DIR, LOGS_DIR, SUPPORTED_EXTENSIONS


def setup_root_logger(level: int = logging.DEBUG) -> logging.Logger:
    """
    Configure and return the root ``mr`` logger.

    Logs go to both *stderr* (INFO+) and a timestamped file inside
    ``LOGS_DIR`` (DEBUG+).
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("mr")
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (INFO+)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (DEBUG+)
    fh = logging.FileHandler(LOGS_DIR / "pipeline.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def get_file_logger(name: str) -> logging.Logger:
    """
    Return a child logger that also writes to ``LOGS_DIR/<name>.log``.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"mr.{name}")
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(LOGS_DIR / f"{name}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def discover_audio_files(directory: Path | None = None) -> List[Path]:
    """
    Return a sorted list of audio files found under *directory*
    (defaults to ``INCOMING_DIR``).
    """
    directory = directory or INCOMING_DIR
    files: List[Path] = []
    if not directory.is_dir():
        return files
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(p)
    return files
