"""Shared fixtures for MR tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Point ``MR_PROJECT_ROOT`` at a temporary directory so tests never
    touch the real project tree.
    """
    monkeypatch.setenv("MR_PROJECT_ROOT", str(tmp_path))

    # Re-import config so paths pick up the new root.  We do this by
    # patching the module-level constants directly.
    import mr.config as cfg

    monkeypatch.setattr(cfg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cfg, "INCOMING_DIR", tmp_path / "incoming")
    monkeypatch.setattr(cfg, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cfg, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(cfg, "DEBUG_DIR", tmp_path / "debug")
    monkeypatch.setattr(cfg, "DB_DIR", tmp_path / "db")
    monkeypatch.setattr(cfg, "DB_FINGERPRINT_CSV", tmp_path / "db" / "fingerprints.csv")
    monkeypatch.setattr(cfg, "DB_TRACK_CSV", tmp_path / "db" / "tracks.csv")
    monkeypatch.setattr(cfg, "DB_FINGERPRINT_JSONL", tmp_path / "db" / "fingerprints.jsonl.gz")
    monkeypatch.setattr(cfg, "DB_TRACK_JSONL", tmp_path / "db" / "tracks.jsonl.gz")
    monkeypatch.setattr(cfg, "DB_TRACK_META_JSONL", tmp_path / "db" / "track_meta.jsonl.gz")
    monkeypatch.setattr(cfg, "FAISS_INDEX_PATH", tmp_path / "db" / "fingerprints.index")
    monkeypatch.setattr(cfg, "TRACK_METADATA_PATH", tmp_path / "db" / "track_metadata.pkl")

    cfg.ensure_dirs()
