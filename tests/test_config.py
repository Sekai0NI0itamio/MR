"""Tests for mr.config."""

from __future__ import annotations

from pathlib import Path

import mr.config as cfg


class TestConfig:
    def test_ensure_dirs_creates_directories(self, tmp_path: Path):
        """ensure_dirs() should create all expected directories."""
        cfg.ensure_dirs()
        assert cfg.INCOMING_DIR.is_dir()
        assert cfg.RESULTS_DIR.is_dir()
        assert cfg.LOGS_DIR.is_dir()
        assert cfg.DEBUG_DIR.is_dir()
        assert cfg.DB_DIR.is_dir()

    def test_supported_extensions(self):
        assert ".mp3" in cfg.SUPPORTED_EXTENSIONS
        assert ".m4a" in cfg.SUPPORTED_EXTENSIONS
        assert ".ogg" in cfg.SUPPORTED_EXTENSIONS
        assert ".wav" not in cfg.SUPPORTED_EXTENSIONS

    def test_default_top_k(self):
        assert cfg.TOP_K == 20

    def test_default_max_workers(self):
        assert cfg.MAX_WORKERS >= 1
