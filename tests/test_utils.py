"""Tests for mr.utils (file discovery, logging)."""

from __future__ import annotations

from pathlib import Path

import mr.config as cfg
from mr.utils import discover_audio_files, setup_root_logger


class TestDiscoverAudioFiles:
    def test_finds_supported_extensions(self, tmp_path: Path):
        incoming = cfg.INCOMING_DIR
        incoming.mkdir(parents=True, exist_ok=True)

        (incoming / "song1.mp3").write_bytes(b"a")
        (incoming / "song2.m4a").write_bytes(b"b")
        (incoming / "song3.ogg").write_bytes(b"c")
        (incoming / "readme.txt").write_bytes(b"d")  # ignored
        (incoming / "image.png").write_bytes(b"e")    # ignored

        files = discover_audio_files(incoming)
        names = [f.name for f in files]

        assert "song1.mp3" in names
        assert "song2.m4a" in names
        assert "song3.ogg" in names
        assert "readme.txt" not in names
        assert "image.png" not in names

    def test_returns_sorted(self, tmp_path: Path):
        incoming = cfg.INCOMING_DIR
        incoming.mkdir(parents=True, exist_ok=True)

        (incoming / "z_song.mp3").write_bytes(b"z")
        (incoming / "a_song.mp3").write_bytes(b"a")

        files = discover_audio_files(incoming)
        assert files[0].name == "a_song.mp3"
        assert files[1].name == "z_song.mp3"

    def test_empty_directory(self, tmp_path: Path):
        incoming = cfg.INCOMING_DIR
        incoming.mkdir(parents=True, exist_ok=True)
        assert discover_audio_files(incoming) == []

    def test_missing_directory(self, tmp_path: Path):
        missing = tmp_path / "nonexistent"
        assert discover_audio_files(missing) == []


class TestSetupRootLogger:
    def test_returns_logger(self):
        logger = setup_root_logger()
        assert logger.name == "mr"
        assert len(logger.handlers) >= 1
