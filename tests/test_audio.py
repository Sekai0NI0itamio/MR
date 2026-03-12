"""Tests for mr.audio (with mocked ffmpeg / fpcalc)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mr.audio import (
    AudioProcessingError,
    generate_fingerprint,
    normalize_audio,
    process_audio_file,
)
import mr.config as cfg


class TestNormalizeAudio:
    def test_success(self, tmp_path: Path):
        """normalize_audio should call ffmpeg and return the output path."""
        input_file = tmp_path / "incoming" / "song.mp3"
        input_file.parent.mkdir(parents=True, exist_ok=True)
        input_file.write_bytes(b"fake-audio-data")

        # Fake a successful ffmpeg run
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with patch("mr.audio.subprocess.run", return_value=mock_result) as mock_run:
            # Pre-create the output file so the stat() call doesn't fail
            expected_out = cfg.DEBUG_DIR / "song_normalised.wav"
            expected_out.parent.mkdir(parents=True, exist_ok=True)
            expected_out.write_bytes(b"fake-wav")

            result = normalize_audio(input_file)

        assert result == expected_out
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"

    def test_failure_raises(self, tmp_path: Path):
        """normalize_audio should raise AudioProcessingError on ffmpeg failure."""
        input_file = tmp_path / "incoming" / "bad.mp3"
        input_file.parent.mkdir(parents=True, exist_ok=True)
        input_file.write_bytes(b"bad")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Invalid data found"

        with patch("mr.audio.subprocess.run", return_value=mock_result):
            with pytest.raises(AudioProcessingError, match="ffmpeg failed"):
                normalize_audio(input_file)


class TestGenerateFingerprint:
    def test_success(self, tmp_path: Path):
        """generate_fingerprint should parse fpcalc JSON output."""
        audio_file = tmp_path / "debug" / "song_normalised.wav"
        audio_file.parent.mkdir(parents=True, exist_ok=True)
        audio_file.write_bytes(b"fake")

        fpcalc_output = json.dumps({
            "duration": 180.5,
            "fingerprint": [100, 200, 300, 400],
        })

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fpcalc_output
        mock_result.stderr = ""

        with patch("mr.audio.subprocess.run", return_value=mock_result):
            duration, fp = generate_fingerprint(audio_file)

        assert duration == 180.5
        assert fp == [100, 200, 300, 400]

    def test_failure_raises(self, tmp_path: Path):
        """generate_fingerprint should raise on fpcalc failure."""
        audio_file = tmp_path / "debug" / "bad.wav"
        audio_file.parent.mkdir(parents=True, exist_ok=True)
        audio_file.write_bytes(b"bad")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "couldn't open"

        with patch("mr.audio.subprocess.run", return_value=mock_result):
            with pytest.raises(AudioProcessingError, match="fpcalc failed"):
                generate_fingerprint(audio_file)


class TestProcessAudioFile:
    def test_calls_normalize_and_fingerprint(self, tmp_path: Path):
        """process_audio_file should chain normalise → fingerprint."""
        input_file = tmp_path / "incoming" / "track.ogg"
        input_file.parent.mkdir(parents=True, exist_ok=True)
        input_file.write_bytes(b"data")

        wav_path = tmp_path / "debug" / "track_normalised.wav"

        with patch("mr.audio.normalize_audio", return_value=wav_path) as mock_norm, \
             patch("mr.audio.generate_fingerprint", return_value=(120.0, [1, 2, 3])) as mock_fp:
            result_wav, dur, fp = process_audio_file(input_file)

        assert result_wav == wav_path
        assert dur == 120.0
        assert fp == [1, 2, 3]
        mock_norm.assert_called_once_with(input_file)
        mock_fp.assert_called_once_with(wav_path)
