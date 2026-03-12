"""Tests for mr.db (fingerprint vector utilities)."""

from __future__ import annotations

import numpy as np

from mr.db import fingerprint_to_vector


class TestFingerprintToVector:
    def test_exact_length(self):
        raw = list(range(120))
        vec = fingerprint_to_vector(raw, dim=120)
        assert vec.shape == (120,)
        assert vec.dtype == np.float32

    def test_truncation(self):
        raw = list(range(200))
        vec = fingerprint_to_vector(raw, dim=120)
        assert vec.shape == (120,)
        # Should keep first 120 values
        assert vec[0] == 0.0
        assert vec[119] == 119.0

    def test_padding(self):
        raw = [1, 2, 3]
        vec = fingerprint_to_vector(raw, dim=10)
        assert vec.shape == (10,)
        assert vec[0] == 1.0
        assert vec[2] == 3.0
        assert vec[3] == 0.0  # padded
        assert vec[9] == 0.0

    def test_empty_input(self):
        vec = fingerprint_to_vector([], dim=5)
        assert vec.shape == (5,)
        assert all(v == 0.0 for v in vec)
