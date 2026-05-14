"""Tests for hopfield.metrics — retrieval diagnostics."""

import numpy as np
import pytest

from hopfield.metrics import (
    mattis_overlap,
    mattis_overlaps,
    overlap_matrix,
    winner_analysis,
    cross_layer_consistency,
    reconstruction_success,
    classification_accuracy,
)


@pytest.fixture
def rng():
    return np.random.default_rng(11111)


# ── Overlaps ─────────────────────────────────────────────────────────

class TestOverlaps:
    def test_perfect_overlap(self, rng):
        N = 100
        xi = np.sign(rng.uniform(-1, 1, size=N))
        assert np.isclose(mattis_overlap(xi, xi), 1.0)

    def test_anti_overlap(self, rng):
        N = 100
        xi = np.sign(rng.uniform(-1, 1, size=N))
        assert np.isclose(mattis_overlap(-xi, xi), -1.0)

    def test_vector_output_shape(self, rng):
        K, N = 4, 64
        P = np.sign(rng.uniform(-1, 1, size=(K, N)))
        s = np.sign(rng.uniform(-1, 1, size=N))
        m = mattis_overlaps(s, P)
        assert m.shape == (K,)

    def test_overlap_matrix_shape(self, rng):
        M, K, N = 20, 5, 128
        samples = np.sign(rng.uniform(-1, 1, size=(M, N)))
        arch = np.sign(rng.uniform(-1, 1, size=(K, N)))
        O = overlap_matrix(samples, arch)
        assert O.shape == (M, K)

    def test_overlap_matrix_diagonal(self, rng):
        """Overlaps of patterns with themselves should be 1."""
        K, N = 3, 200
        P = np.sign(rng.uniform(-1, 1, size=(K, N)))
        O = overlap_matrix(P, P)
        np.testing.assert_allclose(np.diag(O), 1.0, atol=1e-14)


# ── Winner analysis ──────────────────────────────────────────────────

class TestWinnerAnalysis:
    def test_correct_winner(self):
        overlaps = np.array([[0.1, 0.9, 0.3],
                              [0.8, 0.2, 0.5]])
        classes, vals, margins = winner_analysis(overlaps)
        np.testing.assert_array_equal(classes, [1, 0])
        np.testing.assert_allclose(vals, [0.9, 0.8])
        np.testing.assert_allclose(margins, [0.6, 0.3])

    def test_shapes(self, rng):
        M, K = 15, 4
        overlaps = rng.uniform(-1, 1, size=(M, K))
        c, v, m = winner_analysis(overlaps)
        assert c.shape == (M,)
        assert v.shape == (M,)
        assert m.shape == (M,)

    def test_margin_non_negative(self, rng):
        overlaps = rng.uniform(0, 1, size=(50, 6))
        _, _, margin = winner_analysis(overlaps)
        assert np.all(margin >= 0.0)


# ── Cross-layer consistency ──────────────────────────────────────────

class TestCrossLayerConsistency:
    def test_consistent(self):
        mags = {
            "sigma": np.array([0.1, 0.9, 0.3]),
            "tau": np.array([0.2, 0.8, 0.1]),
            "phi": np.array([0.0, 0.7, 0.5]),
        }
        ok, winners = cross_layer_consistency(mags)
        assert ok
        assert all(w == 1 for w in winners.values())

    def test_inconsistent(self):
        mags = {
            "sigma": np.array([0.9, 0.1]),
            "tau": np.array([0.1, 0.9]),
            "phi": np.array([0.9, 0.1]),
        }
        ok, winners = cross_layer_consistency(mags)
        assert not ok


# ── Reconstruction success ───────────────────────────────────────────

class TestReconstructionSuccess:
    def test_successful_reconstruction(self):
        mags = {
            "sigma": np.array([0.1, 0.85, 0.0]),
            "tau": np.array([0.05, 0.9, 0.02]),
            "phi": np.array([0.0, 0.8, 0.1]),
        }
        result = reconstruction_success(mags, target=1)
        assert result["success"]
        assert result["consistent"]

    def test_wrong_target(self):
        mags = {
            "sigma": np.array([0.9, 0.1]),
            "tau": np.array([0.8, 0.2]),
            "phi": np.array([0.7, 0.3]),
        }
        result = reconstruction_success(mags, target=1)
        assert not result["success"]

    def test_low_overlap_fails(self):
        mags = {
            "sigma": np.array([0.3, 0.3]),
            "tau": np.array([0.3, 0.3]),
            "phi": np.array([0.3, 0.3]),
        }
        result = reconstruction_success(mags, target=0, overlap_threshold=0.5)
        assert not result["success"]


# ── Classification accuracy ──────────────────────────────────────────

class TestClassificationAccuracy:
    def test_perfect(self):
        y = np.array([0, 1, 2, 1, 0])
        assert classification_accuracy(y, y) == 1.0

    def test_half(self):
        pred = np.array([0, 0, 1, 1])
        true = np.array([0, 0, 0, 1])
        assert np.isclose(classification_accuracy(pred, true), 0.75)
