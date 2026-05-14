"""Tests for hopfield.patterns — archetype construction and spectral sharpening."""

import numpy as np
import pytest

from hopfield.patterns import (
    compute_empirical_archetypes,
    gram_matrix,
    basin_collapse_index,
    apply_spectral_sharpening,
    estimate_r,
    sharpening_flip_count,
)


@pytest.fixture
def rng():
    return np.random.default_rng(9876)


def _make_spin_groups(K, M, N, rng):
    """K classes, M samples each, N spins — samples clustered around archetypes."""
    archetypes = np.sign(rng.uniform(-1, 1, size=(K, N)))
    groups = np.empty((K, M, N))
    for mu in range(K):
        for s in range(M):
            mask = rng.random(N) < 0.1  # 10% flip rate
            groups[mu, s] = archetypes[mu].copy()
            groups[mu, s, mask] *= -1
    return groups, archetypes


# ── Empirical archetypes ─────────────────────────────────────────────

class TestEmpiricalArchetypes:
    def test_binary_output(self, rng):
        groups, _ = _make_spin_groups(3, 50, 64, rng)
        arch = compute_empirical_archetypes(groups, rng=rng)
        assert set(np.unique(arch)).issubset({-1.0, 1.0})

    def test_shape(self, rng):
        groups, _ = _make_spin_groups(4, 30, 128, rng)
        arch = compute_empirical_archetypes(groups, rng=rng)
        assert arch.shape == (4, 128)

    def test_close_to_true_archetypes(self, rng):
        groups, true_arch = _make_spin_groups(3, 200, 64, rng)
        emp_arch = compute_empirical_archetypes(groups, rng=rng)
        for mu in range(3):
            overlap = np.dot(emp_arch[mu], true_arch[mu]) / 64
            assert overlap > 0.7  # Should be close with M=200 and 10% noise

    def test_list_input(self, rng):
        """Accepts a list of variably-sized arrays."""
        groups = [
            np.sign(rng.uniform(-1, 1, size=(50, 32))),
            np.sign(rng.uniform(-1, 1, size=(80, 32))),
        ]
        arch = compute_empirical_archetypes(groups, rng=rng)
        assert arch.shape == (2, 32)


# ── Gram matrix ──────────────────────────────────────────────────────

class TestGramMatrix:
    def test_diagonal_is_one_for_binary(self, rng):
        K, N = 3, 100
        P = np.sign(rng.uniform(-1, 1, size=(K, N)))
        G = gram_matrix(P)
        np.testing.assert_allclose(np.diag(G), 1.0, atol=1e-14)

    def test_symmetric(self, rng):
        K, N = 4, 64
        P = np.sign(rng.uniform(-1, 1, size=(K, N)))
        G = gram_matrix(P)
        np.testing.assert_allclose(G, G.T, atol=1e-14)

    def test_shape(self, rng):
        K, N = 5, 128
        P = np.sign(rng.uniform(-1, 1, size=(K, N)))
        G = gram_matrix(P)
        assert G.shape == (K, K)

    def test_orthogonal_patterns(self):
        """Orthogonal patterns produce identity Gram."""
        P = np.array([[1, 1, -1, -1],
                       [1, -1, 1, -1]], dtype=float)
        G = gram_matrix(P)
        np.testing.assert_allclose(G, np.eye(2), atol=1e-14)


# ── Basin collapse index ─────────────────────────────────────────────

class TestBasinCollapseIndex:
    def test_orthogonal_is_zero(self):
        P = np.array([[1, 1, -1, -1],
                       [1, -1, 1, -1]], dtype=float)
        assert np.isclose(basin_collapse_index(P), 0.0)

    def test_identical_is_one(self):
        P = np.array([[1, 1, 1, 1],
                       [1, 1, 1, 1]], dtype=float)
        assert np.isclose(basin_collapse_index(P), 1.0)

    def test_bounded(self, rng):
        P = np.sign(rng.uniform(-1, 1, size=(5, 200)))
        bci = basin_collapse_index(P)
        assert 0.0 <= bci <= 1.0


# ── Spectral sharpening ─────────────────────────────────────────────

class TestSpectralSharpening:
    def test_reduces_off_diagonal_gram(self, rng):
        K, N = 3, 256
        P = np.sign(rng.uniform(-1, 1, size=(K, N)))
        bci_before = basin_collapse_index(P)
        P_sharp = apply_spectral_sharpening(P, gamma=1e-3)
        G_sharp = (P_sharp @ P_sharp.T) / N
        bci_after = float(np.mean(np.abs(G_sharp[~np.eye(K, dtype=bool)])))
        assert bci_after < bci_before

    def test_continuous_output(self, rng):
        P = np.sign(rng.uniform(-1, 1, size=(3, 64)))
        P_sharp = apply_spectral_sharpening(P, gamma=1e-3, binarise=False)
        # Continuous patterns are generally not in {-1, +1}
        assert not np.all(np.isin(P_sharp, [-1.0, 1.0]))

    def test_binarise_option(self, rng):
        P = np.sign(rng.uniform(-1, 1, size=(3, 64)))
        P_bin = apply_spectral_sharpening(P, gamma=1e-3, binarise=True, rng=rng)
        assert set(np.unique(P_bin)).issubset({-1.0, 1.0})

    def test_identity_for_orthogonal(self):
        """Sharpening orthogonal patterns should approximately preserve them."""
        P = np.array([[1, 1, -1, -1],
                       [1, -1, 1, -1]], dtype=float)
        P_sharp = apply_spectral_sharpening(P, gamma=1e-6, binarise=True)
        np.testing.assert_allclose(P_sharp, P, atol=1e-6)

    def test_shape_preserved(self, rng):
        K, N = 4, 128
        P = np.sign(rng.uniform(-1, 1, size=(K, N)))
        P_sharp = apply_spectral_sharpening(P, gamma=1e-3)
        assert P_sharp.shape == (K, N)


# ── Quality parameter r ──────────────────────────────────────────────

class TestEstimateR:
    def test_perfect_alignment(self, rng):
        K, M, N = 3, 20, 64
        arch = np.sign(rng.uniform(-1, 1, size=(K, N)))
        groups = np.tile(arch[:, np.newaxis, :], (1, M, 1))
        r = estimate_r(groups, arch)
        assert np.isclose(r, 1.0)

    def test_random_is_near_zero(self, rng):
        K, M, N = 3, 100, 2048
        arch = np.sign(rng.uniform(-1, 1, size=(K, N)))
        groups = np.sign(rng.uniform(-1, 1, size=(K, M, N)))
        r = estimate_r(groups, arch)
        assert abs(r) < 0.1

    def test_bounded(self, rng):
        groups, arch = _make_spin_groups(3, 50, 64, rng)
        r = estimate_r(groups, arch)
        assert -1.0 <= r <= 1.0


# ── Flip count ───────────────────────────────────────────────────────

class TestFlipCount:
    def test_zero_for_orthogonal(self):
        P = np.array([[1, 1, -1, -1],
                       [1, -1, 1, -1]], dtype=float)
        assert sharpening_flip_count(P, gamma=1e-6) == 0

    def test_non_negative(self, rng):
        P = np.sign(rng.uniform(-1, 1, size=(5, 64)))
        assert sharpening_flip_count(P) >= 0
