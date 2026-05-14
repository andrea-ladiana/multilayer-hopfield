"""Tests for hopfield.encoding — Ising spin encoders."""

import numpy as np
import pytest

from hopfield.encoding import IsingEncoder, STFTIsingEncoder, LDAIsingEncoder


@pytest.fixture
def rng():
    return np.random.default_rng(54321)


# ── IsingEncoder ─────────────────────────────────────────────────────

class TestIsingEncoder:
    def test_deterministic_with_seed(self):
        X = np.random.RandomState(0).randn(50, 20)
        S1 = IsingEncoder(n_components=10, n_spins=100, random_state=42).fit_transform(X)
        S2 = IsingEncoder(n_components=10, n_spins=100, random_state=42).fit_transform(X)
        np.testing.assert_array_equal(S1, S2)

    def test_output_is_binary(self, rng):
        X = rng.standard_normal((30, 15))
        enc = IsingEncoder(n_spins=64, random_state=0)
        S = enc.fit_transform(X)
        assert set(np.unique(S)).issubset({-1.0, 1.0})

    def test_output_shape(self, rng):
        n, d, n_spins = 40, 12, 128
        X = rng.standard_normal((n, d))
        enc = IsingEncoder(n_spins=n_spins, random_state=0)
        S = enc.fit_transform(X)
        assert S.shape == (n, n_spins)

    def test_transform_before_fit_raises(self):
        enc = IsingEncoder(n_spins=32)
        with pytest.raises(RuntimeError, match="not fitted"):
            enc.transform(np.zeros((5, 10)))

    def test_n_components_caps_rank(self, rng):
        X = rng.standard_normal((30, 5))
        enc = IsingEncoder(n_components=100, n_spins=32, random_state=0)
        enc.fit(X)
        # Effective components should be <= min(n_samples, n_features)
        assert enc._components.shape[0] <= min(30, 5)

    def test_different_seeds_differ(self, rng):
        X = rng.standard_normal((40, 10))
        S1 = IsingEncoder(n_spins=64, random_state=0).fit_transform(X)
        S2 = IsingEncoder(n_spins=64, random_state=1).fit_transform(X)
        assert not np.array_equal(S1, S2)

    def test_locality_sensitive(self, rng):
        """Nearby points should produce more similar hashes than distant ones."""
        d = 20
        # Two tight clusters far apart
        centre_a = rng.standard_normal(d) * 5.0
        centre_b = -centre_a  # maximally separated
        X_a = np.vstack([centre_a + rng.standard_normal(d) * 0.1 for _ in range(30)])
        X_b = np.vstack([centre_b + rng.standard_normal(d) * 0.1 for _ in range(30)])
        X = np.vstack([X_a, X_b])

        enc = IsingEncoder(n_spins=512, random_state=0)
        S = enc.fit_transform(X)

        # Mean normalised Hamming within cluster A vs across clusters
        n_spins = 512
        ham_intra = np.mean([
            np.sum(S[i] != S[j]) / n_spins
            for i in range(30) for j in range(i + 1, 30)
        ])
        ham_inter = np.mean([
            np.sum(S[i] != S[j]) / n_spins
            for i in range(30) for j in range(30, 60)
        ])
        assert ham_intra < ham_inter


# ── STFTIsingEncoder ─────────────────────────────────────────────────

class TestSTFTIsingEncoder:
    def test_output_is_binary(self, rng):
        T, D = 50, 3
        X = rng.standard_normal((20, T * D))
        enc = STFTIsingEncoder(
            n_spins=64, random_state=0,
            nperseg=10, noverlap=5, time_steps=T, channels=D,
        )
        S = enc.fit_transform(X)
        assert set(np.unique(S)).issubset({-1.0, 1.0})

    def test_output_shape(self, rng):
        T, D, n = 40, 3, 15
        X = rng.standard_normal((n, T * D))
        enc = STFTIsingEncoder(
            n_spins=128, random_state=0,
            nperseg=10, noverlap=5, time_steps=T, channels=D,
        )
        S = enc.fit_transform(X)
        assert S.shape == (n, 128)

    def test_deterministic(self, rng):
        T, D = 30, 3
        X = rng.standard_normal((10, T * D))
        enc1 = STFTIsingEncoder(
            n_spins=64, random_state=7,
            nperseg=10, noverlap=5, time_steps=T, channels=D,
        )
        enc2 = STFTIsingEncoder(
            n_spins=64, random_state=7,
            nperseg=10, noverlap=5, time_steps=T, channels=D,
        )
        np.testing.assert_array_equal(enc1.fit_transform(X), enc2.fit_transform(X))


# ── LDAIsingEncoder ──────────────────────────────────────────────────

class TestLDAIsingEncoder:
    def _make_labelled_data(self, rng, K=3, n=60, d=20):
        y = np.repeat(np.arange(K), n // K)
        X = np.empty((len(y), d))
        for c in range(K):
            mu = rng.standard_normal(d) * 3
            X[y == c] = mu + rng.standard_normal((np.sum(y == c), d)) * 0.5
        return X, y

    def test_output_is_binary(self, rng):
        X, y = self._make_labelled_data(rng)
        enc = LDAIsingEncoder(n_spins=64, random_state=0)
        S = enc.fit_transform(X, y)
        assert set(np.unique(S)).issubset({-1.0, 1.0})

    def test_output_shape(self, rng):
        X, y = self._make_labelled_data(rng, K=4, n=80, d=15)
        enc = LDAIsingEncoder(n_spins=128, random_state=0)
        S = enc.fit_transform(X, y)
        assert S.shape == (80, 128)

    def test_transform_before_fit_raises(self):
        enc = LDAIsingEncoder(n_spins=32)
        with pytest.raises(RuntimeError, match="not fitted"):
            enc.transform(np.zeros((5, 10)))

    def test_deterministic(self, rng):
        X, y = self._make_labelled_data(rng)
        S1 = LDAIsingEncoder(n_spins=64, random_state=0).fit_transform(X, y)
        S2 = LDAIsingEncoder(n_spins=64, random_state=0).fit_transform(X, y)
        np.testing.assert_array_equal(S1, S2)

    def test_classes_are_separable(self, rng):
        """With well-separated classes, same-class Hamming should be low."""
        X, y = self._make_labelled_data(rng, K=3, n=60, d=20)
        enc = LDAIsingEncoder(n_spins=256, random_state=0, n_pca_extra=4)
        S = enc.fit_transform(X, y)

        ham_intra, ham_inter = [], []
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                h = np.sum(S[i] != S[j]) / 256
                if y[i] == y[j]:
                    ham_intra.append(h)
                else:
                    ham_inter.append(h)
        assert np.mean(ham_intra) < np.mean(ham_inter)
