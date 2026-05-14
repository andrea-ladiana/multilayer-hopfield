"""Tests for hopfield.datasets — synthetic data generation."""

import numpy as np
import pytest

from hopfield.datasets import (
    curve_helix,
    curve_lissajous,
    curve_scurve,
    generate_curve_family,
    generate_archetypes,
    simulate_3d_curve_dataset,
)


@pytest.fixture
def t_grid():
    return np.linspace(0, 1, 50, endpoint=True)


# ── Curve functions ──────────────────────────────────────────────────

class TestCurveFunctions:
    @pytest.mark.parametrize("curve_fn", [curve_helix, curve_lissajous, curve_scurve])
    def test_output_shape(self, curve_fn, t_grid):
        pts = curve_fn(t_grid)
        assert pts.shape == (50, 3)

    @pytest.mark.parametrize("curve_fn", [curve_helix, curve_lissajous, curve_scurve])
    def test_no_nans(self, curve_fn, t_grid):
        pts = curve_fn(t_grid)
        assert not np.any(np.isnan(pts))

    def test_curves_are_distinct(self, t_grid):
        h = curve_helix(t_grid)
        l = curve_lissajous(t_grid)
        s = curve_scurve(t_grid)
        # No two curves should be identical
        assert not np.allclose(h, l)
        assert not np.allclose(h, s)
        assert not np.allclose(l, s)


# ── Curve family ─────────────────────────────────────────────────────

class TestCurveFamily:
    def test_returns_K_curves(self):
        for K in [1, 2, 3, 5, 10]:
            curves = generate_curve_family(K)
            assert len(curves) == K

    def test_first_three_are_canonical(self):
        curves = generate_curve_family(5)
        assert curves[0] is curve_helix
        assert curves[1] is curve_lissajous
        assert curves[2] is curve_scurve

    def test_deterministic(self):
        c1 = generate_curve_family(6, seed=0)
        c2 = generate_curve_family(6, seed=0)
        t = np.linspace(0, 1, 20)
        for f1, f2 in zip(c1[3:], c2[3:]):
            np.testing.assert_array_equal(f1(t), f2(t))


# ── Archetype generation ─────────────────────────────────────────────

class TestGenerateArchetypes:
    def test_shapes(self):
        t, arch = generate_archetypes(T=80, K=4)
        assert t.shape == (80,)
        assert arch.shape == (4, 80, 3)

    def test_t_grid_bounds(self):
        t, _ = generate_archetypes(T=100)
        assert np.isclose(t[0], 0.0)
        assert np.isclose(t[-1], 1.0)


# ── Full dataset ─────────────────────────────────────────────────────

class TestSimulateDataset:
    def test_shapes(self):
        X_tr, y_tr, X_te, y_te, t, arch = simulate_3d_curve_dataset(
            T=30, K=3, n_train=20, n_test=10,
        )
        assert X_tr.shape == (60, 90)    # 3 * 20, 3 * 30
        assert y_tr.shape == (60,)
        assert X_te.shape == (30, 90)
        assert y_te.shape == (30,)
        assert t.shape == (30,)
        assert arch.shape == (3, 30, 3)

    def test_labels_balanced(self):
        _, y_tr, _, y_te, _, _ = simulate_3d_curve_dataset(
            T=20, K=4, n_train=15, n_test=5,
        )
        for k in range(4):
            assert np.sum(y_tr == k) == 15
            assert np.sum(y_te == k) == 5

    def test_deterministic(self):
        d1 = simulate_3d_curve_dataset(T=20, K=3, n_train=10, n_test=5, seed=0)
        d2 = simulate_3d_curve_dataset(T=20, K=3, n_train=10, n_test=5, seed=0)
        np.testing.assert_array_equal(d1[0], d2[0])
        np.testing.assert_array_equal(d1[1], d2[1])

    def test_different_seeds_differ(self):
        d1 = simulate_3d_curve_dataset(T=20, K=3, n_train=10, seed=0)
        d2 = simulate_3d_curve_dataset(T=20, K=3, n_train=10, seed=1)
        assert not np.array_equal(d1[0], d2[0])

    def test_no_nans(self):
        X_tr, _, X_te, _, _, _ = simulate_3d_curve_dataset(T=20, K=3, n_train=10)
        assert not np.any(np.isnan(X_tr))
        assert not np.any(np.isnan(X_te))
