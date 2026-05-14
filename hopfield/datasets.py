"""
Synthetic datasets for benchmarking the hetero-associative memory.

Primary dataset: 3D space curves (helix, Lissajous, S-curve).
Extended family: parametric curves for capacity scaling (K > 3).
"""

from __future__ import annotations

import numpy as np
from typing import Callable

_Array = np.ndarray


# ── Canonical curve archetypes ───────────────────────────────────────

def curve_helix(t: _Array) -> _Array:
    """Helix winding around the z-axis."""
    return np.stack([np.cos(2.0 * np.pi * t),
                     np.sin(2.0 * np.pi * t), t], axis=1)


def curve_lissajous(t: _Array) -> _Array:
    """3D Lissajous figure."""
    return np.stack([1.5 * np.sin(2.0 * np.pi * t),
                     0.5 * np.sin(4.0 * np.pi * t),
                     np.cos(2.0 * np.pi * t)], axis=1)


def curve_scurve(t: _Array) -> _Array:
    """Polynomial S-curve with sinusoidal z-oscillation."""
    u = 2.0 * t - 1.0
    return np.stack([u, u ** 3, 0.5 * np.sin(3.0 * np.pi * t)], axis=1)


# ── Parametric curve family ──────────────────────────────────────────

def _make_parametric_curve(
    a: float, b: float, c: float,
    fx: float, fy: float, fz: float,
    px: float = 0.0, py: float = 0.0, pz: float = 0.0,
) -> Callable[[_Array], _Array]:
    """
    Factory for 3D Lissajous-like parametric curves::

        x(t) = a sin(fx 2pi t + px)
        y(t) = b sin(fy 2pi t + py)
        z(t) = c sin(fz 2pi t + pz)
    """
    def curve(t: _Array) -> _Array:
        tau = 2.0 * np.pi * t
        return np.stack([a * np.sin(fx * tau + px),
                         b * np.sin(fy * tau + py),
                         c * np.sin(fz * tau + pz)], axis=1)
    return curve


def generate_curve_family(K: int, seed: int = 0) -> list[Callable]:
    """
    K distinct 3D curves.  The first three are the canonical archetypes;
    additional curves are randomly parametrised.
    """
    curves: list[Callable] = [curve_helix, curve_lissajous, curve_scurve]
    if K <= 3:
        return curves[:K]

    rng = np.random.default_rng(seed)
    for _ in range(K - 3):
        a = rng.uniform(0.5, 2.0)
        b = rng.uniform(0.3, 1.5)
        c = rng.uniform(0.3, 1.5)
        fx = float(rng.choice([1, 2, 3, 4]))
        fy = float(rng.choice([1, 2, 3, 4, 5]))
        fz = float(rng.choice([1, 2, 3]))
        px, py, pz = rng.uniform(0, 2 * np.pi, size=3)
        curves.append(_make_parametric_curve(a, b, c, fx, fy, fz, px, py, pz))
    return curves


# ── Dataset generation ───────────────────────────────────────────────

def generate_archetypes(
    T: int, K: int = 3, seed: int = 0,
) -> tuple[_Array, _Array]:
    """
    Generate K archetype curves as 3D time series.

    Returns (t_grid (T,), archetypes (K, T, 3)).
    """
    t_grid = np.linspace(0.0, 1.0, T, endpoint=True)
    curves = generate_curve_family(K, seed=seed)
    archetypes = np.stack([c(t_grid) for c in curves])
    return t_grid, archetypes


def simulate_3d_curve_dataset(
    T: int = 100,
    K: int = 3,
    n_train: int = 200,
    n_test: int = 50,
    r_quality: float = 0.8,
    sigma_near: float = 0.05,
    seed: int = 42,
) -> tuple[_Array, _Array, _Array, _Array, _Array, _Array]:
    """
    Synthetic 3D-curve dataset with train/test split.

    Noise model: each time step is independently drawn as
      - with probability r: Gaussian perturbation of the archetype
      - with probability 1-r: uniform in the global bounding box

    Returns (X_train, y_train, X_test, y_test, t_grid, archetypes).
    X arrays have shape (K*n, 3*T) with the inner dimension ordered as
    (x0, y0, z0, x1, y1, z1, ...).
    """
    rng = np.random.default_rng(seed)
    t_grid, archetypes = generate_archetypes(T, K, seed=seed)

    all_pts = archetypes.reshape(-1, 3)
    bbox_lo = all_pts.min(axis=0) - 0.3
    bbox_hi = all_pts.max(axis=0) + 0.3

    def _build_set(n_per_class: int) -> tuple[_Array, _Array]:
        data = np.empty((K, n_per_class, T, 3))
        for mu in range(K):
            for s in range(n_per_class):
                for ti in range(T):
                    if rng.random() < r_quality:
                        data[mu, s, ti] = (
                            archetypes[mu, ti] + rng.normal(0, sigma_near, 3)
                        )
                    else:
                        data[mu, s, ti] = rng.uniform(bbox_lo, bbox_hi)
        X = data.reshape(K * n_per_class, T * 3)
        y = np.repeat(np.arange(K), n_per_class)
        idx = np.arange(len(y))
        rng.shuffle(idx)
        return X[idx], y[idx]

    X_tr, y_tr = _build_set(n_train)
    X_te, y_te = _build_set(n_test)
    return X_tr, y_tr, X_te, y_te, t_grid, archetypes
