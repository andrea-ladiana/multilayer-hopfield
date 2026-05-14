"""
Archetype construction, Gram diagnostics, and spectral sharpening.

Provides the complete pipeline from encoded spin samples to memory-ready
patterns:

1. **Empirical archetypes** — majority-vote binarisation of per-class means.
2. **Gram matrix** — normalised overlap matrix and basin-collapse index.
3. **Spectral sharpening** — Kanter–Sompolinsky pseudo-inverse decorrelation.
4. **Quality estimation** — mean per-class Mattis overlap *r*.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

_Array = np.ndarray


def compute_empirical_archetypes(
    spin_groups: _Array | list[_Array],
    rng: np.random.Generator | None = None,
) -> _Array:
    """
    Majority-vote archetypes: sign of the per-class spin mean.

    Parameters
    ----------
    spin_groups : (K, M, N) array or list of (M_k, N) arrays
        Spin configurations grouped by class.
    rng : random generator for tie-breaking.

    Returns
    -------
    (K, N) array with values in {−1, +1}.
    """
    rng = rng or np.random.default_rng()

    if isinstance(spin_groups, np.ndarray) and spin_groups.ndim == 3:
        mean_spins = spin_groups.mean(axis=1)
    else:
        mean_spins = np.stack([g.mean(axis=0) for g in spin_groups])

    archetypes = np.sign(mean_spins)
    zeros = archetypes == 0.0
    if np.any(zeros):
        archetypes[zeros] = rng.choice([-1.0, 1.0], size=int(zeros.sum()))
    return archetypes


def gram_matrix(archetypes: _Array) -> _Array:
    """
    Normalised Gram matrix: G[μ,ν] = (1/N) ξ^μ · ξ^ν.

    Parameters
    ----------
    archetypes : (K, N) array.

    Returns
    -------
    (K, K) symmetric positive-semidefinite matrix.
    """
    _, N = archetypes.shape
    return (archetypes @ archetypes.T) / N


def basin_collapse_index(archetypes: _Array) -> float:
    """
    Mean absolute off-diagonal Gram element — measures basin separability.

    A value near zero indicates well-separated basins of attraction.
    """
    G = gram_matrix(archetypes)
    K = G.shape[0]
    mask = ~np.eye(K, dtype=bool)
    return float(np.mean(np.abs(G[mask])))


def apply_spectral_sharpening(
    archetypes: _Array,
    gamma: float = 1e-3,
    binarise: bool = False,
    rng: np.random.Generator | None = None,
) -> _Array:
    """
    Decorrelate patterns via the regularised Gram pseudo-inverse.

    Computes P_sharp = (G + γI)⁻¹ P, where G = (1/N) P P^T.  This is the
    pattern-space equivalent of replacing Hebbian with Kanter–Sompolinsky
    pseudo-inverse couplings.

    Parameters
    ----------
    archetypes : (K, N) binary patterns.
    gamma : Tikhonov regularisation parameter.
    binarise : if True, re-project to {−1, +1} via sign().  Note that for
        nearly anti-correlated patterns this flips zero bits and is provably
        ineffective; continuous patterns should be preferred in general.
    rng : for tie-breaking if *binarise* is True.

    Returns
    -------
    (K, N) sharpened patterns (continuous unless *binarise* is set).
    """
    rng = rng or np.random.default_rng()

    P = archetypes.astype(float)
    K, N = P.shape

    G = (P @ P.T) / N
    G_inv = np.linalg.inv(G + gamma * np.eye(K))
    P_sharp = G_inv @ P

    if binarise:
        result = np.sign(P_sharp)
        zeros = result == 0.0
        if np.any(zeros):
            result[zeros] = rng.choice([-1.0, 1.0], size=int(zeros.sum()))
        return result

    return P_sharp


def estimate_r(
    spin_groups: _Array,
    archetypes: _Array,
) -> float:
    """
    Mean per-class Mattis overlap with archetypes (quality parameter *r*).

    Parameters
    ----------
    spin_groups : (K, M, N) array.
    archetypes : (K, N) array.

    Returns
    -------
    r : float in [0, 1].
    """
    K, M, N = spin_groups.shape
    overlaps = np.array([
        (spin_groups[mu] @ archetypes[mu]).mean() / N
        for mu in range(K)
    ])
    return float(overlaps.mean())


def sharpening_flip_count(
    archetypes: _Array,
    gamma: float = 1e-3,
) -> int:
    """
    Number of bits that would flip under sign re-binarisation after sharpening.

    A high count relative to K*N signals that sharpening fundamentally alters
    the pattern structure (typical for nearly anti-correlated archetypes);
    a low or zero count means re-binarisation is harmless.
    """
    P = archetypes.astype(float)
    K, N = P.shape

    G = (P @ P.T) / N
    G_inv = np.linalg.inv(G + gamma * np.eye(K))
    P_sharp = G_inv @ P

    signs_orig = np.sign(P)
    signs_sharp = np.sign(P_sharp)

    flips = (signs_orig != 0.0) & (signs_sharp != 0.0) & (signs_orig != signs_sharp)
    return int(np.sum(flips))
