"""
Retrieval diagnostics for hetero-associative memory evaluation.

All metrics are retrieval-centric: overlaps, margins, cross-layer consistency.
Classification accuracy is provided as a secondary sanity check.
"""

from __future__ import annotations

import numpy as np
from typing import Any

_Array = np.ndarray


def mattis_overlap(spins: _Array, pattern: _Array) -> float:
    """Scalar Mattis overlap: m = (1/N) sigma . xi."""
    return float(np.dot(spins, pattern) / spins.size)


def mattis_overlaps(spins: _Array, patterns: _Array) -> _Array:
    """Mattis overlap vector between one spin config and K patterns."""
    return patterns @ spins / float(spins.size)


def overlap_matrix(samples: _Array, archetypes: _Array) -> _Array:
    """Full overlap matrix (M, K) between M samples and K archetypes."""
    return (samples @ archetypes.T) / samples.shape[1]


def winner_analysis(overlaps: _Array) -> tuple[_Array, _Array, _Array]:
    """
    Per-sample winner class, overlap, and margin from (M, K) overlaps.

    Returns (winner_class, winner_overlap, winner_margin).
    """
    sorted_ov = np.sort(overlaps, axis=1)
    winner_class = np.argmax(overlaps, axis=1)
    winner_overlap = sorted_ov[:, -1]
    margin = sorted_ov[:, -1] - sorted_ov[:, -2]
    return winner_class, winner_overlap, margin


def cross_layer_consistency(
    final_mags: dict[str, _Array],
) -> tuple[bool, dict[str, int]]:
    """Check whether all three layers converge to the same winner."""
    winners = {
        name: int(np.argmax(final_mags[name]))
        for name in ("sigma", "tau", "phi")
    }
    vals = list(winners.values())
    return all(v == vals[0] for v in vals), winners


def reconstruction_success(
    final_mags: dict[str, _Array],
    target: int,
    overlap_threshold: float = 0.5,
    margin_threshold: float = 0.1,
) -> dict[str, Any]:
    """
    Evaluate whether pattern reconstruction succeeded.

    Success requires cross-layer consistency, correct winner, sufficient
    overlap and margin in every layer.
    """
    layers = ("sigma", "tau", "phi")
    winners, target_overlaps, margins = {}, {}, {}

    for layer in layers:
        m = final_mags[layer]
        w = int(np.argmax(m))
        winners[layer] = w
        target_overlaps[layer] = float(m[target])
        s = np.sort(m)
        margins[layer] = float(s[-1] - s[-2])

    vals = list(winners.values())
    consistent = all(v == vals[0] for v in vals)
    success = (
        consistent
        and vals[0] == target
        and all(target_overlaps[l] >= overlap_threshold for l in layers)
        and all(margins[l] >= margin_threshold for l in layers)
    )
    return {
        "success": success, "consistent": consistent, "winners": winners,
        "target_overlap": target_overlaps, "winner_margin": margins,
    }


def classification_accuracy(predicted: _Array, truth: _Array) -> float:
    """Simple accuracy between predicted and true label arrays."""
    return float((np.asarray(predicted) == np.asarray(truth)).mean())
