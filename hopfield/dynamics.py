"""
Tri-layer hetero-associative memory dynamics.

Architecture
------------
Three binary visible layers σ, τ, φ ∈ {−1, +1}^N, coupled through K stored
pattern triplets (ξ^μ, η^μ, χ^μ).  Inter-layer coupling strengths g_{στ},
g_{σφ}, g_{τφ} weight the mutual information channels.

Update rule (Glauber-like with uniform dither)::

    s_i^{t+1} = sign( tanh(β h_i^t) + u_i ),   u_i ~ U[−1, 1]

The uniform dither replaces the standard Boltzmann acceptance probability and
produces equivalent equilibrium statistics while avoiding evaluation of the
exponential.

References
----------
Kanter, I. & Sompolinsky, H. (1987).  Associative recall of memory without
errors.  *Phys. Rev. A*, 35(1), 380.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

_Array = np.ndarray


# ── Data containers ──────────────────────────────────────────────────

@dataclass
class Couplings:
    """Inter-layer coupling strengths g_{στ}, g_{σφ}, g_{τφ}."""
    sigma_tau: float
    sigma_phi: float
    tau_phi: float


@dataclass
class Rhos:
    """Effective dataset entropies per layer: ρ = (1 − r²) / (M r²)."""
    sigma: float
    tau: float
    phi: float


@dataclass
class GramInverses:
    """
    Regularised inverse Gram matrices for pseudo-inverse couplings.

    The Gram matrix G = (1/N) Ξ Ξ^T governs pattern cross-talk in Hebbian
    couplings.  Replacing the Hebbian field h = Ξ^T m with h = Ξ^T G⁻¹ m
    yields decorrelated mean-field dynamics  m_{t+1} = f(β m_t), enabling
    winner-take-all selection from symmetric mixture states.
    """
    xi:  _Array   # (K, K)
    eta: _Array   # (K, K)
    chi: _Array   # (K, K)


@dataclass
class TAMState:
    """Instantaneous spin configuration of the three visible layers."""
    sigma: _Array   # (N_σ,)
    tau:   _Array   # (N_τ,)
    phi:   _Array   # (N_φ,)


@dataclass
class TAMPatterns:
    """
    Stored patterns for each layer.

    Attributes
    ----------
    XI, ETA, CHI : array (K, N_l)
        True (binary) archetypes.
    XI_mean, ETA_mean, CHI_mean : array (K, N_l)
        Effective patterns used in local-field evaluation.  These may be the
        raw archetypes or their spectrally sharpened (continuous) counterparts.
    """
    XI:       _Array
    ETA:      _Array
    CHI:      _Array
    XI_mean:  _Array
    ETA_mean: _Array
    CHI_mean: _Array


# ── Magnetisation ────────────────────────────────────────────────────

def mattis_magnetisation(spins: _Array, patterns: _Array) -> _Array:
    """
    Mattis magnetisation vector: m^μ = (1/N) Σ_i ξ^μ_i σ_i.

    Parameters
    ----------
    spins : (N,) array
    patterns : (K, N) array

    Returns
    -------
    (K,) array of per-pattern overlaps.
    """
    return patterns @ spins / float(spins.size)


def rho_from_r(r: float, M: int) -> float:
    """Effective dataset entropy: ρ(r, M) = (1 − r²) / (M r²)."""
    return (1.0 - r * r) / (M * r * r)


# ── Gram inverse ─────────────────────────────────────────────────────

def compute_gram_inverses(
    XI: _Array,
    ETA: _Array,
    CHI: _Array,
    gamma: float = 1e-3,
) -> GramInverses:
    """
    Regularised Gram inverse (G + γI)⁻¹ for each layer.

    Parameters
    ----------
    XI, ETA, CHI : (K, N_l) binary pattern matrices.
    gamma : Tikhonov regularisation strength.
    """
    def _inv(P: _Array) -> _Array:
        K, N = P.shape
        G = (P @ P.T) / N
        return np.linalg.inv(G + gamma * np.eye(K))

    return GramInverses(xi=_inv(XI), eta=_inv(ETA), chi=_inv(CHI))


# ── Local field ──────────────────────────────────────────────────────

def _local_field(
    g_y: float,
    g_z: float,
    x: _Array,
    y: _Array,
    z: _Array,
    W_x: _Array,
    W_y: _Array,
    W_z: _Array,
    rho_x: float,
    rho_y: float,
    rho_z: float,
    G_inv: _Array | None = None,
) -> _Array:
    """
    Local field acting on layer *x*, driven by layers *y* and *z*.

    With Hebbian couplings::

        h_i = Σ_μ W_x[μ,i] · C^μ

    where C^μ encodes the coupling-weighted magnetisation from the two
    partner layers.  When *G_inv* is supplied, the field is pre-multiplied
    by the inverse Gram matrix in pattern space, yielding pseudo-inverse
    (Kanter–Sompolinsky) dynamics.
    """
    m_y = mattis_magnetisation(y, W_y)
    m_z = mattis_magnetisation(z, W_z)

    N_x, N_y, N_z = x.size, y.size, z.size

    drive = (
        m_y * np.sqrt(N_y) * g_y * np.sqrt((1.0 + rho_x) * (1.0 + rho_y))
        + m_z * np.sqrt(N_z) * g_z * np.sqrt((1.0 + rho_x) * (1.0 + rho_z))
    ) / np.sqrt(N_x)

    if G_inv is not None:
        drive = G_inv @ drive

    return np.sum(drive * W_x.T, axis=1)


# ── Stochastic update ───────────────────────────────────────────────

def _apply_update(
    h: _Array,
    beta: float,
    rng: np.random.Generator,
) -> _Array:
    """sign(tanh(β h) + u), u ~ U[−1, 1], with random tie-breaking."""
    u = rng.uniform(-1.0, 1.0, size=h.shape)
    s = np.sign(np.tanh(beta * h) + u)
    zeros = s == 0.0
    if np.any(zeros):
        s[zeros] = rng.choice([-1.0, 1.0], size=int(zeros.sum()))
    return s


# ── State constructors ───────────────────────────────────────────────

def build_cued_state(
    patterns: TAMPatterns,
    mu_cue: int,
    cued_layer: str = "sigma",
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> TAMState:
    """
    Initial state with one layer clamped to a (possibly noisy) archetype.

    Parameters
    ----------
    patterns : stored patterns.
    mu_cue : index of the cued pattern.
    cued_layer : which layer receives the cue ('sigma', 'tau', or 'phi').
    noise : per-spin flip probability (0 = clean cue).
    rng : random generator.
    """
    rng = rng or np.random.default_rng()

    K, N1 = patterns.XI.shape
    _, N2 = patterns.ETA.shape
    _, N3 = patterns.CHI.shape

    def _random_spins(n: int) -> _Array:
        s = np.sign(rng.uniform(-1.0, 1.0, size=n))
        s[s == 0.0] = 1.0
        return s

    def _noisy_copy(pattern: _Array) -> _Array:
        if noise <= 0.0:
            return pattern.copy()
        mask = rng.random(size=pattern.shape) < noise
        out = pattern.copy()
        out[mask] *= -1.0
        return out

    sigma = _random_spins(N1)
    tau = _random_spins(N2)
    phi = _random_spins(N3)

    layer = cued_layer.lower()
    if layer == "sigma":
        sigma = _noisy_copy(patterns.XI[mu_cue])
    elif layer == "tau":
        tau = _noisy_copy(patterns.ETA[mu_cue])
    elif layer == "phi":
        phi = _noisy_copy(patterns.CHI[mu_cue])
    else:
        raise ValueError(f"Unknown cued_layer '{cued_layer}'")

    return TAMState(sigma=sigma, tau=tau, phi=phi)


def build_mixture_state(
    patterns: TAMPatterns,
    indices: tuple[int, ...],
    weights: tuple[float, ...] | None = None,
    rng: np.random.Generator | None = None,
) -> TAMState:
    """
    Symmetric mixture initial state for disentanglement experiments.

    All layers are initialised to sign(Σ_k w_k · pattern_k[μ_k]).
    """
    rng = rng or np.random.default_rng()
    if weights is None:
        weights = tuple(1.0 for _ in indices)

    def _mix(P: _Array) -> _Array:
        v = sum(w * P[mu] for w, mu in zip(weights, indices))
        s = np.sign(v)
        zeros = s == 0.0
        if np.any(zeros):
            s[zeros] = rng.choice([-1.0, 1.0], size=int(zeros.sum()))
        return s

    return TAMState(
        sigma=_mix(patterns.XI),
        tau=_mix(patterns.ETA),
        phi=_mix(patterns.CHI),
    )


# ── Monte Carlo steps ───────────────────────────────────────────────

def mc_step_parallel(
    state: TAMState,
    patterns: TAMPatterns,
    rhos: Rhos,
    couplings: Couplings,
    beta: float,
    rng: np.random.Generator | None = None,
    gram_inverses: GramInverses | None = None,
) -> TAMState:
    """
    Parallel Monte Carlo sweep: all three layers updated simultaneously.

    Each spin is updated according to the local field computed from the
    *current* configuration of the partner layers, so all three layers
    see the same state.
    """
    rng = rng or np.random.default_rng()
    sigma, tau, phi = state.sigma, state.tau, state.phi
    gi = gram_inverses

    h_sigma = _local_field(
        couplings.sigma_tau, couplings.sigma_phi,
        sigma, tau, phi,
        patterns.XI_mean, patterns.ETA_mean, patterns.CHI_mean,
        rhos.sigma, rhos.tau, rhos.phi,
        gi.xi if gi else None,
    )
    h_tau = _local_field(
        couplings.sigma_tau, couplings.tau_phi,
        tau, sigma, phi,
        patterns.ETA_mean, patterns.XI_mean, patterns.CHI_mean,
        rhos.tau, rhos.sigma, rhos.phi,
        gi.eta if gi else None,
    )
    h_phi = _local_field(
        couplings.sigma_phi, couplings.tau_phi,
        phi, tau, sigma,
        patterns.CHI_mean, patterns.ETA_mean, patterns.XI_mean,
        rhos.phi, rhos.tau, rhos.sigma,
        gi.chi if gi else None,
    )

    return TAMState(
        sigma=_apply_update(h_sigma, beta, rng),
        tau=_apply_update(h_tau, beta, rng),
        phi=_apply_update(h_phi, beta, rng),
    )


def mc_step_sequential(
    state: TAMState,
    patterns: TAMPatterns,
    rhos: Rhos,
    couplings: Couplings,
    beta: float,
    rng: np.random.Generator | None = None,
    gram_inverses: GramInverses | None = None,
) -> TAMState:
    """
    Sequential (Glauber-style) Monte Carlo sweep: σ → τ → φ.

    Each layer is updated using the freshly updated partner layers, which
    breaks the symmetry lock that parallel updates impose on mixture states
    and improves disentanglement.
    """
    rng = rng or np.random.default_rng()
    gi = gram_inverses

    sigma = state.sigma.copy()
    tau = state.tau.copy()
    phi = state.phi.copy()

    # σ: depends on current τ, φ
    h_sigma = _local_field(
        couplings.sigma_tau, couplings.sigma_phi,
        sigma, tau, phi,
        patterns.XI_mean, patterns.ETA_mean, patterns.CHI_mean,
        rhos.sigma, rhos.tau, rhos.phi,
        gi.xi if gi else None,
    )
    sigma = _apply_update(h_sigma, beta, rng)

    # τ: depends on updated σ, current φ
    h_tau = _local_field(
        couplings.sigma_tau, couplings.tau_phi,
        tau, sigma, phi,
        patterns.ETA_mean, patterns.XI_mean, patterns.CHI_mean,
        rhos.tau, rhos.sigma, rhos.phi,
        gi.eta if gi else None,
    )
    tau = _apply_update(h_tau, beta, rng)

    # φ: depends on updated σ, updated τ
    h_phi = _local_field(
        couplings.sigma_phi, couplings.tau_phi,
        phi, tau, sigma,
        patterns.CHI_mean, patterns.ETA_mean, patterns.XI_mean,
        rhos.phi, rhos.tau, rhos.sigma,
        gi.chi if gi else None,
    )
    phi = _apply_update(h_phi, beta, rng)

    return TAMState(sigma=sigma, tau=tau, phi=phi)


# ── Trajectories ─────────────────────────────────────────────────────

def _init_random_state(
    patterns: TAMPatterns,
    rng: np.random.Generator,
) -> TAMState:
    """Fully random initial spin configuration."""
    def _rand(n: int) -> _Array:
        s = np.sign(rng.uniform(-1.0, 1.0, size=n))
        s[s == 0.0] = 1.0
        return s

    _, N1 = patterns.XI.shape
    _, N2 = patterns.ETA.shape
    _, N3 = patterns.CHI.shape
    return TAMState(sigma=_rand(N1), tau=_rand(N2), phi=_rand(N3))


def _record_magnetisations(
    state: TAMState,
    patterns: TAMPatterns,
) -> tuple[_Array, _Array, _Array]:
    """Compute Mattis magnetisations against the true archetypes."""
    return (
        mattis_magnetisation(state.sigma, patterns.XI),
        mattis_magnetisation(state.tau, patterns.ETA),
        mattis_magnetisation(state.phi, patterns.CHI),
    )


def run_trajectory(
    patterns: TAMPatterns,
    rhos: Rhos,
    couplings: Couplings,
    beta: float,
    steps: int,
    init_state: TAMState | None = None,
    rng: np.random.Generator | None = None,
    gram_inverses: GramInverses | None = None,
) -> dict[str, object]:
    """
    Constant-β Monte Carlo trajectory with parallel updates.

    Returns
    -------
    dict with keys 'sigma', 'tau', 'phi' (each (steps+1, K) magnetisation
    arrays) and 'final_state'.
    """
    rng = rng or np.random.default_rng()
    K = patterns.XI.shape[0]

    state = init_state or _init_random_state(patterns, rng)

    mag_s = np.empty((steps + 1, K))
    mag_t = np.empty((steps + 1, K))
    mag_p = np.empty((steps + 1, K))

    mag_s[0], mag_t[0], mag_p[0] = _record_magnetisations(state, patterns)

    for i in range(1, steps + 1):
        state = mc_step_parallel(
            state, patterns, rhos, couplings, beta,
            rng=rng, gram_inverses=gram_inverses,
        )
        mag_s[i], mag_t[i], mag_p[i] = _record_magnetisations(state, patterns)

    return {
        "sigma": mag_s,
        "tau": mag_t,
        "phi": mag_p,
        "final_state": state,
    }


def run_trajectory_annealing(
    patterns: TAMPatterns,
    rhos: Rhos,
    couplings: Couplings,
    beta_schedule: _Array,
    init_state: TAMState | None = None,
    rng: np.random.Generator | None = None,
    sequential: bool = False,
    gram_inverses: GramInverses | None = None,
) -> dict[str, object]:
    """
    Simulated-annealing Monte Carlo trajectory.

    Parameters
    ----------
    beta_schedule : (steps,) array
        Inverse temperature at each sweep.  Typically ramps from a low value
        (exploration) to a high value (crystallisation).
    sequential : bool
        If True, use Glauber-style sequential layer updates.

    Returns
    -------
    dict with keys 'sigma', 'tau', 'phi' (magnetisation trajectories),
    'final_state', and 'beta_schedule'.
    """
    rng = rng or np.random.default_rng()
    K = patterns.XI.shape[0]
    steps = len(beta_schedule)

    state = init_state or _init_random_state(patterns, rng)

    mag_s = np.empty((steps + 1, K))
    mag_t = np.empty((steps + 1, K))
    mag_p = np.empty((steps + 1, K))

    mag_s[0], mag_t[0], mag_p[0] = _record_magnetisations(state, patterns)

    update = mc_step_sequential if sequential else mc_step_parallel

    for i in range(steps):
        state = update(
            state, patterns, rhos, couplings, beta_schedule[i],
            rng=rng, gram_inverses=gram_inverses,
        )
        mag_s[i + 1], mag_t[i + 1], mag_p[i + 1] = (
            _record_magnetisations(state, patterns)
        )

    return {
        "sigma": mag_s,
        "tau": mag_t,
        "phi": mag_p,
        "final_state": state,
        "beta_schedule": beta_schedule,
    }
