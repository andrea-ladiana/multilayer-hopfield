"""Tests for hopfield.dynamics — MC updates, trajectories, annealing."""

import numpy as np
import pytest

from hopfield.dynamics import (
    Couplings, Rhos, TAMState, TAMPatterns, GramInverses,
    mattis_magnetisation, rho_from_r,
    compute_gram_inverses,
    mc_step_parallel, mc_step_sequential,
    build_cued_state, build_mixture_state,
    run_trajectory, run_trajectory_annealing,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def _random_patterns(K, N, rng):
    return np.sign(rng.uniform(-1, 1, size=(K, N)))


@pytest.fixture
def small_model(rng):
    """K=3, N=64 model with pseudo-inverse couplings."""
    K, N = 3, 64
    XI = _random_patterns(K, N, rng)
    ETA = _random_patterns(K, N, rng)
    CHI = _random_patterns(K, N, rng)
    patterns = TAMPatterns(
        XI=XI, ETA=ETA, CHI=CHI,
        XI_mean=XI.copy(), ETA_mean=ETA.copy(), CHI_mean=CHI.copy(),
    )
    rhos = Rhos(sigma=0.0, tau=0.0, phi=0.0)
    couplings = Couplings(sigma_tau=1.0, sigma_phi=1.0, tau_phi=1.0)
    gi = compute_gram_inverses(XI, ETA, CHI)
    return patterns, rhos, couplings, gi


# ── Mattis magnetisation ─────────────────────────────────────────────

class TestMagnetisation:
    def test_perfect_alignment(self, rng):
        N = 128
        pattern = _random_patterns(1, N, rng)[0]
        m = mattis_magnetisation(pattern, pattern.reshape(1, -1))
        assert m.shape == (1,)
        assert np.isclose(m[0], 1.0)

    def test_anti_alignment(self, rng):
        N = 128
        pattern = _random_patterns(1, N, rng)[0]
        m = mattis_magnetisation(-pattern, pattern.reshape(1, -1))
        assert np.isclose(m[0], -1.0)

    def test_random_is_near_zero(self, rng):
        N = 4096
        pattern = _random_patterns(1, N, rng)[0]
        spins = _random_patterns(1, N, rng)[0]
        m = mattis_magnetisation(spins, pattern.reshape(1, -1))
        assert abs(m[0]) < 0.1  # ~ 1/sqrt(N) fluctuation

    def test_output_shape(self, rng):
        K, N = 5, 100
        patterns = _random_patterns(K, N, rng)
        spins = _random_patterns(1, N, rng)[0]
        m = mattis_magnetisation(spins, patterns)
        assert m.shape == (K,)


# ── rho_from_r ───────────────────────────────────────────────────────

class TestRhoFromR:
    def test_perfect_alignment(self):
        # r=1 → ρ=0 regardless of M
        assert np.isclose(rho_from_r(1.0, 100), 0.0)

    def test_positive(self):
        assert rho_from_r(0.8, 50) > 0.0

    def test_scaling_with_M(self):
        rho1 = rho_from_r(0.8, 50)
        rho2 = rho_from_r(0.8, 100)
        assert np.isclose(rho1 / rho2, 2.0, rtol=1e-10)


# ── Gram inverses ────────────────────────────────────────────────────

class TestGramInverses:
    def test_identity_for_orthogonal_patterns(self):
        # Hadamard-like orthogonal patterns
        N = 4
        P = np.array([[1, 1, 1, 1],
                       [1, -1, 1, -1],
                       [1, 1, -1, -1]], dtype=float)
        gi = compute_gram_inverses(P, P, P, gamma=0.0)
        # G = P P^T / N; for these patterns G is diagonal
        G = (P @ P.T) / N
        np.testing.assert_allclose(gi.xi @ G, np.eye(3), atol=1e-10)

    def test_regularisation(self, rng):
        K, N = 3, 64
        P = _random_patterns(K, N, rng)
        gi = compute_gram_inverses(P, P, P, gamma=1e-2)
        # G_inv @ (G + gamma I) should be identity
        G = (P @ P.T) / N
        product = gi.xi @ (G + 1e-2 * np.eye(K))
        np.testing.assert_allclose(product, np.eye(K), atol=1e-10)

    def test_shape(self, rng):
        K, N = 5, 100
        P = _random_patterns(K, N, rng)
        gi = compute_gram_inverses(P, P, P)
        assert gi.xi.shape == (K, K)
        assert gi.eta.shape == (K, K)
        assert gi.chi.shape == (K, K)


# ── State constructors ───────────────────────────────────────────────

class TestCuedState:
    def test_cued_layer_matches_pattern(self, small_model, rng):
        patterns, *_ = small_model
        state = build_cued_state(patterns, mu_cue=0, cued_layer="sigma", rng=rng)
        np.testing.assert_array_equal(state.sigma, patterns.XI[0])

    def test_other_layers_are_random(self, small_model, rng):
        patterns, *_ = small_model
        state = build_cued_state(patterns, mu_cue=0, cued_layer="sigma", rng=rng)
        # Tau and phi should not match any pattern perfectly
        m_tau = (patterns.ETA @ state.tau) / state.tau.size
        assert np.all(np.abs(m_tau) < 0.8)

    def test_noisy_cue(self, small_model, rng):
        patterns, *_ = small_model
        state = build_cued_state(
            patterns, mu_cue=0, cued_layer="sigma", noise=0.3, rng=rng,
        )
        overlap = np.dot(state.sigma, patterns.XI[0]) / state.sigma.size
        assert 0.2 < overlap < 1.0  # Not perfect, but correlated

    def test_each_layer_can_be_cued(self, small_model, rng):
        patterns, *_ = small_model
        for layer, pat_attr in [("sigma", "XI"), ("tau", "ETA"), ("phi", "CHI")]:
            state = build_cued_state(patterns, mu_cue=1, cued_layer=layer, rng=rng)
            expected = getattr(patterns, pat_attr)[1]
            actual = getattr(state, layer)
            np.testing.assert_array_equal(actual, expected)

    def test_invalid_layer_raises(self, small_model, rng):
        patterns, *_ = small_model
        with pytest.raises(ValueError, match="Unknown"):
            build_cued_state(patterns, mu_cue=0, cued_layer="foo", rng=rng)


class TestMixtureState:
    def test_binary_output(self, small_model, rng):
        patterns, *_ = small_model
        state = build_mixture_state(patterns, indices=(0, 1), rng=rng)
        for arr in [state.sigma, state.tau, state.phi]:
            assert set(np.unique(arr)).issubset({-1.0, 1.0})

    def test_symmetric_mixture_has_equal_overlaps(self, small_model, rng):
        patterns, *_ = small_model
        state = build_mixture_state(patterns, indices=(0, 1), rng=rng)
        m0 = np.dot(state.sigma, patterns.XI[0]) / state.sigma.size
        m1 = np.dot(state.sigma, patterns.XI[1]) / state.sigma.size
        # In a symmetric mixture, overlaps should be comparable
        assert abs(abs(m0) - abs(m1)) < 0.5


# ── MC steps ─────────────────────────────────────────────────────────

class TestMCSteps:
    def test_parallel_output_is_binary(self, small_model, rng):
        patterns, rhos, couplings, gi = small_model
        state0 = build_cued_state(patterns, mu_cue=0, rng=rng)
        state1 = mc_step_parallel(
            state0, patterns, rhos, couplings, beta=2.0,
            rng=rng, gram_inverses=gi,
        )
        for arr in [state1.sigma, state1.tau, state1.phi]:
            assert set(np.unique(arr)).issubset({-1.0, 1.0})

    def test_sequential_output_is_binary(self, small_model, rng):
        patterns, rhos, couplings, gi = small_model
        state0 = build_cued_state(patterns, mu_cue=0, rng=rng)
        state1 = mc_step_sequential(
            state0, patterns, rhos, couplings, beta=2.0,
            rng=rng, gram_inverses=gi,
        )
        for arr in [state1.sigma, state1.tau, state1.phi]:
            assert set(np.unique(arr)).issubset({-1.0, 1.0})

    def test_shapes_preserved(self, small_model, rng):
        patterns, rhos, couplings, gi = small_model
        state0 = build_cued_state(patterns, mu_cue=0, rng=rng)
        state1 = mc_step_parallel(
            state0, patterns, rhos, couplings, beta=2.0,
            rng=rng, gram_inverses=gi,
        )
        assert state1.sigma.shape == state0.sigma.shape
        assert state1.tau.shape == state0.tau.shape
        assert state1.phi.shape == state0.phi.shape

    def test_zero_temperature_is_deterministic(self, small_model, rng):
        """At beta=inf, tanh saturates and the uniform dither is dominated."""
        patterns, rhos, couplings, gi = small_model
        state0 = build_cued_state(patterns, mu_cue=0, rng=rng)
        state_a = mc_step_parallel(
            state0, patterns, rhos, couplings, beta=1e6,
            rng=np.random.default_rng(42), gram_inverses=gi,
        )
        state_b = mc_step_parallel(
            state0, patterns, rhos, couplings, beta=1e6,
            rng=np.random.default_rng(42), gram_inverses=gi,
        )
        np.testing.assert_array_equal(state_a.sigma, state_b.sigma)

    def test_hebbian_mode_runs(self, small_model, rng):
        """MC step without gram_inverses should still produce valid output."""
        patterns, rhos, couplings, _ = small_model
        state0 = build_cued_state(patterns, mu_cue=0, rng=rng)
        state1 = mc_step_parallel(
            state0, patterns, rhos, couplings, beta=2.0,
            rng=rng, gram_inverses=None,
        )
        for arr in [state1.sigma, state1.tau, state1.phi]:
            assert set(np.unique(arr)).issubset({-1.0, 1.0})


# ── Trajectories ─────────────────────────────────────────────────────

class TestTrajectory:
    def test_magnetisation_shapes(self, small_model, rng):
        patterns, rhos, couplings, gi = small_model
        K = patterns.XI.shape[0]
        steps = 10
        result = run_trajectory(
            patterns, rhos, couplings, beta=2.0, steps=steps,
            rng=rng, gram_inverses=gi,
        )
        assert result["sigma"].shape == (steps + 1, K)
        assert result["tau"].shape == (steps + 1, K)
        assert result["phi"].shape == (steps + 1, K)

    def test_final_state_is_binary(self, small_model, rng):
        patterns, rhos, couplings, gi = small_model
        result = run_trajectory(
            patterns, rhos, couplings, beta=2.0, steps=5,
            rng=rng, gram_inverses=gi,
        )
        state = result["final_state"]
        for arr in [state.sigma, state.tau, state.phi]:
            assert set(np.unique(arr)).issubset({-1.0, 1.0})

    def test_cued_reconstruction_high_beta(self, small_model, rng):
        """With a clean cue and high beta, the cued pattern should dominate."""
        patterns, rhos, couplings, gi = small_model
        state0 = build_cued_state(patterns, mu_cue=0, cued_layer="sigma", rng=rng)
        result = run_trajectory(
            patterns, rhos, couplings, beta=5.0, steps=30,
            init_state=state0, rng=rng, gram_inverses=gi,
        )
        final_mags = result["sigma"][-1]
        winner = int(np.argmax(final_mags))
        assert winner == 0  # Should retrieve the cued pattern

    def test_initial_magnetisation_matches_init_state(self, small_model, rng):
        patterns, rhos, couplings, gi = small_model
        state0 = build_cued_state(patterns, mu_cue=1, cued_layer="sigma", rng=rng)
        result = run_trajectory(
            patterns, rhos, couplings, beta=2.0, steps=5,
            init_state=state0, rng=rng, gram_inverses=gi,
        )
        m0_expected = mattis_magnetisation(state0.sigma, patterns.XI)
        np.testing.assert_allclose(result["sigma"][0], m0_expected)


class TestAnnealingTrajectory:
    def test_output_shapes(self, small_model, rng):
        patterns, rhos, couplings, gi = small_model
        K = patterns.XI.shape[0]
        schedule = np.linspace(0.5, 5.0, 20)
        result = run_trajectory_annealing(
            patterns, rhos, couplings, schedule,
            rng=rng, gram_inverses=gi,
        )
        assert result["sigma"].shape == (21, K)
        np.testing.assert_array_equal(result["beta_schedule"], schedule)

    def test_sequential_flag(self, small_model, rng):
        patterns, rhos, couplings, gi = small_model
        schedule = np.linspace(0.5, 5.0, 10)
        result = run_trajectory_annealing(
            patterns, rhos, couplings, schedule,
            rng=rng, gram_inverses=gi, sequential=True,
        )
        state = result["final_state"]
        for arr in [state.sigma, state.tau, state.phi]:
            assert set(np.unique(arr)).issubset({-1.0, 1.0})
