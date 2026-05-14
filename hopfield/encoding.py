"""
Ising spin encoders for continuous multivariate data.

All encoders follow a fit / transform API analogous to scikit-learn.

Encoding pipeline
-----------------
1. Centre and reduce dimensionality (PCA, LDA, or STFT pre-processing).
2. Whiten the projected features.
3. Apply SimHash: random hyperplane projection followed by sign binarisation.

The SimHash step maps each sample to a binary vector in {−1, +1}^N whose
Hamming distances approximate angular distances in the whitened feature space
(locality-sensitive hashing guarantee).
"""

from __future__ import annotations

import numpy as np
import scipy.signal
from dataclasses import dataclass, field
from typing import Optional

_Array = np.ndarray


@dataclass
class IsingEncoder:
    """
    PCA + whitening + SimHash encoder.

    Parameters
    ----------
    n_components : number of PCA components to retain (None = full rank).
    n_spins : dimensionality of the output Ising vector.
    random_state : seed for reproducible hyperplane generation.
    """
    n_components: int | None = None
    n_spins: int = 512
    random_state: int | None = None

    # Fitted state (private)
    _mean: _Array | None = field(default=None, repr=False)
    _components: _Array | None = field(default=None, repr=False)
    _eigvals: _Array | None = field(default=None, repr=False)
    _hyperplanes: _Array | None = field(default=None, repr=False)
    _rng: np.random.Generator | None = field(default=None, repr=False)

    def _check_fitted(self) -> None:
        if self._mean is None:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

    def fit(self, X: _Array) -> IsingEncoder:
        """Fit PCA + whitening + random hyperplanes on training data X."""
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        self._rng = np.random.default_rng(self.random_state)

        # Centre
        self._mean = X.mean(axis=0)
        Xc = X - self._mean

        # Compact SVD
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        eigvals = S ** 2 / max(1, n_samples - 1)

        d = Vt.shape[0]
        if self.n_components is not None:
            d = min(self.n_components, d)

        components = Vt[:d]
        eigvals = eigvals[:d]

        # Drop near-zero eigenvalues
        keep = eigvals > 1e-12
        self._components = components[keep]
        self._eigvals = eigvals[keep]

        # Random hyperplanes in whitened space
        d_eff = self._components.shape[0]
        self._hyperplanes = self._rng.standard_normal((self.n_spins, d_eff))
        return self

    def _whiten(self, X: _Array) -> _Array:
        self._check_fitted()
        Xc = np.asarray(X, dtype=float) - self._mean
        Z = Xc @ self._components.T
        return Z / np.sqrt(self._eigvals)

    def transform(self, X: _Array) -> _Array:
        """Encode X into {−1, +1}^n_spins via SimHash."""
        Z = self._whiten(X)
        H = Z @ self._hyperplanes.T
        S = np.sign(H)
        zeros = S == 0.0
        if np.any(zeros):
            S[zeros] = self._rng.choice([-1.0, 1.0], size=int(zeros.sum()))
        return S

    def fit_transform(self, X: _Array) -> _Array:
        return self.fit(X).transform(X)


@dataclass
class STFTIsingEncoder(IsingEncoder):
    """
    STFT magnitude + SimHash encoder for multivariate time series.

    The short-time Fourier transform extracts frequency content while
    abstracting away exact local phase, producing translation-tolerant
    spin representations.

    Parameters
    ----------
    nperseg : STFT window length.
    noverlap : STFT overlap.
    time_steps : number of time steps per sample.
    channels : number of spatial channels per sample.
    """
    nperseg: int = 20
    noverlap: int = 15
    time_steps: int = 100
    channels: int = 3

    def _stft_features(self, X: _Array) -> _Array:
        n = X.shape[0]
        # Input is (n, time_steps * channels); reshape to (n, channels, time_steps)
        X3 = X.reshape(n, self.time_steps, self.channels).transpose(0, 2, 1)
        _, _, Zxx = scipy.signal.stft(
            X3, axis=-1, nperseg=self.nperseg, noverlap=self.noverlap,
        )
        return np.abs(Zxx).reshape(n, -1)

    def fit(self, X: _Array) -> STFTIsingEncoder:
        super().fit(self._stft_features(X))
        return self

    def transform(self, X: _Array) -> _Array:
        return super().transform(self._stft_features(X))


@dataclass
class LDAIsingEncoder:
    """
    LDA + PCA residual + SimHash supervised encoder.

    Uses Fisher linear discriminant analysis for the primary projection,
    augmented with additional PCA directions from the within-class residual
    to provide geometric room for the archetypes.

    With K classes, LDA yields K−1 discriminant directions.  If
    *n_pca_extra* > 0, additional top PCA eigenvectors (orthogonal to the
    LDA subspace) are appended, giving SimHash more angular diversity and
    reducing archetype cross-correlations (BCI).

    Parameters
    ----------
    n_spins : output dimensionality.
    n_pca_extra : extra PCA directions beyond LDA.
    random_state : for reproducibility.
    """
    n_spins: int = 512
    n_pca_extra: int = 8
    random_state: int | None = None

    # Fitted state
    _mean: _Array | None = field(default=None, repr=False)
    _projection: _Array | None = field(default=None, repr=False)
    _scale: _Array | None = field(default=None, repr=False)
    _hyperplanes: _Array | None = field(default=None, repr=False)
    _rng: np.random.Generator | None = field(default=None, repr=False)

    def _check_fitted(self) -> None:
        if self._mean is None:
            raise RuntimeError("Encoder not fitted. Call fit(X, y) first.")

    def fit(self, X: _Array, y: _Array) -> LDAIsingEncoder:
        """Fit LDA + PCA residual + SimHash on labelled data."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_samples, n_features = X.shape
        self._rng = np.random.default_rng(self.random_state)

        classes = np.unique(y)
        K = len(classes)

        # Global mean
        self._mean = X.mean(axis=0)
        Xc = X - self._mean

        # Scatter matrices
        S_w = np.zeros((n_features, n_features))
        S_b = np.zeros((n_features, n_features))
        for c in classes:
            Xc_c = X[y == c]
            mu_c = Xc_c.mean(axis=0)
            diff = Xc_c - mu_c
            S_w += diff.T @ diff
            d = (mu_c - self._mean).reshape(-1, 1)
            S_b += Xc_c.shape[0] * (d @ d.T)

        S_w += 1e-6 * np.eye(n_features)

        # Generalised eigenvalue problem S_w⁻¹ S_b
        eigvals, eigvecs = np.linalg.eigh(np.linalg.solve(S_w, S_b))
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        n_lda = min(K - 1, n_features)
        keep = eigvals[:n_lda] > 1e-10
        lda_vecs = eigvecs[:, :n_lda][:, keep]
        lda_vals = eigvals[:n_lda][keep]

        # PCA on within-class residual orthogonal to LDA
        directions = [lda_vecs]
        scales = [np.sqrt(lda_vals)]

        if self.n_pca_extra > 0:
            residual = Xc - (Xc @ lda_vecs) @ lda_vecs.T
            _, S_res, Vt_res = np.linalg.svd(residual, full_matrices=False)
            pca_vals = S_res ** 2 / max(1, n_samples - 1)
            n_extra = min(self.n_pca_extra, Vt_res.shape[0])
            pca_vecs = Vt_res[:n_extra].T
            pca_vals = pca_vals[:n_extra]
            keep_pca = pca_vals > 1e-12
            directions.append(pca_vecs[:, keep_pca])
            scales.append(np.sqrt(pca_vals[keep_pca]))

        all_dirs = np.concatenate(directions, axis=1)
        all_scales = np.concatenate(scales)

        self._projection = all_dirs.T       # (d_total, n_features)
        self._scale = all_scales            # (d_total,)

        d_total = all_dirs.shape[1]
        self._hyperplanes = self._rng.standard_normal((self.n_spins, d_total))
        return self

    def transform(self, X: _Array) -> _Array:
        """Encode X into {−1, +1}^n_spins."""
        self._check_fitted()
        Xc = np.asarray(X, dtype=float) - self._mean
        Z = Xc @ self._projection.T
        Z_white = Z / (self._scale + 1e-20)
        H = Z_white @ self._hyperplanes.T
        S = np.sign(H)
        zeros = S == 0.0
        if np.any(zeros):
            S[zeros] = self._rng.choice([-1.0, 1.0], size=int(zeros.sum()))
        return S

    def fit_transform(self, X: _Array, y: _Array) -> _Array:
        return self.fit(X, y).transform(X)
