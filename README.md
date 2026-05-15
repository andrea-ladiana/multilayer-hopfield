# multilayer-hopfield

Simulation core for **multilayer hetero-associative memories** with
Kanter–Sompolinsky pseudo-inverse couplings.

The model stores *K* multivariate patterns as triplets (ξ, η, χ) across three
binary visible layers σ, τ, φ ∈ {−1, +1}^N, and retrieves them via stochastic
Monte Carlo dynamics at finite inverse temperature β.

## Features

| Module | Description |
|--------|-------------|
| `hopfield.dynamics` | Parallel and sequential MC updates, constant-β trajectories, simulated annealing |
| `hopfield.patterns` | Archetype extraction, Gram diagnostics, spectral sharpening (Kanter–Sompolinsky) |
| `hopfield.encoding` | Continuous → Ising encoders: PCA + SimHash, STFT + SimHash, LDA + SimHash |
| `hopfield.metrics` | Mattis overlaps, winner margins, cross-layer consistency, reconstruction success |
| `hopfield.datasets` | Synthetic 3D-curve datasets and parametric curve families for capacity studies |

## Installation

```bash
pip install -e .
```

For development (includes test dependencies):

```bash
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from hopfield.datasets import simulate_3d_curve_dataset
from hopfield.encoding import IsingEncoder
from hopfield.patterns import compute_empirical_archetypes, apply_spectral_sharpening
from hopfield.dynamics import (
    TAMPatterns, Couplings, Rhos,
    compute_gram_inverses, build_cued_state, run_mc_trajectory,
)

# Generate data
X_train, y_train, *_ = simulate_3d_curve_dataset(K=3, n_train=200)

# Encode each layer independently
enc = IsingEncoder(n_components=32, n_spins=512, random_state=0)
S = enc.fit_transform(X_train)

# Extract archetypes
groups = np.stack([S[y_train == k] for k in range(3)])
xi = compute_empirical_archetypes(groups)

# Build TAM patterns (all layers identical for this demo)
patterns = TAMPatterns(
    XI=xi, ETA=xi, CHI=xi,
    XI_mean=xi, ETA_mean=xi, CHI_mean=xi,
)

# Run retrieval
rhos = Rhos(sigma=0.0, tau=0.0, phi=0.0)
couplings = Couplings(sigma_tau=1.0, sigma_phi=1.0, tau_phi=1.0)
gi = compute_gram_inverses(xi, xi, xi)
state0 = build_cued_state(patterns, mu_cue=0, cued_layer="sigma")
result = run_trajectory(
    patterns, rhos, couplings, beta=3.0, steps=50,
    init_state=state0, gram_inverses=gi,
)
print("Final sigma-magnetisations:", result["sigma"][-1])
```

## Running tests

```bash
pytest
```

## Project layout

```
multilayer-hopfield/
├── hopfield/
│   ├── __init__.py
│   ├── dynamics.py      # Monte Carlo update rules and trajectories
│   ├── patterns.py      # Archetype extraction and spectral sharpening
│   ├── encoding.py      # Ising spin encoders
│   ├── metrics.py       # Retrieval quality diagnostics
│   └── datasets.py      # Synthetic dataset generators
├── tests/
│   ├── test_dynamics.py
│   ├── test_patterns.py
│   ├── test_encoding.py
│   ├── test_metrics.py
│   └── test_datasets.py
├── pyproject.toml
├── LICENSE
└── README.md
```

## Citation

If you use this code, please cite:
```bibtex
@misc{ladiana2026finite,
    title = {Finite-size scaling of hetero-associative retrieval in continuous-signal-driven Ising spin systems},
    author = {Andrea Ladiana},
    howpublished = {arXiv:2605.14059},
    year = {2026},
    url = {https://arxiv.org/abs/2605.14059},
    doi = {10.48550/arXiv.2605.14059}
}
```

## License

MIT — see [LICENSE](LICENSE).
