"""CauseMe-style synthetic/semi-real benchmark generators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lag_synthetic import standardize


def _linear_fn(x):
    return x


def _squared_fn(x):
    x = np.clip(x, -10.0, 10.0)
    return x**2


@dataclass(frozen=True)
class CauseMeScenario:
    """Configuration for a CauseMe-style structural process benchmark."""

    name: str
    K: int
    T: int
    max_lag: int = 3
    density: float = 0.3
    nonlinear: bool = False

    def generate(self, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        if self.nonlinear:
            return generate_causeme_nonlinear(
                K=self.K,
                T=self.T,
                max_lag=self.max_lag,
                density=self.density,
                seed=seed,
            )
        return generate_causeme_linear(
            K=self.K,
            T=self.T,
            max_lag=self.max_lag,
            density=self.density,
            seed=seed,
        )


def default_causeme_scenarios(include_nonlinear: bool = True) -> list[CauseMeScenario]:
    """Return the default semi-real benchmark scenarios."""
    scenarios = [
        CauseMeScenario(name="CauseMe-linear-K5", K=5, T=500, density=0.3),
        CauseMeScenario(name="CauseMe-linear-K10", K=10, T=800, density=0.2),
        CauseMeScenario(name="CauseMe-linear-K20", K=20, T=1000, density=0.15),
    ]
    if include_nonlinear:
        scenarios.extend(
            [
                CauseMeScenario(
                    name="CauseMe-nonlinear-K5",
                    K=5,
                    T=500,
                    density=0.3,
                    nonlinear=True,
                ),
                CauseMeScenario(
                    name="CauseMe-nonlinear-K10",
                    K=10,
                    T=800,
                    density=0.2,
                    nonlinear=True,
                ),
            ]
        )
    return scenarios


def _links_to_ground_truth(
    links: dict[int, list[tuple[tuple[int, int], float, object]]],
    *,
    K: int,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    G_lag = np.zeros((max_lag, K, K))
    for target, link_list in links.items():
        for (source, lag), _, _ in link_list:
            if source != target and 1 <= abs(lag) <= max_lag:
                G_lag[abs(lag) - 1, target, source] = 1.0

    G_agg = (G_lag.sum(axis=0) > 0).astype(float)
    np.fill_diagonal(G_agg, 0.0)
    return G_agg, G_lag


def _fallback_var(
    *,
    K: int,
    T: int,
    max_lag: int,
    density: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Fallback generator if tigramite is unavailable."""
    from data.synthetic import generate_random_sparse_var

    X, _, G_agg = generate_random_sparse_var(K=K, T=T, density=density, seed=seed)
    G_lag = np.zeros((max_lag, K, K))
    G_lag[0] = G_agg
    np.fill_diagonal(G_lag[0], 0.0)
    np.fill_diagonal(G_agg, 0.0)
    return standardize(X), G_agg, G_lag, f"Fallback-VAR K={K} T={T}"


def generate_causeme_linear(
    K: int = 5,
    T: int = 500,
    max_lag: int = 3,
    density: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Generate a linear CauseMe-style structural process with exact lag ground truth.

    These are still synthetic benchmarks, but they are less hand-crafted than our
    local VAR generators and use Tigramite's structural process machinery when
    available.
    """
    try:
        from tigramite.toymodels import structural_causal_processes as scp

        rng = np.random.RandomState(seed)
        links: dict[int, list[tuple[tuple[int, int], float, object]]] = {}
        for target in range(K):
            links[target] = [((target, -1), 0.5, _linear_fn)]
            for source in range(K):
                if source == target or rng.random() >= density:
                    continue
                lag = rng.randint(1, max_lag + 1)
                coeff = rng.uniform(0.2, 0.5) * rng.choice([-1, 1])
                links[target].append(((source, -lag), coeff, _linear_fn))

        X, _ = scp.structural_causal_process(
            links=links,
            T=T,
            seed=seed,
            noises=None,
        )
        G_agg, G_lag = _links_to_ground_truth(links, K=K, max_lag=max_lag)
        return standardize(np.asarray(X, dtype=np.float32)), G_agg, G_lag, (
            f"CauseMe-linear K={K} T={T} density={density}"
        )
    except Exception:
        return _fallback_var(K=K, T=T, max_lag=max_lag, density=density, seed=seed)


def generate_causeme_nonlinear(
    K: int = 5,
    T: int = 500,
    max_lag: int = 3,
    density: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Generate a nonlinear CauseMe-style structural process."""
    try:
        from tigramite.toymodels import structural_causal_processes as scp

        rng = np.random.RandomState(seed)
        links: dict[int, list[tuple[tuple[int, int], float, object]]] = {}
        for target in range(K):
            links[target] = [((target, -1), 0.4, _linear_fn)]
            for source in range(K):
                if source == target or rng.random() >= density:
                    continue
                lag = rng.randint(1, max_lag + 1)
                coeff = rng.uniform(0.2, 0.4) * rng.choice([-1, 1])
                func = rng.choice([_linear_fn, _squared_fn])
                links[target].append(((source, -lag), coeff, func))

        X, _ = scp.structural_causal_process(links=links, T=T, seed=seed)
        G_agg, G_lag = _links_to_ground_truth(links, K=K, max_lag=max_lag)
        X = np.clip(np.asarray(X, dtype=np.float32), -10.0, 10.0)
        return standardize(X), G_agg, G_lag, f"CauseMe-nonlinear K={K} T={T}"
    except Exception:
        return _fallback_var(K=K, T=T, max_lag=max_lag, density=density, seed=seed)
