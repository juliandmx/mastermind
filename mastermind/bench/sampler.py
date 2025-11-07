import numpy as np

def secrets_uniform(n: int, k: int, repeats: int, seed: int | None = 42) -> np.ndarray:
    """Ziehe 'repeats' Geheimcodes gleichverteilt."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, k, size=(repeats, n), dtype=np.int16)
