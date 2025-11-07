import numpy as np
from .base import AbstractSolver

class RandomValid(AbstractSolver):
    """Baseline: wähle zufällig aus den noch gültigen Kandidaten."""
    def next_guess(self, history, candidates: np.ndarray) -> np.ndarray:
        idx = np.random.randint(len(candidates))
        return candidates[idx]
