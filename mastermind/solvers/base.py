from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class AbstractSolver(ABC):
    """Interface für Solver: stateless pro Zug, stateful über history."""
    def __init__(self, n: int, k: int):
        self.n, self.k = n, k

    @abstractmethod
    def next_guess(self, history, candidates: np.ndarray) -> np.ndarray:
        """Gib den nächsten Tipp als 1D-Array der Länge n zurück."""
        ...

    def reset(self):
        pass
