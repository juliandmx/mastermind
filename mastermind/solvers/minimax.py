from collections import defaultdict
import numpy as np
from .base import AbstractSolver
from ..core.feedback import feedback

class Minimax(AbstractSolver):
    """Wähle den Tipp, der die maximale Partitionsgröße (Worst-Case) minimiert.
    Aus Einfachheitsgründen wählen wir nur aus 'candidates' (nicht aus dem gesamten Raum).
    """
    def next_guess(self, history, candidates: np.ndarray) -> np.ndarray:
        if len(candidates) <= 2:
            return candidates[0]

        best_idx = 0
        best_wc = 10**12
        sample_cap = 2000
        if len(candidates) > sample_cap:
            sample_idx = np.random.choice(len(candidates), size=sample_cap, replace=False)
        else:
            sample_idx = np.arange(len(candidates))

        for gi in sample_idx:
            g = candidates[gi]
            parts = defaultdict(int)
            for ci in range(len(candidates)):
                b,w = feedback(g, candidates[ci])
                parts[(b,w)] += 1
            wc = max(parts.values())
            if wc < best_wc:
                best_wc = wc
                best_idx = gi

        return candidates[best_idx]
