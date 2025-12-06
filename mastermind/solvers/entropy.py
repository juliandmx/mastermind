from collections import defaultdict
import math
import numpy as np

from .base import AbstractSolver
from ..core.feedback import feedback


class Entropy(AbstractSolver):
    """
    Entropie-basierter Solver nach Bestavros & Belal (1986).

    Strategie (MaxEnt):
      - Für jeden möglichen Tipp G aus den noch gültigen Kandidaten:
        * Partitioniere den Kandidatenpool P nach dem Feedback (black, white).
        * Sei p_j = |C_j| / |P| die Wahrscheinlichkeit der Antwort j.
        * Shannon-Entropie H(G) = - Sum_j p_j * log2(p_j).
      - Wähle den Tipp mit maximaler Entropie H(G).
      - Tie-Breaker: bei gleicher Entropie bevorzuge den Tipp mit kleinerem
        Worst-Case (maximale Klassengröße).
    """

    def next_guess(self, history, candidates: np.ndarray) -> np.ndarray:
        # Wenn nur noch sehr wenige Kandidaten übrig sind, rate direkt
        if len(candidates) <= 2:
            return candidates[0]

        n_candidates = len(candidates)

        best_idx = 0
        best_entropy = -1.0
        best_wc = 10**12
        eps = 1e-12

        # Wie im Paper: Guesses nur aus dem aktuellen Pool P_{i-1}
        for gi in range(n_candidates):
            g = candidates[gi]

            # Zähle, wie viele Kandidaten zu welchem Feedback gehören
            parts = defaultdict(int)
            for ci in range(n_candidates):
                b, w = feedback(g, candidates[ci])
                parts[(b, w)] += 1

            # Shannon-Entropie + Worst Case aus diesen Klassen berechnen
            H = 0.0
            wc = 0
            for cnt in parts.values():
                if cnt > wc:
                    wc = cnt
                p = cnt / n_candidates
                # p > 0, da nur nichtleere Klassen im dict
                H -= p * math.log2(p)

            # Besser, wenn:
            #   - höhere Entropie, oder
            #   - gleiche Entropie, aber kleinerer Worst Case
            if (H > best_entropy + eps) or (abs(H - best_entropy) <= eps and wc < best_wc):
                best_entropy = H
                best_wc = wc
                best_idx = gi

        return candidates[best_idx]
