from collections import defaultdict
import numpy as np
from .base import AbstractSolver
from ..core.feedback import feedback
from ..core.space import all_codes


class Minimax(AbstractSolver):
    """
    Knuth-ähnlicher Minimax-Solver:
    - Kandidaten = alle noch möglichen Geheimcodes
    - Mögliche GUESSES = gesamter Raum all_codes(n, k)
    - Auswahl: Minimiere die maximale Rest-Kandidatanzahl (Worst-Case),
      Tie-Breaker: bevorzuge gültige Kandidaten.
    """

    def next_guess(self, history, candidates: np.ndarray) -> np.ndarray:
        # Triviale Fälle: wenn nur noch sehr wenige Kandidaten übrig sind,
        # rate einfach direkt einen davon.
        if len(candidates) <= 2:
            return candidates[0]

        # Gesamter Raum als mögliche GUESSES
        full_space = all_codes(self.n, self.k, dtype=candidates.dtype)

        # Optional: fester Startzug wie bei Knuth (z.B. 0011 bei Farben 0..5)
        if not history:
            # Prüfen, ob dieser Startzug überhaupt im Raum liegt
            start_guess = np.array([0, 0, 1, 1], dtype=candidates.dtype)
            # Sicherheitshalber: wenn n != 4, fallback auf generischen Minimax
            if len(start_guess) == candidates.shape[1]:
                return start_guess

        best_guess = None
        best_wc = 10**12
        best_is_candidate = False  # fürs Tie-Breaking

        # Für JEDEN möglichen Tipp aus dem vollen Raum:
        for g in full_space:
            parts = defaultdict(int)

            # Partitioniere die aktuellen Kandidaten nach Feedback zu g
            for c in candidates:
                b, w = feedback(g, c)
                parts[(b, w)] += 1

            wc = max(parts.values())  # Worst-Case-Größe

            # Ist g selbst noch ein gültiger Kandidat?
            # (das ist der Knuth-Tiebreaker)
            is_candidate = np.any(np.all(candidates == g, axis=1))

            # Besser, wenn:
            # - strenger kleineres worst-case, oder
            # - gleiches worst-case, aber g ist Kandidat und der bisherige nicht
            if (wc < best_wc) or (wc == best_wc and is_candidate and not best_is_candidate):
                best_wc = wc
                best_guess = g
                best_is_candidate = is_candidate

        # Fallback (sollte eigentlich nie nötig sein)
        if best_guess is None:
            return candidates[0]

        return best_guess
