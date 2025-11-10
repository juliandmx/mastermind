import numpy as np
from .base import AbstractSolver


class Frequency(AbstractSolver):
    """
    Erzeugt einen Tipp, indem für jede Position die häufigste Farbe in
    den verbleibenden Kandidaten gewählt wird.

    Beispiel:
        Kandidaten (n=4, k=6)
        0 1 2 3
        1 1 2 0
        1 3 2 0
    -> Für Position 0 ist '1' am häufigsten, für Pos 1 '1',
      für Pos 2 '2', für Pos 3 '0' → Tipp = [1,1,2,0]
    """

    def next_guess(self, history, candidates: np.ndarray) -> np.ndarray:
        n, k = self.n, self.k

        # Wenn nur eine Möglichkeit existiert -> direkt zurückgeben
        if len(candidates) == 1:
            return candidates[0]
        if not history:
            return [0,0,1,2]

        # Hier speichern wir den konstruierten Tipp (eine Farbe pro Position)
        guess = np.empty(n, dtype=np.int16)

        # --- Schritt 1: pro Position die häufigste Farbe bestimmen ---
        guess = np.empty(n, dtype=np.int16)
        for i in range(n):
            counts = np.bincount(candidates[:, i], minlength=k)
            guess[i] = np.argmax(counts)

        # --- Schritt 2: prüfen, ob dieser Tipp gültig ist ---
        is_valid = np.any(np.all(candidates == guess, axis=1))
        if is_valid:
            return guess

        # --- Schritt 3: ähnlichsten gültigen Kandidaten wählen ---
        # Hamming-Distanz = Anzahl unterschiedlicher Pins
        distances = np.count_nonzero(candidates != guess, axis=1)
        best_idx = np.argmin(distances)

        # Der ähnlichste Kandidat ist der neue Tipp
        return candidates[best_idx]

        print(guess)
        return guess