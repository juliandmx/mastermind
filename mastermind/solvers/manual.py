import numpy as np
from .base import AbstractSolver

def _parse_guess(text: str, n: int, k: int) -> np.ndarray:
    text = text.strip()
    # Eingabeformate erlauben: "0 1 2 3", "0123" oder "0,1,2,3"
    if " " in text or "," in text:
        parts = [p for p in text.replace(",", " ").split() if p]
        if len(parts) != n:
            raise ValueError(f"Bitte genau {n} Zahlen angeben.")
        vals = [int(p) for p in parts]
    else:
        if len(text) != n:
            raise ValueError(f"Bitte genau {n} Ziffern eingeben.")
        vals = [int(ch) for ch in text]
    if any(v < 0 or v >= k for v in vals):
        raise ValueError(f"Zahlen müssen in 0..{k-1} liegen.")
    return np.asarray(vals, dtype=np.int16)

class Manual(AbstractSolver):
    """Lässt dich selbst (per Tastatur) Mastermind spielen.
    Nicht für Benchmarks gedacht – nur für interaktives Spielen.
    """
    def next_guess(self, history, candidates: np.ndarray) -> np.ndarray:
        if history:
            last_guess, (b, w) = history[-1]
            print(f"Letztes Feedback: {last_guess.tolist()} -> schwarz={b}, weiß={w}")
        while True:
            try:
                raw = input(f"Gib deinen Tipp (n={self.n}, Farben 0..{self.k-1}): ")
                return _parse_guess(raw, self.n, self.k)
            except Exception as e:
                print(f"Fehler: {e}")
