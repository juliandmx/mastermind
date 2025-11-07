from typing import Tuple
import numpy as np

try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def deco(fn): return fn
        return deco

@njit
def feedback(guess: np.ndarray, secret: np.ndarray) -> Tuple[int, int]:
    """
    Berechnet (black, white) für einen Tipp und den Geheimcode.
    Annahmen:
      - guess und secret sind 1D int-Arrays gleicher Länge
      - Farben sind 0..(k-1)
    """
    n = len(guess)
    black = 0

    MAX_COLORS = 64
    counts_g = np.zeros(MAX_COLORS, dtype=np.int16)
    counts_s = np.zeros(MAX_COLORS, dtype=np.int16)

    for i in range(n):
        if guess[i] == secret[i]:
            black += 1
        else:
            counts_g[guess[i]] += 1
            counts_s[secret[i]] += 1

    white = 0
    for c in range(MAX_COLORS):
        if counts_g[c] and counts_s[c]:
            white += counts_g[c] if counts_g[c] < counts_s[c] else counts_s[c]

    return black, white
