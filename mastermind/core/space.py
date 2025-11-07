from typing import Tuple
import numpy as np

def all_codes(n: int, k: int, dtype="int8") -> np.ndarray:
    """Erzeuge den vollständigen Kandidatenraum (k^n x n)."""
    grid = np.indices((k,)*n).reshape(n, -1).T.astype(dtype)
    return grid

def is_consistent(guess: np.ndarray, fb: Tuple[int,int], candidates: np.ndarray, feedback_fn) -> np.ndarray:
    """Maske der Kandidaten, die zum Feedback (black, white) passen."""
    b,w = fb
    mask = np.empty(len(candidates), dtype=bool)
    for i in range(len(candidates)):
        bb, ww = feedback_fn(guess, candidates[i])
        mask[i] = (bb == b) and (ww == w)
    return mask
