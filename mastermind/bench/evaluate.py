from __future__ import annotations
import time
import numpy as np
from ..core.feedback import feedback
from ..core.space import all_codes, is_consistent

def play_once(solver, secret: np.ndarray, n: int, k: int) -> tuple[int, float]:
    """Spiele bis gelöst. Rückgabe: (Anzahl Züge, Spielzeit in Sekunden)."""
    candidates = all_codes(n, k)
    solver.reset() if hasattr(solver, "reset") else None
    history = []

    t0 = time.perf_counter()
    while True:
        guess = solver.next_guess(history, candidates)
        b, w = feedback(guess, secret)
        history.append((guess, (b,w)))
        if b == n:
            return len(history), (time.perf_counter() - t0)
        mask = is_consistent(guess, (b,w), candidates, feedback)
        candidates = candidates[mask]

def benchmark(solver_cls, n: int, k: int, secrets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    moves, times = [], []
    for s in secrets:
        solver = solver_cls(n, k)
        m, t = play_once(solver, s, n, k)
        moves.append(m)
        times.append(t)
    return np.array(moves), np.array(times)
