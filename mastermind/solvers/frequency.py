# mastermind/solvers/frequency.py
import numpy as np
from .base import AbstractSolver

class Frequency(AbstractSolver):
    """
    Per-position frequency with simple anti-repeat:
    - build argmax-by-position guess
    - if repeated / already tried, flip one position to 2nd-best color
      (choose the position where top1 and top2 are closest)
    - optional snap to nearest valid candidate by Hamming distance
    """
    def __init__(self, n: int, k: int, snap_to_candidate: bool = True, rng_seed: int | None = None):
        super().__init__(n, k)
        self.snap_to_candidate = snap_to_candidate
        self._tried = set()  # set of tuples
        self.rng = np.random.default_rng(rng_seed)

    def next_guess(self, history, candidates: np.ndarray) -> np.ndarray:
        n, k = self.n, self.k
        if len(candidates) == 0:
            raise ValueError("Keine Kandidaten vorhanden.")
        if len(candidates) == 1:
            g = candidates[0]
            self._tried.add(tuple(int(x) for x in g))
            return g

        # Informative first guess
        if not history:
            g0 = (np.arange(n, dtype=np.int16) % k)
            self._tried.add(tuple(int(x) for x in g0))
            return g0

        # --- build per-position argmax + keep top2 for tie-breaking ---
        counts = [np.bincount(candidates[:, i], minlength=k) for i in range(n)]
        top1 = np.empty(n, dtype=np.int16)
        top2 = np.empty(n, dtype=np.int16)
        gap  = np.empty(n, dtype=np.int32)
        for i in range(n):
            # indices of colors sorted by count desc
            order = np.argsort(counts[i])[::-1]
            top1[i] = int(order[0])
            top2[i] = int(order[1]) if k > 1 else int(order[0])
            gap[i]  = int(counts[i][top1[i]] - counts[i][top2[i]])

        guess = top1.copy()

        # optional: snap to a valid candidate if constructed not in candidates
        if self.snap_to_candidate and not np.any(np.all(candidates == guess, axis=1)):
            # nearest by Hamming distance
            d = np.count_nonzero(candidates != guess, axis=1)
            guess = candidates[int(np.argmin(d))].astype(np.int16, copy=False)

        # --- anti-repeat: avoid proposing the same guess again ---
        last_guess = history[-1][0]
        tried_tuple = tuple(int(x) for x in guess)
        last_tuple  = tuple(int(x) for x in last_guess)

        if tried_tuple == last_tuple or tried_tuple in self._tried:
            # flip one position to 2nd-best where it hurts least (smallest gap)
            order_pos = np.argsort(gap)  # ascending -> smallest gap first
            changed = False
            for i in order_pos:
                alt = guess.copy()
                alt[i] = top2[i]
                t = tuple(int(x) for x in alt)
                if t != last_tuple and t not in self._tried:
                    guess = alt
                    tried_tuple = t
                    changed = True
                    break
            if not changed:
                # fallback: pick a random untried candidate
                # (keeps us moving even in degenerate frequency ties)
                idx = self.rng.permutation(len(candidates))
                for j in idx:
                    t = tuple(int(x) for x in candidates[j])
                    if t not in self._tried:
                        guess = candidates[j]
                        tried_tuple = t
                        break  # if all tried, we’ll reuse (rare)

        self._tried.add(tried_tuple)
        return guess
