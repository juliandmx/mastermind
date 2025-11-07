# Mastermind Bench (Mini-Repo)

Ein kleines, reproduzierbares Gerüst zum wissenschaftlichen Vergleich verschiedener Mastermind-Solver.
Fokus: sauberes Benchmarking, einfache Erweiterbarkeit, optional beschleunigbare Hotspots.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> `numba` ist optional. Ohne Numba läuft alles in reinem Python/NumPy, nur langsamer.

## Schnellstart
```bash
# Beispiel: klassisches Mastermind (n=4, k=6) mit 200 zufälligen Geheimcodes
python scripts/run_bench.py --n 4 --k 6 --repeats 200 --solvers random,minimax
```

Die Ergebnisse (Moves/Runtime) erscheinen als CSV auf STDOUT. Um sie zu speichern:
```bash
python scripts/run_bench.py --n 4 --k 6 --repeats 200 > results.csv
```
