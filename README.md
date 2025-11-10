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
### Manuelles Spiel
```bash
# Eine Runde manuell spielen, um korrekte Funktionsweise zu testen
python -m mastermind.scripts.run_bench --n 4 --k 6 --repeats 1 --solvers manual
```

### Alle Algorithmen testen & Plots erstellen
```bash
# Beispiel: klassisches Mastermind (n=4, k=6) mit 10 zufälligen Geheimcodes,
# wobei die Ergebnisse in der csv-Datei gespeichert werden.
python -m mastermind.scripts.run_bench --n 4 --k 6 --repeats 10 --solvers random,minimax,frequency > results.csv 

# Plots erstellen
python -m mastermind.scripts.plot_results
```