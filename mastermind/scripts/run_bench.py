import argparse, sys
import numpy as np
import pandas as pd

from mastermind.bench.sampler import secrets_uniform
from mastermind.bench.evaluate import benchmark
from mastermind.solvers.random_valid import RandomValid
from mastermind.solvers.minimax import Minimax

SOLVER_REGISTRY = {
    "random": RandomValid,
    "minimax": Minimax,
}

def parse_args():
    ap = argparse.ArgumentParser(description="Mastermind Benchmark Runner")
    ap.add_argument("--n", type=int, default=4, help="Code-Länge")
    ap.add_argument("--k", type=int, default=6, help="Anzahl Farben (0..k-1)")
    ap.add_argument("--repeats", type=int, default=100, help="Anzahl Geheimcodes")
    ap.add_argument("--seed", type=int, default=42, help="Seed für Sampler")
    ap.add_argument("--solvers", type=str, default="random",
                    help="Kommagetrennte Liste: random,minimax")
    return ap.parse_args()

def main():
    args = parse_args()
    secrets = secrets_uniform(args.n, args.k, args.repeats, args.seed)

    rows = []
    for name in [s.strip() for s in args.solvers.split(",") if s.strip()]:
        if name not in SOLVER_REGISTRY:
            print(f"# Unbekannter Solver: {name}", file=sys.stderr)
            continue
        solver_cls = SOLVER_REGISTRY[name]
        moves, times = benchmark(solver_cls, args.n, args.k, secrets)
        df = pd.DataFrame({
            "solver": name,
            "n": args.n,
            "k": args.k,
            "moves": moves,
            "time_s": times,
            "secret_id": np.arange(len(moves)),
        })
        rows.append(df)

    if not rows:
        print("# Keine gültigen Solver angegeben.", file=sys.stderr)
        sys.exit(1)

    print(rows)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(sys.stdout, index=False)

if __name__ == "__main__":
    main()
