import argparse, sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from mastermind.bench.sampler import secrets_uniform
from mastermind.bench.evaluate import benchmark
from mastermind.solvers.random_valid import RandomValid
from mastermind.solvers.minimax import Minimax
from mastermind.solvers.manual import Manual
from mastermind.solvers.frequency import Frequency
from mastermind.solvers.entropy import Entropy

SOLVER_REGISTRY = {
    "random": RandomValid,
    "minimax": Minimax,
    "manual": Manual,
    "frequency": Frequency,
    "entropy": Entropy,
}

def parse_args():
    ap = argparse.ArgumentParser(description="Mastermind Benchmark Runner")
    ap.add_argument("--n", type=int, default=4, help="Code-Länge")
    ap.add_argument("--k", type=int, default=6, help="Anzahl Farben (0..k-1)")
    ap.add_argument("--repeats", type=int, default=10, help="Anzahl Geheimcodes")
    ap.add_argument("--seed", type=int, default=42, help="Seed für Sampler")
    ap.add_argument("--solvers", type=str, default="random,minimax,frequency",
                    help="Kommagetrennte Liste: random,minimax,manual,frequency")
    return ap.parse_args()

def main():
    args = parse_args()
    solver_names = [s.strip() for s in args.solvers.split(",") if s.strip()]
    secrets_all = secrets_uniform(args.n, args.k, args.repeats+1, args.seed)

    rows = []

    # total runs counted in the bar: (repeats - 1) per solver (since we skip warm-up)
    measured_runs_per_solver = max(0, len(secrets_all) - 1)
    total_runs = measured_runs_per_solver * len(solver_names)

    with tqdm(total=total_runs, desc="Runs", ncols=80) as pbar:
        for name in solver_names:
            if name not in SOLVER_REGISTRY:
                print(f"# Unbekannter Solver: {name}", file=sys.stderr)
                continue
            solver_cls = SOLVER_REGISTRY[name]

            # 1) Warm-up (not counted)
            if len(secrets_all) >= 1:
                _ = benchmark(solver_cls, args.n, args.k, secrets_all[:1], on_progress=None)

            # 2) Real runs with a single global progress bar
            run_counter = 0
            def on_progress():
                nonlocal run_counter
                run_counter += 1
                pbar.set_postfix(solver=name, run=f"{run_counter}/{measured_runs_per_solver}")
                pbar.update(1)

            moves, times = benchmark(solver_cls, args.n, args.k, secrets_all[1:], on_progress=on_progress)

            df = pd.DataFrame({
                "solver": name,
                "n": args.n,
                "k": args.k,
                "moves": moves,
                "time_s": times,
                "secret_id": np.arange(1, len(moves) + 1),  # shifted because of warm-up
            })
            rows.append(df)

    if not rows:
        print("# Keine gültigen Solver angegeben.", file=sys.stderr)
        sys.exit(1)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(sys.stdout, index=False)

if __name__ == "__main__":
    main()
