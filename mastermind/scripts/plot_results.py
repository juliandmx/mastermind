# mastermind/scripts/plot_results.py
import argparse, io, os
import pandas as pd
import matplotlib.pyplot as plt

HEADER = "solver,n,k,moves,time_s,secret_id"

def read_results_robust(path: str) -> pd.DataFrame:
    # --- detect encoding from BOM ---
    with open(path, "rb") as f:
        raw = f.read()
    enc = "utf-8"
    if raw.startswith(b"\xff\xfe"):
        enc = "utf-16"
    elif raw.startswith(b"\xfe\xff"):
        enc = "utf-16-be"
    elif raw.startswith(b"\xef\xbb\xbf"):
        enc = "utf-8-sig"

    text = raw.decode(enc, errors="replace")

    # --- sanitize to a clean CSV ---
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = []
    header_seen = False
    for ln in lines:
        if not ln or ln in {",", ";"}:
            continue
        if ln.startswith("[") or ln.startswith("]"):
            # drop python-list-looking chunks like: [   solver ... ]
            continue
        if ln.lower().startswith("solver"):
            if not header_seen:
                cleaned.append(HEADER)   # normalize header once
                header_seen = True
            continue
        # keep only rows with exactly 5 commas (6 fields)
        if ln.count(",") == 5:
            cleaned.append(ln)

    if not cleaned:
        raise SystemExit("No valid CSV rows found after cleaning.")

    buf = io.StringIO("\n".join(cleaned))
    df = pd.read_csv(buf)
    return df

def main():
    ap = argparse.ArgumentParser(description="Plot Mastermind benchmark results (boxplots).")
    ap.add_argument("--csv", required=True, help="Path to results.csv")
    ap.add_argument("--outdir", default="plots", help="Directory to save PNGs")
    ap.add_argument("--logtime", action="store_true", help="Use log-scale on time_s axis")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = read_results_robust(args.csv)

    # sanity check
    need = {"solver","n","k","moves","time_s","secret_id"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns after cleaning: {missing}")

    # consistent solver order
    solvers = sorted(df["solver"].unique())

    # --- Boxplot: moves ---
    fig1, ax1 = plt.subplots()
    data_moves = [df.loc[df["solver"] == s, "moves"].values for s in solvers]
    ax1.boxplot(data_moves, tick_labels=solvers, showfliers=True)
    ax1.set_title("Moves per solver")
    ax1.set_xlabel("Solver")
    ax1.set_ylabel("Moves")
    fig1.tight_layout()
    fig1.savefig(os.path.join(args.outdir, "boxplot_moves.png"), dpi=150)

    # --- Boxplot: time_s ---
    fig2, ax2 = plt.subplots()
    data_time = [df.loc[df["solver"] == s, "time_s"].values for s in solvers]
    ax2.boxplot(data_time, tick_labels=solvers, showfliers=True)
    ax2.set_title("Runtime per solver" + (" (log scale)" if args.logtime else ""))
    ax2.set_xlabel("Solver")
    ax2.set_ylabel("Time [s]")
    if args.logtime:
        ax2.set_yscale("log")
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.outdir, "boxplot_time.png"), dpi=150)

    print(f"Saved plots to: {os.path.abspath(args.outdir)}")
    print(" - boxplot_moves.png")
    print(" - boxplot_time.png")

if __name__ == "__main__":
    main()
