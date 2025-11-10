import argparse, io, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HEADER = "solver,n,k,moves,time_s,secret_id"

def read_results_robust(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        raw = f.read()
    enc = "utf-8"
    if raw.startswith(b"\xff\xfe"): enc = "utf-16"
    elif raw.startswith(b"\xfe\xff"): enc = "utf-16-be"
    elif raw.startswith(b"\xef\xbb\xbf"): enc = "utf-8-sig"
    text = raw.decode(enc, errors="replace")

    lines = [ln.strip() for ln in text.splitlines()]
    cleaned, header_seen = [], False
    for ln in lines:
        if not ln or ln in {",", ";"}: continue
        if ln.startswith("[") or ln.startswith("]"): continue
        if ln.lower().startswith("solver"):
            if not header_seen:
                cleaned.append(HEADER); header_seen = True
            continue
        if ln.count(",") == 5:
            cleaned.append(ln)
    if not cleaned:
        raise SystemExit("No valid CSV rows found after cleaning.")
    return pd.read_csv(io.StringIO("\n".join(cleaned)))

def _title_with_params(base: str, df: pd.DataFrame) -> str:
    n = int(df["n"].iloc[0])
    k = int(df["k"].iloc[0])
    repeats = df["secret_id"].nunique()
    return f"{base} (n={n}, k={k}, repeats={repeats})"

def _add_grid(ax):
    ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.5)

def plot_moves_ecdf(df: pd.DataFrame, outdir: str):
    solvers = sorted(df["solver"].unique())
    fig, ax = plt.subplots()
    for s in solvers:
        x = np.sort(df.loc[df["solver"] == s, "moves"].values)
        y = np.arange(1, len(x)+1) / len(x)
        ax.plot(x, y, drawstyle="steps-post", label=s)
    # --- mean markers ---
    for s in solvers:
        vals = df.loc[df["solver"] == s, "moves"].values
        mean_val = np.mean(vals)
        ax.axvline(mean_val, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(mean_val, 0.02, f"{s} mean={mean_val:.2f}",
                rotation=90, va="bottom", ha="right", fontsize=8, alpha=0.7)
    ax.set_title(_title_with_params("ECDF – Moves per solver", df))
    ax.set_xlabel("Moves (≤ m)")
    ax.set_ylabel("Fraction of games")
    _add_grid(ax)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "moves_ecdf.png"), dpi=150)

def plot_moves_pmf(df: pd.DataFrame, outdir: str):
    solvers = sorted(df["solver"].unique())
    all_moves = np.unique(df["moves"].values)
    fig, ax = plt.subplots()
    idx = np.arange(len(all_moves))
    width = 0.8 / max(1, len(solvers))
    for j, s in enumerate(solvers):
        mv = df.loc[df["solver"] == s, "moves"].values
        counts = np.array([np.sum(mv == m) for m in all_moves], dtype=float)
        probs = counts / counts.sum() if counts.sum() > 0 else counts
        ax.bar(idx + j*width, probs, width=width, align="edge", label=s)
    ax.set_title(_title_with_params("PMF – Moves distribution", df))
    ax.set_xlabel("Moves")
    ax.set_ylabel("Probability")
    ax.set_xticks(idx + (len(solvers)-1)*width/2)
    ax.set_xticklabels([str(m) for m in all_moves])
    _add_grid(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "moves_pmf.png"), dpi=150)

def plot_time_ecdf(df: pd.DataFrame, outdir: str, logx: bool, suffix: str):
    solvers = sorted(df["solver"].unique())
    fig, ax = plt.subplots()
    for s in solvers:
        t = np.sort(df.loc[df["solver"] == s, "time_s"].values)
        y = np.arange(1, len(t)+1) / len(t)
        ax.plot(t, y, drawstyle="steps-post", label=s)
    ax.set_title(_title_with_params("ECDF – Runtime per solver", df) + (" (log-x)" if logx else ""))
    ax.set_xlabel("Time [s]" + (" (log)" if logx else ""))
    ax.set_ylabel("Fraction of games")
    if logx:
        ax.set_xscale("log")
    _add_grid(ax)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"time_ecdf{suffix}.png"), dpi=150)

def main():
    ap = argparse.ArgumentParser(description="Plot Mastermind results")
    ap.add_argument("--csv", default="results.csv", help="Pfad zur CSV (Standard: results.csv)")
    ap.add_argument("--outdir", default="plots", help="Output-Verzeichnis")
    ap.add_argument("--skip-pmf", action="store_true", help="PMF-Moves-Plot überspringen")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = read_results_robust(args.csv)

    plot_moves_ecdf(df, args.outdir)
    if not args.skip_pmf:
        plot_moves_pmf(df, args.outdir)

    plot_time_ecdf(df, args.outdir, logx=False, suffix="")
    plot_time_ecdf(df, args.outdir, logx=True,  suffix="_log")

    print("Saved plots in:", os.path.abspath(args.outdir))
    print(" - moves_ecdf.png")
    if not args.skip_pmf: print(" - moves_pmf.png")
    print(" - time_ecdf.png")
    print(" - time_ecdf_log.png")

if __name__ == "__main__":
    main()
