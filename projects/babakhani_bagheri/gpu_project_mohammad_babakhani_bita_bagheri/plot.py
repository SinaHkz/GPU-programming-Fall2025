import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "ncu_metrics_summary.csv"
OUT_DIR = BASE_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_COLS = [
    "N",
    "branch_efficiency_pct",
    "dram_throughput_pct",
    "sm_throughput_pct",
    "time_us",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot NCU metric summary.")
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Path to ncu_metrics_summary.csv")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="Output directory for plots")
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    return parser.parse_args()


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def warn_suspicious_jumps(df: pd.DataFrame):
    for col in ("sm_throughput_pct", "dram_throughput_pct", "branch_efficiency_pct"):
        ratios = []
        vals = df[col].tolist()
        ns = df["N"].tolist()
        for i in range(1, len(vals)):
            prev = vals[i - 1]
            curr = vals[i]
            if prev <= 0 or curr <= 0:
                continue
            ratio = max(curr / prev, prev / curr)
            if ratio > 20:
                ratios.append((ns[i - 1], ns[i], prev, curr, ratio))
        if ratios:
            print(f"\n[WARN] suspicious jump(s) in {col}:")
            for n0, n1, v0, v1, ratio in ratios:
                print(f"  N={n0} -> N={n1}: {v0:.4g} -> {v1:.4g} (x{ratio:.1f})")


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=NUM_COLS).sort_values("N").reset_index(drop=True)

    if df.empty:
        raise ValueError("Input CSV has no valid numeric rows.")

    print("\nLoaded Data:")
    print(df.to_string(index=False))

    warn_suspicious_jumps(df)

    df["time_ms"] = df["time_us"] / 1000.0
    df["pair_interactions"] = df["N"] * (df["N"] - 1)
    df["time_per_interaction_ns"] = (df["time_us"] * 1000.0) / df["pair_interactions"]

    # 1) TIME vs N
    plt.figure(figsize=(7, 4.2))
    plt.plot(df["N"], df["time_ms"], marker="o", linewidth=2, label="Kernel Time (ms)")
    plt.xscale("log", base=2)
    plt.xlabel("N (log2 scale)")
    plt.ylabel("Time (ms)")
    plt.title("Kernel Time vs N (compute_accel_tiled)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(args.out / "time_vs_N.png")

    # 2) TIME per interaction vs N
    plt.figure(figsize=(7, 4.2))
    plt.plot(df["N"], df["time_per_interaction_ns"], marker="o", linewidth=2)
    plt.xscale("log", base=2)
    plt.xlabel("N (log2 scale)")
    plt.ylabel("Time per interaction (ns)")
    plt.title("Time per Interaction vs N")
    plt.grid(True, alpha=0.3)
    savefig(args.out / "time_per_interaction_vs_N.png")

    # 3) SM / DRAM throughput vs N (linear)
    plt.figure(figsize=(7, 4.2))
    plt.plot(df["N"], df["sm_throughput_pct"], marker="o", linewidth=2, label="SM Throughput %")
    plt.plot(df["N"], df["dram_throughput_pct"], marker="o", linewidth=2, label="DRAM Throughput %")
    plt.xscale("log", base=2)
    plt.xlabel("N (log2 scale)")
    plt.ylabel("% of peak sustained")
    plt.title("SM vs DRAM Throughput vs N (Linear Scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(args.out / "sm_dram_throughput_vs_N.png")

    # 3b) SM / DRAM throughput vs N (symlog for mixed scales)
    plt.figure(figsize=(7, 4.2))
    plt.plot(df["N"], df["sm_throughput_pct"], marker="o", linewidth=2, label="SM Throughput %")
    plt.plot(df["N"], df["dram_throughput_pct"], marker="o", linewidth=2, label="DRAM Throughput %")
    plt.xscale("log", base=2)
    plt.yscale("symlog", linthresh=1.0)
    plt.xlabel("N (log2 scale)")
    plt.ylabel("% of peak sustained (symlog y)")
    plt.title("SM vs DRAM Throughput vs N (Symlog Scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(args.out / "sm_dram_throughput_vs_N_symlog.png")

    # 4) Branch efficiency vs N
    plt.figure(figsize=(7, 4.2))
    plt.plot(df["N"], df["branch_efficiency_pct"], marker="o", linewidth=2)
    plt.xscale("log", base=2)
    plt.ylim(0, 100)
    plt.xlabel("N (log2 scale)")
    plt.ylabel("Branch efficiency (%)")
    plt.title("Branch Efficiency vs N")
    plt.grid(True, alpha=0.3)
    savefig(args.out / "branch_efficiency_vs_N.png")

    # 5) Scatter: Time vs SM throughput
    plt.figure(figsize=(7, 4.2))
    plt.scatter(df["sm_throughput_pct"], df["time_ms"], s=45)
    for _, r in df.iterrows():
        plt.annotate(str(int(r["N"])), (r["sm_throughput_pct"], r["time_ms"]), fontsize=8)
    plt.xlabel("SM Throughput (%)")
    plt.ylabel("Time (ms)")
    plt.title("Time vs SM Throughput")
    plt.grid(True, alpha=0.3)
    savefig(args.out / "time_vs_sm_scatter.png")

    # 6) Scatter: Time vs DRAM throughput
    plt.figure(figsize=(7, 4.2))
    plt.scatter(df["dram_throughput_pct"], df["time_ms"], s=45)
    for _, r in df.iterrows():
        plt.annotate(str(int(r["N"])), (r["dram_throughput_pct"], r["time_ms"]), fontsize=8)
    plt.xlabel("DRAM Throughput (%)")
    plt.ylabel("Time (ms)")
    plt.title("Time vs DRAM Throughput")
    plt.grid(True, alpha=0.3)
    savefig(args.out / "time_vs_dram_scatter.png")

    # 7) Per-N bars with dynamic ylim to avoid misleading tiny bars
    for _, r in df.iterrows():
        n_val = int(r["N"])
        labels = ["BranchEff", "DRAM%", "SM%"]
        values = [r["branch_efficiency_pct"], r["dram_throughput_pct"], r["sm_throughput_pct"]]

        vmax = max(values)
        y_top = 110 if vmax > 80 else max(10, vmax * 1.25)

        plt.figure(figsize=(5.6, 4.0))
        bars = plt.bar(labels, values)
        for bar, v in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        plt.ylim(0, y_top)
        plt.title(f"Key Metrics @ N={n_val}")
        plt.ylabel("Percent")
        plt.grid(True, axis="y", alpha=0.3)
        savefig(args.out / f"metrics_bar_N{n_val}.png")

    # 8) Correlation matrix
    corr_cols = ["branch_efficiency_pct", "dram_throughput_pct", "sm_throughput_pct", "time_ms"]
    corr = df[corr_cols].corr(numeric_only=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(corr, interpolation="nearest", cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=30, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Matrix (Metrics vs Time)")
    plt.colorbar()
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")
    savefig(args.out / "correlation_matrix.png")

    if args.show:
        plt.show()

    print(f"\nSaved plots to: {args.out}")


if __name__ == "__main__":
    main()
