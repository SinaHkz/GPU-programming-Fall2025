import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Data (copied from your report)
# -----------------------
rows = [
    # Version, Batch, Block, Tile, GPU_Util_percent, GPU_Active_us
    ("Baseline (Naive)", 64, 256, "16x16", 84.5, 4199899),
    ("Baseline (Naive)", 32, 256, "16x16", 88.5, 4498256),
    ("Baseline (Naive)", 8, 1024, "16x16", 71.6, 6611119),
    ("Baseline (Naive)", 8, 256, "16x16", 62.5, 5077420),

    ("Opt1 (Tiled Conv)", 64, 256, "16x16", 93.6, 4181863),
    ("Opt1 (Tiled Conv)", 8, 512, "16x16", 67.9, 5462704),
    ("Opt1 (Tiled Conv)", 8, 1024, "16x16", 83.2, 6524291),
    ("Opt1 (Tiled Conv)", 8, 256, "16x16", 62.3, 5009579),

    ("Opt2 (Fused)", 64, 256, "16x16", 83.51, 4296950),
    ("Opt2 (Fused)", 8, 512, "16x16", 71.7, 5734764),
    ("Opt2 (Fused)", 8, 1024, "16x16", 67.17, 6558107),
    ("Opt2 (Fused)", 8, 256, "16x16", 68.0, 4978762),
    ("Opt2 (Fused)", 32, 256, "16x16", 73.1, 4475917),
    ("Opt2 (Fused)", 64, 512, "16x16", 75.1, 4235282),

    ("Opt2 (Fused) + Tile=8x8", 64, 512, "8x8", 73.6, 4196260),
    ("Opt2 (Fused) + Tile=8x8", 8, 256, "8x8", 62.9, 5002725),

    ("Opt4 (Streams)", 64, 256, "16x16", 38.9, 428901),
    ("Opt4 (Streams)", 8, 512, "16x16", 34.25, 1584978),
    ("Opt4 (Streams)", 8, 1024, "16x16", 50.7, 2519363),
    ("Opt4 (Streams)", 8, 256, "16x16", 32.2, 1270796),
    ("Opt4 (Streams)", 32, 256, "16x16", 46.2, 590936),
    ("Opt4 (Streams)", 64, 512, "16x16", 30.0, 456422),
]

df = pd.DataFrame(rows, columns=[
    "version", "batch", "block", "tile", "gpu_util_percent", "gpu_active_us"
])

# Optional: convert active time to ms for plotting
df["gpu_active_ms"] = df["gpu_active_us"] / 1000.0

# Save CSV for your report / future edits
df.to_csv("profiling_results.csv", index=False)
print("Saved: profiling_results.csv")

# -----------------------
# Plot 1: Active time vs Block (per batch)
# -----------------------
for batch in sorted(df["batch"].unique()):
    sub = df[df["batch"] == batch].copy()
    if sub.empty:
        continue

    plt.figure()
    for ver in sorted(sub["version"].unique()):
        s = sub[sub["version"] == ver].sort_values("block")
        # Use markers to make sparse points readable
        plt.plot(s["block"], s["gpu_active_ms"], marker="o", label=ver)

    plt.xlabel("Block size (threads)")
    plt.ylabel("GPU Active Time (ms)")
    plt.title(f"GPU Active Time vs Block Size (Batch={batch})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"active_time_batch_{batch}.png", dpi=200)

# -----------------------
# Plot 2: Utilization vs Block (per batch)
# -----------------------
for batch in sorted(df["batch"].unique()):
    sub = df[df["batch"] == batch].copy()
    if sub.empty:
        continue

    plt.figure()
    for ver in sorted(sub["version"].unique()):
        s = sub[sub["version"] == ver].sort_values("block")
        plt.plot(s["block"], s["gpu_util_percent"], marker="o", label=ver)

    plt.xlabel("Block size (threads)")
    plt.ylabel("GPU Utilization (%)")
    plt.title(f"GPU Utilization vs Block Size (Batch={batch})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"gpu_util_batch_{batch}.png", dpi=200)

# -----------------------
# Plot 3: Heatmap (Active time) per version (Batch x Block)
# -----------------------
for ver in sorted(df["version"].unique()):
    sub = df[df["version"] == ver].copy()
    if sub.empty:
        continue

    pivot = sub.pivot_table(index="batch", columns="block", values="gpu_active_ms", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)

    plt.figure()
    img = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(img, label="GPU Active Time (ms)")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Block size (threads)")
    plt.ylabel("Batch size")
    plt.title(f"Heatmap: GPU Active Time (ms) - {ver}")
    plt.tight_layout()
    # filename-safe
    safe = ver.replace(" ", "_").replace("/", "_").replace("+", "plus").replace("=", "")
    plt.savefig(f"heatmap_active_time_{safe}.png", dpi=200)

print("Saved plots: active_time_batch_*.png, gpu_util_batch_*.png, heatmap_active_time_*.png")
