#!/usr/bin/env python3
"""
Comprehensive N‑Body Analysis Script
Run this on your local machine after copying the CSV files from the server.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import os
import sys

# ----------------------------------------------------------------------
# 1. Scaling plot (benchmark.csv)
# ----------------------------------------------------------------------
def plot_scaling():
    if not os.path.exists('benchmark.csv'):
        print("⚠️  benchmark.csv not found – skipping scaling plot.")
        return

    df = pd.read_csv('benchmark.csv')
    df = df.sort_values('N')

    plt.figure(figsize=(8, 5))
    plt.loglog(df['N'], df['gips'], 'bo-', linewidth=2, markersize=8,
               label='GIPS (Giga‑Interactions/s)')
    plt.xlabel('Number of Bodies (N)')
    plt.ylabel('GIPS')
    plt.title('N‑Body Scalability on RTX 3060 (Tiled Kernel)')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('scaling.png', dpi=150)
    print("✓ Saved scaling.png")

# ----------------------------------------------------------------------
# 2. Energy conservation (energy.csv)
# ----------------------------------------------------------------------
def plot_energy():
    if not os.path.exists('energy.csv'):
        print("⚠️  energy.csv not found – skipping energy plot.")
        return

    energy = pd.read_csv('energy.csv')
    # columns: step,kinetic,potential,total,drift

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(energy['step'], energy['kinetic'], 'r-', label='Kinetic')
    plt.plot(energy['step'], energy['potential'], 'b-', label='Potential')
    plt.plot(energy['step'], energy['total'], 'k--', label='Total')
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.title('Energy Components')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(energy['step'], energy['drift'] * 100, 'g-', linewidth=1)
    plt.xlabel('Step')
    plt.ylabel('Drift (%)')
    plt.title('Relative Energy Drift')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('energy_drift.png', dpi=150)
    print("✓ Saved energy_drift.png")

# ----------------------------------------------------------------------
# 3. Kernel comparison (kernel_comparison.csv)
# ----------------------------------------------------------------------
def plot_kernel_comparison():
    if not os.path.exists('kernel_comparison.csv'):
        print("⚠️  kernel_comparison.csv not found – skipping kernel comparison plot.")
        return

    df = pd.read_csv('kernel_comparison.csv')
    df_naive = df[df['kernel'] == 'Naive'].sort_values('N')
    df_tiled = df[df['kernel'] == 'Tiled'].sort_values('N')

    # Merge to compute speedup
    merged = pd.merge(df_naive, df_tiled, on='N', suffixes=('_naive', '_tiled'))
    merged['speedup'] = merged['avg_time_ms_naive'] / merged['avg_time_ms_tiled']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Execution time
    ax = axes[0]
    ax.loglog(df_naive['N'], df_naive['avg_time_ms'], 'ro-', label='Naive', linewidth=2, markersize=8)
    ax.loglog(df_tiled['N'], df_tiled['avg_time_ms'], 'bs-', label='Tiled', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Bodies (N)')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title('Kernel Execution Time')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    # Right: Speedup
    ax = axes[1]
    ax.semilogx(merged['N'], merged['speedup'], 'g*-', linewidth=2, markersize=10)
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Baseline (Naive)')
    ax.set_xlabel('Number of Bodies (N)')
    ax.set_ylabel('Speedup (Naive / Tiled)')
    ax.set_title('Tiled Kernel Speedup over Naive')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('kernel_comparison.png', dpi=150)
    print("✓ Saved kernel_comparison.png")

# ----------------------------------------------------------------------
# 4. 3D animation (simulation.csv)
# ----------------------------------------------------------------------
def create_animation(save_gif=True, save_mp4=True):
    if not os.path.exists('simulation.csv'):
        print("⚠️  simulation.csv not found – skipping animation.")
        return

    print("Loading simulation data...")
    df = pd.read_csv('simulation.csv')
    steps = sorted(df['step'].unique())
    n_particles = df['particle_id'].nunique()
    print(f"Found {len(steps)} time steps, {n_particles} particles.")

    # Fixed axis limits
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    z_min, z_max = df['z'].min(), df['z'].max()
    pad = 0.2
    xlim = (x_min - pad, x_max + pad)
    ylim = (y_min - pad, y_max + pad)
    zlim = (z_min - pad, z_max + pad)

    # Galaxy colours (if 'galaxy' column exists)
    if 'galaxy' in df.columns:
        color_map = {0: 'red', 1: 'cyan'}
    else:
        color_map = None

    # Set up figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Initial empty scatter
    scat = ax.scatter([], [], [], s=1, alpha=0.7)

    def animate(frame):
        step = steps[frame]
        data = df[df['step'] == step]
        x = data['x'].values
        y = data['y'].values
        z = data['z'].values

        if color_map is not None:
            colours = data['galaxy'].map(color_map).values
            scat.set_color(colours)
        else:
            scat.set_color('cyan')

        scat._offsets3d = (x, y, z)
        ax.set_title(f'Step {step}')
        return scat,

    print("Creating animation...")
    ani = FuncAnimation(fig, animate, frames=len(steps), interval=50, blit=False)

    if save_gif:
        print("Saving as GIF...")
        writer = PillowWriter(fps=20)
        ani.save('galaxy_collision.gif', writer=writer, dpi=100)
        print("✓ Saved galaxy_collision.gif")

    if save_mp4:
        try:
            print("Saving as MP4...")
            writer = FFMpegWriter(fps=20, metadata=dict(title='Galaxy Collision'))
            ani.save('galaxy_collision.mp4', writer=writer, dpi=100)
            print("✓ Saved galaxy_collision.mp4")
        except Exception as e:
            print("⚠️  Could not save MP4 (ffmpeg missing?):", e)

    plt.close(fig)  # avoid showing the static plot

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("N‑Body Analysis Tool")
    print("=" * 60)

    # Command line flags to skip certain plots
    do_scaling = '--no-scaling' not in sys.argv
    do_energy  = '--no-energy' not in sys.argv
    do_kernel  = '--no-kernel' not in sys.argv
    do_anim    = '--no-anim' not in sys.argv

    if do_scaling:
        plot_scaling()
    if do_energy:
        plot_energy()
    if do_kernel:
        plot_kernel_comparison()
    if do_anim:
        create_animation(save_gif=True, save_mp4=True)

    print("\nAll done! Generated files:")
    for f in ['scaling.png', 'energy_drift.png', 'kernel_comparison.png',
              'galaxy_collision.gif', 'galaxy_collision.mp4']:
        if os.path.exists(f):
            print(f"  - {f}")
