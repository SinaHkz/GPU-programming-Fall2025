"""
Enhanced N-Body Simulation Visualization and Analysis
Author: GPU Project - Solar System Dynamics
Hardware: NVIDIA RTX 4070 Mobile

Advanced Features:
1. 3D animated trajectory visualization
2. Comet trajectory analysis with close approach detection
3. Kernel performance comparison
4. Energy conservation detailed analysis
5. Orbital stability metrics
6. Interactive plots and animations
7. Comprehensive performance scaling analysis
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['animation.html'] = 'html5'

# Constants
AU = 1.496e11  # Astronomical Unit in meters
G = 6.67430e-11  # Gravitational constant

class EnhancedNBodyVisualizer:
    """Enhanced N-body simulation visualization with advanced analytics"""
    
    def __init__(self,output_dir="output"):
        """Load simulation data from specified output directory"""
        self.output_dir = output_dir
        print("="*70)
        print(f"Loading Enhanced N-Body Simulation Data from: {output_dir}")
        print("="*70)

        # File paths using output_dir
        positions_file = os.path.join(output_dir, 'positions.csv')
        energy_file = os.path.join(output_dir, 'energy.csv')
        performance_file = os.path.join(output_dir, 'performance.csv')
        comet_file = os.path.join(output_dir, 'comet_track.csv')
        benchmark_file = os.path.join(output_dir, 'kernel_benchmark.csv')

        # Load positions
        if os.path.exists(positions_file):
            self.positions = pd.read_csv(positions_file)
            print(f"✓ Loaded {len(self.positions)} position records")
        else:
            print(f"✗ Warning: {positions_file} not found")
            self.positions = None

        # Load energy
        if os.path.exists(energy_file):
            self.energy = pd.read_csv(energy_file)
            print(f"✓ Loaded {len(self.energy)} energy records")
        else:
            print(f"✗ Warning: {energy_file} not found")
            self.energy = None

        # Load performance
        if os.path.exists(performance_file):
            self.performance = pd.read_csv(performance_file)
            print(f"✓ Loaded {len(self.performance)} performance records")
        else:
            print(f"✗ Warning: {performance_file} not found")
            self.performance = None

        # Load comet trajectory
        if os.path.exists(comet_file):
            self.comet = pd.read_csv(comet_file)
            print(f"✓ Loaded {len(self.comet)} comet trajectory points")
        else:
            print(f"✗ Warning: {comet_file} not found")
            self.comet = None

        # Load kernel benchmark
        if os.path.exists(benchmark_file):
            self.benchmark = pd.read_csv(benchmark_file)
            print(f"✓ Loaded kernel benchmark data")
        else:
            print(f"✗ Warning: {benchmark_file} not found")
            self.benchmark = None

        # Get metadata
        if self.positions is not None:
            self.n_bodies = self.positions['body_id'].nunique()
            self.times = self.positions['time'].unique()
            self.n_timesteps = len(self.times)

            print(f"\nSimulation Details:")
            print(f"  Bodies: {self.n_bodies}")
            print(f"  Timesteps: {self.n_timesteps}")
            print(f"  Duration: {self.times[-1]/86400:.1f} days")

            # Identify central mass
            body_0_data = self.positions[self.positions['body_id'] == 0].iloc[0]
            self.central_mass = body_0_data['mass']
            print(f"  Central mass: {self.central_mass:.2e} kg")

        print("="*70 + "\n")

    def plot_kernel_comparison(self, save=True):
        """Compare different kernel implementations"""
        if self.benchmark is None:
            print("No benchmark data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        kernels = self.benchmark['kernel'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(kernels)))

        # Execution time comparison
        ax1 = axes[0, 0]
        x_pos = np.arange(len(kernels))
        times = [self.benchmark[self.benchmark['kernel'] == k]['avg_time_ms'].values[0]
                 for k in kernels]
        bars = ax1.bar(x_pos, times, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(kernels, rotation=15, ha='right')
        ax1.set_ylabel('Average Execution Time (ms)')
        ax1.set_title('Kernel Execution Time Comparison')
        ax1.grid(True, alpha=0.3, axis='y')

        # Add values on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f}ms', ha='center', va='bottom', fontsize=9)

        # Speedup comparison
        ax2 = axes[0, 1]
        baseline = times[0]  # Naive kernel as baseline
        speedups = [baseline / t for t in times]
        bars = ax2.bar(x_pos, speedups, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(kernels, rotation=15, ha='right')
        ax2.set_ylabel('Speedup vs Naive')
        ax2.set_title('Performance Speedup (Higher is Better)')
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()

        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{speedup:.2f}x', ha='center', va='bottom', fontsize=9)

        # GFLOPS comparison
        ax3 = axes[1, 0]
        gflops = [self.benchmark[self.benchmark['kernel'] == k]['gflops'].values[0]
                  for k in kernels]
        bars = ax3.bar(x_pos, gflops, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(kernels, rotation=15, ha='right')
        ax3.set_ylabel('GFLOPS')
        ax3.set_title('Computational Throughput')
        ax3.grid(True, alpha=0.3, axis='y')

        for bar, gflop in zip(bars, gflops):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{gflop:.1f}', ha='center', va='bottom', fontsize=9)

        # Occupancy comparison
        ax4 = axes[1, 1]
        # بررسی وجود ستون occupancy، در غیر این صورت مقدار 0 قرار داده می‌شود
        if 'occupancy_percent' in self.benchmark.columns:
            occupancy = [self.benchmark[self.benchmark['kernel'] == k]['occupancy_percent'].values[0] for k in kernels]
        else:
            occupancy = [0 for _ in kernels]
        bars = ax4.bar(x_pos, occupancy, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(kernels, rotation=15, ha='right')
        ax4.set_ylabel('Occupancy (%)')
        ax4.set_title('GPU Occupancy')
        ax4.set_ylim([0, 105])
        ax4.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Maximum')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.legend()

        for bar, occ in zip(bars, occupancy):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{occ}%', ha='center', va='bottom', fontsize=9)

        fig.suptitle('CUDA Kernel Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.output_dir,'kernel_comparison.png'), dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir}/kernel_comparison.png")

        # plt.show()

    def plot_energy_conservation_detailed(self, save=True):
    #  """Detailed energy conservation analysis with robust handling"""
        if self.energy is None:
            print("No energy data available")
            return

        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)

        time_days = self.energy['time'] / 86400.0

        # Extract columns
        if all(col in self.energy.columns for col in ['kinetic', 'potential']):
            total_energy = self.energy['total_energy']
            ke = self.energy['kinetic']
            pe = self.energy['potential']
        else:
            total_energy = self.energy['total_energy']
            ke = None
            pe = None

        # Plot 1: Total energy over time
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time_days, total_energy, 'b-', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Total Energy (J)')
        ax1.set_title('Total System Energy Over Time', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        # اگر KE و PE جداگانه موجود باشد، پلات‌های جداگانه
        if ke is not None and pe is not None:
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.plot(time_days, ke, 'r-', linewidth=2, label='Kinetic')
            ax2.plot(time_days, pe, 'g-', linewidth=2, label='Potential')
            ax2.plot(time_days, total_energy, 'b-', linewidth=2, label='Total')
            ax2.set_xlabel('Time (days)')
            ax2.set_ylabel('Energy (J)')
            ax2.set_title('Kinetic vs Potential vs Total Energy', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Relative energy drift over time (robust calculation)
        initial_energy = total_energy.iloc[0]
        if np.isnan(initial_energy) or initial_energy == 0:
            relative_drift = np.zeros_like(total_energy.values)
        else:
            relative_drift = (total_energy - initial_energy) / np.abs(initial_energy)

        # Filter finite values
        mask = np.isfinite(relative_drift)
        relative_drift_clean = relative_drift[mask]
        time_drift_clean = time_days[mask]

        ax4 = fig.add_subplot(gs[2, :])
        ax4.plot(time_drift_clean, relative_drift_clean, 'm-', linewidth=1.5, alpha=0.8)
        ax4.set_xlabel('Time (days)')
        ax4.set_ylabel('Relative Energy Drift')
        ax4.set_title('Relative Energy Drift Over Time', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        # Histogram of relative drift - فقط اگر داده معتبر و variation کافی باشد
        ax5 = fig.add_subplot(gs[3, :])
        if len(relative_drift_clean) > 1 and np.std(relative_drift_clean) > 1e-15:
            ax5.hist(relative_drift_clean, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            ax5.set_xlabel('Relative Energy Drift')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Distribution of Relative Energy Drift', fontweight='bold')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No significant energy drift detected\n(Perfect conservation or insufficient variation/data)',
                    transform=ax5.transAxes, ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8))
            ax5.set_title('Energy Drift Histogram (No variation)', fontweight='bold')
            ax5.axis('off')

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.output_dir, 'energy_conservation_detailed.png'), dpi=300, bbox_inches='tight')
            print("✓ Saved: energy_conservation_detailed.png")

        # plt.show()

    def plot_comet_analysis(self, save=True):
        """Analyze and visualize comet trajectory"""
        if self.comet is None:
            print("No comet data available")
            return

        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        time_days = self.comet['time'] / 86400.0
        distance_au = self.comet['distance'] / AU

        # 3D trajectory
        ax1 = fig.add_subplot(gs[0:2, 0:2], projection='3d')
        
        # Color by time
        scatter = ax1.scatter(self.comet['x']/AU, self.comet['y']/AU, self.comet['z']/AU,
                            c=time_days, cmap='plasma', s=20, alpha=0.8)
        
        # Mark start and end
        ax1.scatter([self.comet['x'].iloc[0]/AU], [self.comet['y'].iloc[0]/AU],
                   [self.comet['z'].iloc[0]/AU], color='green', s=200, marker='o',
                   label='Start', edgecolor='black', linewidth=2)
        ax1.scatter([self.comet['x'].iloc[-1]/AU], [self.comet['y'].iloc[-1]/AU],
                   [self.comet['z'].iloc[-1]/AU], color='red', s=200, marker='s',
                   label='End', edgecolor='black', linewidth=2)
        
        # Sun at center
        ax1.scatter([0], [0], [0], color='yellow', s=300, marker='*',
                   label='Sun', edgecolor='orange', linewidth=2)
        
        ax1.set_xlabel('X (AU)')
        ax1.set_ylabel('Y (AU)')
        ax1.set_zlabel('Z (AU)')
        ax1.set_title('Comet 3D Trajectory', fontweight='bold', fontsize=14)
        ax1.legend()
        plt.colorbar(scatter, ax=ax1, label='Time (days)', pad=0.1)

        # Distance vs time
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(time_days, distance_au, 'b-', linewidth=2)
        
        # Find perihelion and aphelion
        perihelion_idx = distance_au.idxmin()
        aphelion_idx = distance_au.idxmax()
        
        ax2.plot(time_days.iloc[perihelion_idx], distance_au.iloc[perihelion_idx],
                'ro', markersize=12, label=f'Perihelion: {distance_au.iloc[perihelion_idx]:.2f} AU')
        ax2.plot(time_days.iloc[aphelion_idx], distance_au.iloc[aphelion_idx],
                'go', markersize=12, label=f'Aphelion: {distance_au.iloc[aphelion_idx]:.2f} AU')
        
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Distance from Sun (AU)')
        ax2.set_title('Orbital Distance', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)

        # Speed vs time
        ax3 = fig.add_subplot(gs[1, 2])
        speed_km_s = self.comet['speed'] / 1000.0
        ax3.plot(time_days, speed_km_s, 'r-', linewidth=2)
        
        # Mark max speed at perihelion
        ax3.plot(time_days.iloc[perihelion_idx], speed_km_s.iloc[perihelion_idx],
                'ro', markersize=12, label=f'Max: {speed_km_s.iloc[perihelion_idx]:.1f} km/s')
        
        ax3.set_xlabel('Time (days)')
        ax3.set_ylabel('Speed (km/s)')
        ax3.set_title('Orbital Velocity', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # XY projection
        ax4 = fig.add_subplot(gs[2, 0])
        scatter = ax4.scatter(self.comet['x']/AU, self.comet['y']/AU,
                            c=time_days, cmap='plasma', s=10, alpha=0.6)
        ax4.plot(0, 0, 'y*', markersize=20)
        ax4.set_xlabel('X (AU)')
        ax4.set_ylabel('Y (AU)')
        ax4.set_title('XY Plane', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')

        # XZ projection
        ax5 = fig.add_subplot(gs[2, 1])
        scatter = ax5.scatter(self.comet['x']/AU, self.comet['z']/AU,
                            c=time_days, cmap='plasma', s=10, alpha=0.6)
        ax5.plot(0, 0, 'y*', markersize=20)
        ax5.set_xlabel('X (AU)')
        ax5.set_ylabel('Z (AU)')
        ax5.set_title('XZ Plane', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.set_aspect('equal')

        # Speed vs distance (phase space)
        ax6 = fig.add_subplot(gs[2, 2])
        scatter = ax6.scatter(distance_au, speed_km_s, c=time_days,
                            cmap='plasma', s=20, alpha=0.6)
        ax6.set_xlabel('Distance (AU)')
        ax6.set_ylabel('Speed (km/s)')
        ax6.set_title('Phase Space', fontweight='bold')
        ax6.grid(True, alpha=0.3)

        # Orbital statistics
        eccentricity = (distance_au.max() - distance_au.min()) / (distance_au.max() + distance_au.min())
        
        stats_text = (f'Perihelion: {distance_au.min():.3f} AU\n'
                     f'Aphelion: {distance_au.max():.3f} AU\n'
                     f'Eccentricity: {eccentricity:.3f}\n'
                     f'Max speed: {speed_km_s.max():.1f} km/s')
        
        fig.text(0.98, 0.02, stats_text, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                verticalalignment='bottom', horizontalalignment='right')

        fig.suptitle('Comet Trajectory Analysis', fontsize=18, fontweight='bold')

        if save:
            plt.savefig(os.path.join(self.output_dir,'comet_analysis.png'), dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir}comet_analysis.png")

        # plt.show()

    def plot_performance_scaling(self, save=True):
        """Analyze performance metrics over time"""
        if self.performance is None:
            print("No performance data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        steps = self.performance['step']
        time_days = steps * 3600 / 86400.0  # Convert to days

        # Kernel time
        ax1 = axes[0, 0]
        ax1.plot(time_days, self.performance['kernel_time_ms'], 'b-', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Simulation Time (days)')
        ax1.set_ylabel('Kernel Time (ms)')
        ax1.set_title('Kernel Execution Time', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add rolling average
        window = max(10, len(self.performance) // 20)
        rolling_avg = self.performance['kernel_time_ms'].rolling(window=window).mean()
        ax1.plot(time_days, rolling_avg, 'r--', linewidth=2, label=f'{window}-step avg')
        ax1.legend()

        # Interactions per second
        ax2 = axes[0, 1]
        ax2.plot(time_days, self.performance['interactions_per_sec'],
                'g-', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Simulation Time (days)')
        ax2.set_ylabel('Interactions/second')
        ax2.set_title('Computational Throughput', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        # GFLOPS
        ax3 = axes[1, 0]
        ax3.plot(time_days, self.performance['gflops'], 'r-', linewidth=2, alpha=0.7)
        ax3.set_xlabel('Simulation Time (days)')
        ax3.set_ylabel('GFLOPS')
        ax3.set_title('Floating Point Performance', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Performance statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats = {
            'Mean kernel time': f"{self.performance['kernel_time_ms'].mean():.3f} ms",
            'Std kernel time': f"{self.performance['kernel_time_ms'].std():.3f} ms",
            'Mean interactions/s': f"{self.performance['interactions_per_sec'].mean():.3e}",
            'Mean GFLOPS': f"{self.performance['gflops'].mean():.2f}",
            'Max GFLOPS': f"{self.performance['gflops'].max():.2f}",
            'Performance stability': f"{(1 - self.performance['kernel_time_ms'].std() / self.performance['kernel_time_ms'].mean()) * 100:.1f}%"
        }
        
        stats_text = '\n'.join([f'{k}: {v}' for k, v in stats.items()])
        ax4.text(0.1, 0.9, 'Performance Statistics', fontsize=16, fontweight='bold',
                verticalalignment='top')
        ax4.text(0.1, 0.75, stats_text, fontsize=12, verticalalignment='top',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle('Performance Metrics Over Time', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.output_dir,'performance_analysis.png'), dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir}performance_analysis.png")

        # plt.show()

    def plot_3d_trajectories_enhanced(self, num_bodies_to_plot=None, save=True):
        """Enhanced 3D orbital trajectory visualization"""
        if self.positions is None:
            print("No position data available")
            return

        if num_bodies_to_plot is None:
            num_bodies_to_plot = min(20, self.n_bodies)

        fig = plt.figure(figsize=(18, 14))
        ax = fig.add_subplot(111, projection='3d')

        # Custom colormap
        colors = plt.cm.rainbow(np.linspace(0, 1, num_bodies_to_plot))

        # Plot central body
        central = self.positions[self.positions['body_id'] == 0]
        ax.plot(central['x']/AU, central['y']/AU, central['z']/AU,
               'yo', markersize=20, label='Sun', zorder=100,
               markeredgecolor='orange', markeredgewidth=2)

        # Plot trajectories
        for i in range(1, num_bodies_to_plot):
            body_data = self.positions[self.positions['body_id'] == i]

            if len(body_data) == 0:
                continue

            # Determine line style based on body type
            if i <= 5:  # Planets
                linewidth = 2.0
                alpha = 0.8
            else:  # Asteroids
                linewidth = 1.0
                alpha = 0.4

            # Plot trajectory with gradient
            points = ax.plot(body_data['x']/AU, body_data['y']/AU, body_data['z']/AU,
                   color=colors[i], alpha=alpha, linewidth=linewidth,
                   label=f'Body {i}' if i <= 10 else None)

            # Mark initial and final positions
            ax.plot([body_data['x'].iloc[0]/AU],
                   [body_data['y'].iloc[0]/AU],
                   [body_data['z'].iloc[0]/AU],
                   'o', color=colors[i], markersize=6, alpha=0.8,
                   markeredgecolor='black', markeredgewidth=0.5)

            ax.plot([body_data['x'].iloc[-1]/AU],
                   [body_data['y'].iloc[-1]/AU],
                   [body_data['z'].iloc[-1]/AU],
                   's', color=colors[i], markersize=6, alpha=0.8,
                   markeredgecolor='black', markeredgewidth=0.5)

        ax.set_xlabel('X Position (AU)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position (AU)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z Position (AU)', fontsize=12, fontweight='bold')
        ax.set_title(f'3D Orbital Trajectories (N={self.n_bodies}, showing {num_bodies_to_plot})',
                    fontsize=14, fontweight='bold')

        # Set equal aspect ratio
        max_range = self.positions[['x', 'y', 'z']].abs().max().max() / AU
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

        # Custom legend
        if num_bodies_to_plot <= 10:
            ax.legend(loc='upper right', fontsize=10)

        # Add grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.output_dir,'3d_trajectories_enhanced.png'), dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir}3d_trajectories_enhanced.png")

        # plt.show()

    def create_animation_enhanced(self, num_bodies=20, num_frames=200, save_gif=True):
        """Create high-quality animated visualization"""
        if self.positions is None:
            print("No position data available")
            return

        print(f"Creating animation with {num_frames} frames...")

        fig = plt.figure(figsize=(16, 12))
        
        # Create subplots
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        ax3d = fig.add_subplot(gs[:, 0], projection='3d')
        ax_xy = fig.add_subplot(gs[0, 1])
        ax_dist = fig.add_subplot(gs[1, 1])

        num_bodies = min(num_bodies, self.n_bodies)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_bodies))

        # Sample frames evenly
        frame_indices = np.linspace(0, len(self.times)-1, num_frames, dtype=int)
        frame_times = self.times[frame_indices]

        # Initialize 3D plot elements
        lines_3d = []
        points_3d = []
        for i in range(num_bodies):
            line, = ax3d.plot([], [], [], color=colors[i], alpha=0.6, linewidth=1.5)
            point, = ax3d.plot([], [], [], 'o', color=colors[i], markersize=6)
            lines_3d.append(line)
            points_3d.append(point)

        # Initialize XY plot
        lines_xy = []
        points_xy = []
        for i in range(num_bodies):
            line, = ax_xy.plot([], [], color=colors[i], alpha=0.6, linewidth=1.5)
            point, = ax_xy.plot([], [], 'o', color=colors[i], markersize=6)
            lines_xy.append(line)
            points_xy.append(point)

        # Initialize distance plot
        dist_lines = []
        for i in range(1, min(10, num_bodies)):
            line, = ax_dist.plot([], [], color=colors[i], alpha=0.7, linewidth=1.5)
            dist_lines.append(line)

        # Set up plot limits
        max_range = self.positions[['x', 'y', 'z']].abs().max().max() / AU
        ax3d.set_xlim([-max_range, max_range])
        ax3d.set_ylim([-max_range, max_range])
        ax3d.set_zlim([-max_range, max_range])
        ax3d.set_xlabel('X (AU)')
        ax3d.set_ylabel('Y (AU)')
        ax3d.set_zlabel('Z (AU)')

        ax_xy.set_xlim([-max_range, max_range])
        ax_xy.set_ylim([-max_range, max_range])
        ax_xy.set_xlabel('X (AU)')
        ax_xy.set_ylabel('Y (AU)')
        ax_xy.set_title('XY Projection')
        ax_xy.grid(True, alpha=0.3)
        ax_xy.set_aspect('equal')

        ax_dist.set_xlim([0, self.times[-1]/86400])
        ax_dist.set_xlabel('Time (days)')
        ax_dist.set_ylabel('Distance (AU)')
        ax_dist.set_title('Distance from Sun')
        ax_dist.grid(True, alpha=0.3)

        title_3d = ax3d.set_title('', fontsize=14, fontweight='bold')

        def animate(frame_idx):
            frame = frame_indices[frame_idx]
            time_val = self.times[frame]
            current_data = self.positions[self.positions['time'] == time_val]

            # Update 3D plot
            for i in range(num_bodies):
                body_history = self.positions[(self.positions['body_id'] == i) &
                                             (self.positions['time'] <= time_val)]
                if len(body_history) > 0:
                    lines_3d[i].set_data(body_history['x']/AU, body_history['y']/AU)
                    lines_3d[i].set_3d_properties(body_history['z']/AU)

                    current = current_data[current_data['body_id'] == i]
                    if len(current) > 0:
                        points_3d[i].set_data([current['x'].values[0]/AU],
                                              [current['y'].values[0]/AU])
                        points_3d[i].set_3d_properties([current['z'].values[0]/AU])

            # Update XY plot
            for i in range(num_bodies):
                body_history = self.positions[(self.positions['body_id'] == i) &
                                             (self.positions['time'] <= time_val)]
                if len(body_history) > 0:
                    lines_xy[i].set_data(body_history['x']/AU, body_history['y']/AU)

                    current = current_data[current_data['body_id'] == i]
                    if len(current) > 0:
                        points_xy[i].set_data([current['x'].values[0]/AU],
                                              [current['y'].values[0]/AU])

            # Update distance plot
            for idx, i in enumerate(range(1, min(10, num_bodies))):
                body_history = self.positions[(self.positions['body_id'] == i) &
                                             (self.positions['time'] <= time_val)]
                if len(body_history) > 0:
                    distances = np.sqrt(body_history['x']**2 + body_history['y']**2 +
                                       body_history['z']**2) / AU
                    time_days = body_history['time'] / 86400.0
                    dist_lines[idx].set_data(time_days, distances)

            title_3d.set_text(f'N-Body Simulation - Day {time_val/86400:.1f}')
            
            progress = (frame_idx + 1) / num_frames * 100
            if frame_idx % max(1, num_frames // 20) == 0:
                print(f"Animation progress: {progress:.1f}%")

            return lines_3d + points_3d + lines_xy + points_xy + dist_lines + [title_3d]

        anim = FuncAnimation(fig, animate, frames=num_frames,
                           interval=50, blit=False)

        if save_gif:
            print("Saving animation (this may take a while)...")
            writer = PillowWriter(fps=20)
            anim.save(os.path.join(self.output_dir,'simulation_animation.gif'), writer=writer, dpi=100)
            print(f"✓ Saved: {self.output_dir}simulation_animation.gif")

        # plt.show()

    def generate_comprehensive_report(self):
        """Generate complete analysis with all plots"""
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE N-BODY SIMULATION REPORT")
        print("="*70 + "\n")

        # 1. Kernel comparison
        if self.benchmark is not None:
            print("1. Kernel Performance Comparison...")
            self.plot_kernel_comparison(save=True)

        # 2. Energy conservation
        if self.energy is not None:
            print("\n2. Detailed Energy Conservation Analysis...")
            self.plot_energy_conservation_detailed(save=True)

        # 3. Comet analysis
        if self.comet is not None:
            print("\n3. Comet Trajectory Analysis...")
            self.plot_comet_analysis(save=True)

        # 4. Performance metrics
        if self.performance is not None:
            print("\n4. Performance Scaling Analysis...")
            self.plot_performance_scaling(save=True)

        # 5. 3D trajectories
        if self.positions is not None:
            print("\n5. Enhanced 3D Trajectory Visualization...")
            self.plot_3d_trajectories_enhanced(save=True)

        print("\n" + "="*70)
        print("REPORT GENERATION COMPLETE")
        print("="*70)
        print("\nGenerated files:")
        print(f"  ✓ {self.output_dir}/kernel_comparison.png")
        print(f"  ✓ {self.output_dir}/energy_conservation_detailed.png")
        print(f"  ✓ {self.output_dir}/comet_analysis.png")
        print(f"  ✓ {self.output_dir}/performance_analysis.png")
        print(f"  ✓ {self.output_dir}/3d_trajectories_enhanced.png")
        print("="*70 + "\n")


def main():
    """Main execution"""
    # Determine output directory from command line
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.isdigit():
            output_dir = f"output_N{int(arg)}"
        else:
            output_dir = arg
    else:
        output_dir = 'output'

    if not os.path.exists(output_dir):
        print(f"Error: Directory '{output_dir}' does not exist!")
        print("Available directories: output_N* or output")
        return

    # Create visualizer with specified directory
    viz = EnhancedNBodyVisualizer(output_dir=output_dir)

    # Generate comprehensive report
    viz.generate_comprehensive_report()

    # Optional animation (uncomment if needed)
    viz.create_animation_enhanced(num_bodies=15, num_frames=200, save_gif=True)

    print("\n" + "="*70)
    print(f"All visualizations complete for {output_dir}!")
    print("="*70)

if __name__ == "__main__":
    main()
