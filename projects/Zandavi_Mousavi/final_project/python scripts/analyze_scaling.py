"""
Scaling Analysis Script for N-Body Simulation
Compares performance across different N values and kernel types
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def analyze_scaling():
    """Analyze performance scaling across multiple N values"""
    
    # Find all output directories
    output_dirs = sorted(glob.glob('output_N*'))
    
    if not output_dirs:
        print("No scaling data found. Run: make scaling")
        return
    
    print(f"Found {len(output_dirs)} datasets")
    
    # Collect data
    results = []
    
    for output_dir in output_dirs:
        # Extract N from directory name
        N = int(output_dir.split('_N')[-1])
        
        # Load benchmark data
        benchmark_file = os.path.join(output_dir, 'kernel_benchmark.csv')
        if os.path.exists(benchmark_file):
            df = pd.read_csv(benchmark_file)
            
            for _, row in df.iterrows():
                results.append({
                    'N': N,
                    'kernel': row['kernel'],
                    'time_ms': row['avg_time_ms'],
                    'interactions_per_sec': row['interactions_per_sec'],
                    'gflops': row['gflops'],
                    # 'occupancy': row['occupancy_percent']
                })
    
    if not results:
        print("No benchmark data found in output directories")
        return
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    kernels = df_results['kernel'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(kernels)))
    
    # 1. Execution Time vs N
    ax = axes[0, 0]
    for i, kernel in enumerate(kernels):
        data = df_results[df_results['kernel'] == kernel]
        ax.loglog(data['N'], data['time_ms'], 'o-', label=kernel, 
                 color=colors[i], linewidth=2, markersize=8)
    
    # Add O(N²) reference line
    N_ref = df_results['N'].unique()
    time_ref = df_results[df_results['kernel'] == kernels[0]]['time_ms'].iloc[0]
    N_ref_0 = df_results['N'].unique()[0]
    ax.loglog(N_ref, time_ref * (N_ref / N_ref_0)**2, 'k--', 
             alpha=0.5, linewidth=2, label='O(N²) reference')
    
    ax.set_xlabel('Number of Bodies (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Scaling: Execution Time vs N', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # 2. GFLOPS vs N
    ax = axes[0, 1]
    for i, kernel in enumerate(kernels):
        data = df_results[df_results['kernel'] == kernel]
        ax.semilogx(data['N'], data['gflops'], 'o-', label=kernel,
                   color=colors[i], linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Bodies (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('GFLOPS', fontsize=12, fontweight='bold')
    ax.set_title('Computational Throughput vs N', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Speedup vs Naive
    ax = axes[0, 2]
    baseline_kernel = kernels[0]  # Assume first is naive
    
    for i, kernel in enumerate(kernels[1:], 1):  # Skip baseline
        speedups = []
        N_values = []
        
        for N in df_results['N'].unique():
            baseline_time = df_results[(df_results['kernel'] == baseline_kernel) & 
                                      (df_results['N'] == N)]['time_ms'].values
            kernel_time = df_results[(df_results['kernel'] == kernel) & 
                                    (df_results['N'] == N)]['time_ms'].values
            
            if len(baseline_time) > 0 and len(kernel_time) > 0:
                speedup = baseline_time[0] / kernel_time[0]
                speedups.append(speedup)
                N_values.append(N)
        
        ax.semilogx(N_values, speedups, 'o-', label=kernel,
                   color=colors[i], linewidth=2, markersize=8)
    
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel('Number of Bodies (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup vs Naive', fontsize=12, fontweight='bold')
    ax.set_title('Optimization Speedup', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Efficiency (% of peak GFLOPS)
    ax = axes[1, 0]
    # Assume RTX 4070 peak: ~29 TFLOPS FP32
    peak_gflops = 29000  # Adjust for your GPU
    
    for i, kernel in enumerate(kernels):
        data = df_results[df_results['kernel'] == kernel]
        efficiency = (data['gflops'] / peak_gflops) * 100
        ax.semilogx(data['N'], efficiency, 'o-', label=kernel,
                   color=colors[i], linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Bodies (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Efficiency (% of Peak)', fontsize=12, fontweight='bold')
    ax.set_title('GPU Utilization Efficiency', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(10, (df_results['gflops'].max() / peak_gflops * 100) * 1.2)])
    
    # 5. Interactions per second
    ax = axes[1, 1]
    for i, kernel in enumerate(kernels):
        data = df_results[df_results['kernel'] == kernel]
        ax.loglog(data['N'], data['interactions_per_sec'], 'o-', label=kernel,
                 color=colors[i], linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Bodies (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Interactions/second', fontsize=12, fontweight='bold')
    ax.set_title('Interaction Throughput', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # 6. Summary table
    ax = axes[1, 2]
    ax.axis('off')
    
    # Create summary statistics
    summary_text = "Performance Summary\n" + "="*40 + "\n\n"
    
    for kernel in kernels:
        kernel_data = df_results[df_results['kernel'] == kernel]
        avg_gflops = kernel_data['gflops'].mean()
        max_gflops = kernel_data['gflops'].max()
        
        # Calculate speedup vs naive at largest N
        largest_N = df_results['N'].max()
        baseline_time = df_results[(df_results['kernel'] == baseline_kernel) & 
                                  (df_results['N'] == largest_N)]['time_ms'].values
        kernel_time = df_results[(df_results['kernel'] == kernel) & 
                                (df_results['N'] == largest_N)]['time_ms'].values
        
        if len(baseline_time) > 0 and len(kernel_time) > 0:
            speedup = baseline_time[0] / kernel_time[0]
        else:
            speedup = 1.0
        
        summary_text += f"{kernel}:\n"
        summary_text += f"  Avg GFLOPS: {avg_gflops:.1f}\n"
        summary_text += f"  Max GFLOPS: {max_gflops:.1f}\n"
        summary_text += f"  Speedup @ N={largest_N}: {speedup:.2f}x\n\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=11, family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('N-Body Simulation: Scaling Analysis', 
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('scaling_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: scaling_analysis.png")
    
    plt.show()
    
    # Print detailed results
    print("\n" + "="*70)
    print("DETAILED SCALING RESULTS")
    print("="*70)
    print(df_results.to_string(index=False))
    print("="*70)
    
    # Save to CSV
    df_results.to_csv('scaling_results.csv', index=False)
    print("\n✓ Saved: scaling_results.csv")


if __name__ == "__main__":
    analyze_scaling()
