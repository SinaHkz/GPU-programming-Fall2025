import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# Set a professional style for the plots
plt.style.use('ggplot')

print("Generating Performance Plots...")

# ==========================================
# Plot 1: Overall Execution Times
# ==========================================
labels = []
times = []

# Parse the Execution_Times.txt file
with open("Execution_Times.txt", "r") as f:
    for line in f:
        if "CPU Time:" in line:
            labels.append("CPU")
            times.append(float(re.findall(r"[\d.]+", line)[1]))
        elif "GPU Naive Time:" in line:
            labels.append("GPU Naive")
            times.append(float(re.findall(r"[\d.]+", line)[1]))
        elif "GPU Tiled Time:" in line:
            labels.append("GPU Tiled")
            times.append(float(re.findall(r"[\d.]+", line)[1]))
        elif "GPU Streamed Time:" in line:
            labels.append("GPU Streamed")
            times.append(float(re.findall(r"[\d.]+", line)[1]))

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, times, color=['#E24A33', '#348ABD', '#988ED5', '#777777'])
ax.set_ylabel('Execution Time (Seconds)')
ax.set_title('Overall Simulation Time (Lower is Better)')
ax.set_yscale('log') # Log scale because the CPU is so much slower!

# Add the exact time text on top of each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}s', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("Plot_1_Execution_Times.png", dpi=300)
print("✅ Saved Plot_1_Execution_Times.png")

# ==========================================
# Plot 2: GPU Kernel Execution Time (Pure Math)
# ==========================================
# FIX: Added skiprows=1 to ignore the NVIDIA "Processing..." message
df_kernels = pd.read_csv("nsys_kernels.csv", skiprows=1)

# Filter just the bodyForce kernels
df_forces = df_kernels[df_kernels['Name'].str.contains("bodyForce")]
kernel_names = ["Naive (Global Mem)", "Tiled (Shared Mem)"]
kernel_times = df_forces['Avg (ns)'].values / 1000.0 # Convert to microseconds

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(kernel_names, kernel_times, color=['#E24A33', '#8EBA42'])
ax.set_ylabel('Average Kernel Time (Microseconds)')
ax.set_title('Kernel Compute Time: Naive vs Tiled')

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f} µs', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("Plot_2_Kernel_Times.png", dpi=300)
print("✅ Saved Plot_2_Kernel_Times.png")

# ==========================================
# Plot 3: CUDA API Overhead (Pie Chart)
# ==========================================
# FIX: Added skiprows=1
df_api = pd.read_csv("nsys_api.csv", skiprows=1)

# Get the top 4 API calls by time percentage
top_api = df_api.head(4)
other_api_time = df_api['Time (%)'][4:].sum()

labels = list(top_api['Name']) + ['Other']
sizes = list(top_api['Time (%)']) + [other_api_time]

fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E'])
ax.set_title('CUDA API Time Distribution (Overhead)')

plt.tight_layout()
plt.savefig("Plot_3_API_Overhead.png", dpi=300)
print("✅ Saved Plot_3_API_Overhead.png")

print("All charts generated successfully!")