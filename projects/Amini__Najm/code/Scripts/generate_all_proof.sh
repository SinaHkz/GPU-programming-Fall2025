#!/bin/bash

RESULTS_DIR="Final_Project_Results"

echo "=========================================="
echo " Starting the Ultimate Profiling Suite"
echo "=========================================="

# Create the results folder
mkdir -p $RESULTS_DIR

# 1. Standard Build (Optimized, No Gprof Overhead)
echo "[1/6] Compiling standard optimized build..."
make clean
make

# 2. Execution Times (The Main Benchmark)
echo "[2/6] Running standard Execution Time benchmarks..."
./nbody_sim 4096 > $RESULTS_DIR/Execution_Times.txt
echo "      -> Saved to Execution_Times.txt"

# 3. CPU Profiling: Perf (Hardware Counters)
echo "[3/6] Profiling CPU with 'perf'..."
sudo perf record -g ./nbody_sim 4096
sudo perf report --stdio > $RESULTS_DIR/cpu_perf_report.txt
echo "      -> Saved to cpu_perf_report.txt"

# 4. GPU Profiling: Nsight Systems (API & Kernel Timings)
echo "[4/6] Profiling GPU with Nsight Systems..."
# Save the text report and the visual files
sudo /usr/local/cuda-13.0/bin/nsys profile --stats=true --force-overwrite=true -o gpu_sim_nsys ./nbody_sim 4096 > $RESULTS_DIR/gpu_nsys_report.txt
# Extract the CSVs
sudo /usr/local/cuda-13.0/bin/nsys stats --format=csv --report=cuda_gpu_kern_sum gpu_sim_nsys.sqlite > $RESULTS_DIR/nsys_kernels.csv
sudo /usr/local/cuda-13.0/bin/nsys stats --format=csv --report=cuda_api_sum gpu_sim_nsys.sqlite > $RESULTS_DIR/nsys_api.csv
echo "      -> Saved NSYS Text and CSV reports"

# 5. GPU Profiling: Nsight Compute (Hardware Metrics for Naive & Tiled)
echo "[5/6] Profiling GPU Hardware with Nsight Compute..."

# -- NAIVE KERNEL --
# Save the text report and visual file
sudo /usr/local/cuda-13.0/bin/ncu --set default -k bodyForceKernelNaive -c 1 --force-overwrite -o gpu_naive_ncu ./nbody_sim 4096 > $RESULTS_DIR/gpu_ncu_naive_report.txt
# Save the FULL CSV metrics
sudo /usr/local/cuda-13.0/bin/ncu --csv --set full -k bodyForceKernelNaive -c 1 ./nbody_sim 4096 > $RESULTS_DIR/ncu_naive_full.csv

# -- TILED KERNEL --
# Save the text report and visual file
sudo /usr/local/cuda-13.0/bin/ncu --set default -k bodyForceKernelTiled -c 1 --force-overwrite -o gpu_tiled_ncu ./nbody_sim 4096 > $RESULTS_DIR/gpu_ncu_tiled_report.txt
# Save the FULL CSV metrics
sudo /usr/local/cuda-13.0/bin/ncu --csv --set full -k bodyForceKernelTiled -c 1 ./nbody_sim 4096 > $RESULTS_DIR/ncu_tiled_full.csv
echo "      -> Saved NCU Text reports, Visual files, and FULL CSVs"

# 6. CPU Profiling: Gprof (Requires Recompilation)
echo "[6/6] Recompiling code with -pg flag for 'gprof'..."
make clean
# We override the Makefile flags to inject the -pg profiling hook
make CXXFLAGS="-O3 -Wall -std=c++11 -g -pg" NVCCFLAGS="-O3 -std=c++11 -g -lineinfo -Xcompiler -pg"
./nbody_sim 4096
gprof ./nbody_sim gmon.out > $RESULTS_DIR/cpu_gprof_report.txt
echo "      -> Saved to cpu_gprof_report.txt"

# Clean up messy visual files from the root directory into the results folder
echo "Cleaning up temporary files..."
mv *.sqlite *.nsys-rep *.ncu-rep perf.data perf.data.old gmon.out $RESULTS_DIR/ 2>/dev/null
make clean

echo "=========================================="
echo " DONE! All your project proof is waiting in:"
echo " ./${RESULTS_DIR}/"
echo "=========================================="