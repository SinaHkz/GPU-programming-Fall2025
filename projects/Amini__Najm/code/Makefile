# Compilers
CXX = g++
NVCC = nvcc

# Absolute paths for profilers to survive 'sudo' path stripping
NSYS = /usr/local/cuda-13.0/bin/nsys
NCU = /usr/local/cuda-13.0/bin/ncu

# Flags
CXXFLAGS = -O3 -Wall -std=c++11 -g
NVCCFLAGS = -O3 -std=c++11 -g -lineinfo

# Executable name
TARGET = nbody_sim

# Object files
OBJS = main.o init.o physics_cpu.o physics_gpu.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(OBJS)

main.o: main.cpp types.h physics.h
	$(CXX) $(CXXFLAGS) -c main.cpp

init.o: init.cpp types.h
	$(CXX) $(CXXFLAGS) -c init.cpp

physics_cpu.o: physics_cpu.cpp types.h physics.h
	$(CXX) $(CXXFLAGS) -c physics_cpu.cpp

physics_gpu.o: physics_gpu.cu types.h
	$(NVCC) $(NVCCFLAGS) -c physics_gpu.cu

# --- PROFILING TARGETS ---

# 1. Nsight Systems - Export to CSV (Kernel execution times and API calls)
profile_nsys_csv: $(TARGET)
	@echo "Running Nsight Systems (Generating CSV)..."
	sudo $(NSYS) profile --stats=true -o gpu_sim_nsys --force-overwrite=true ./$(TARGET) 4096
	sudo $(NSYS) stats --format=csv --report=cuda_gpu_kern_sum gpu_sim_nsys.sqlite > nsys_kernels.csv
	sudo $(NSYS) stats --format=csv --report=cuda_api_sum gpu_sim_nsys.sqlite > nsys_api.csv
	@echo "✅ nsys CSVs saved to nsys_kernels.csv and nsys_api.csv"

# 2. Nsight Compute - FULL METRICS for Naive Kernel (Generating CSV)
profile_ncu_naive_csv: $(TARGET)
	@echo "Running Nsight Compute FULL for Naive Kernel..."
	sudo $(NCU) --csv --set full -k bodyForceKernelNaive -c 1 --force-overwrite -o gpu_naive_ncu ./$(TARGET) 4096 > ncu_naive_full.csv
	@echo "✅ ncu Naive full metrics saved to ncu_naive_full.csv"

# 3. Nsight Compute - FULL METRICS for Tiled Kernel (Generating CSV)
profile_ncu_tiled_csv: $(TARGET)
	@echo "Running Nsight Compute FULL for Tiled Kernel..."
	sudo $(NCU) --csv --set full -k bodyForceKernelTiled -c 1 --force-overwrite -o gpu_tiled_ncu ./$(TARGET) 4096 > ncu_tiled_full.csv
	@echo "✅ ncu Tiled full metrics saved to ncu_tiled_full.csv"

clean:
	rm -f *.o $(TARGET) *.csv *.txt *.nsys-rep *.ncu-rep *.sqlite