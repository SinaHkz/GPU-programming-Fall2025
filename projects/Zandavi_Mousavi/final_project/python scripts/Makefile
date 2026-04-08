# Makefile for GPU-Accelerated N-Body Simulation
# Target: NVIDIA RTX 4070 Mobile (Compute Capability 8.9)

# Compiler
NVCC = nvcc

# Compute capability (adjust for your GPU)
ARCH ?= -arch=sm_89

# Compiler flags
NVCCFLAGS = -O3 -Xcompiler -Wall

# Target executable
TARGET = nbody

# Source file
SRC = nbody_optimized.cu

# List of N values for scaling
N_VALUES = 1024 2048 4096 8192 16384 32768 65536 
# 16384 32768 65536
# Default rule
all: $(TARGET)

# Compile the CUDA program
$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)
	@echo "Compilation successful! Run with: ./$(TARGET)"

# Clean build artifacts
clean:
	rm -f $(TARGET)
	rm -rf output output_N*
	@echo "Cleaned build artifacts and output directories"

# Run targets
run: $(TARGET)
	./$(TARGET)

run-1k: $(TARGET)   ./$(TARGET) 1024
run-2k: $(TARGET)   ./$(TARGET) 2048
run-4k: $(TARGET)   ./$(TARGET) 4096
run-8k: $(TARGET)   ./$(TARGET) 8192
run-16k: $(TARGET)  ./$(TARGET) 16384
run-32k: $(TARGET)  ./$(TARGET) 32768
run-64k: $(TARGET)  ./$(TARGET) 65536

# Visualize targets
visualize:
	python3 visualize_nbody_enhanced.py

visualize-1k:  python3 visualize_nbody_enhanced.py 1024
visualize-2k:  python3 visualize_nbody_enhanced.py 2048
visualize-4k:  python3 visualize_nbody_enhanced.py 4096
visualize-8k:  python3 visualize_nbody_enhanced.py 8192
visualize-16k: python3 visualize_nbody_enhanced.py 16384
visualize-32k: python3 visualize_nbody_enhanced.py 32768
visualize-64k: python3 visualize_nbody_enhanced.py 65536

visualize-N:
	@if [ -z "$(N)" ]; then \
		echo "Error: Please specify N, e.g., make visualize-N N=8192"; \
		exit 1; \
	fi
	python3 visualize_nbody_enhanced.py $(N)

# Full workflow for specific N
full-1k:  all run-1k  visualize-1k
full-2k:  all run-2k  visualize-2k
full-4k:  all run-4k  visualize-4k
full-8k:  all run-8k  visualize-8k
full-16k: all run-16k visualize-16k
full-32k: all run-32k visualize-32k
full-64k: all run-64k visualize-64k

# Scaling: run multiple N values
scaling: $(TARGET)
	@echo "Running scaling analysis..."
	@for n in $(N_VALUES); do \
		echo "=== Running N=$$n ==="; \
		./$(TARGET) $$n || exit 1; \
		if [ -d "output_N$$n" ]; then rm -rf "output_N$$n"; fi; \
		mv output "output_N$$n"; \
	done
	@echo "Scaling complete! Results in output_N* directories"

# NEW: Complete scaling + individual visualization + final scaling plot
scaling-visualize: scaling
	@echo "Generating individual visualizations for each N..."
	@for n in $(N_VALUES); do \
		echo "=== Visualizing N=$$n ==="; \
		python3 visualize_nbody_enhanced.py $$n || exit 1; \
	done
	@echo "Generating overall scaling analysis..."
	python3 analyze_scaling.py
	@echo "======================================================================"
	@echo "ALL DONE!"
	@echo "• Individual reports: in each output_N*/"
	@echo "• Scaling plots: scaling_analysis.png + scaling_results.csv"
	@echo "======================================================================"

# Help
help:
	@echo "Main targets:"
	@echo "  make scaling              - Run all N values"
	@echo "  make scaling-visualize    - Run all N + visualize each + final scaling plot (RECOMMENDED)"
	@echo "  make visualize-Xk         - Visualize specific N"
	@echo "  make full-Xk              - Compile + run + visualize specific N"
	@echo "  make clean                - Remove everything"

.PHONY: all clean run run-1k run-2k run-4k run-8k run-16k run-32k run-64k \
        visualize visualize-1k visualize-2k visualize-4k visualize-8k visualize-16k visualize-32k visualize-64k visualize-N \
        full-1k full-2k full-4k full-8k full-16k full-32k full-64k \
        scaling scaling-visualize help