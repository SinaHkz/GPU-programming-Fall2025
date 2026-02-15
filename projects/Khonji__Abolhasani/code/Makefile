# Makefile for N-Body Galaxy Simulation
# Targets:
#   all         : compile and run headless simulation (default N=16384, steps=500)
#   visual      : compile with OpenGL and run visual simulation (requires X11)
#   benchmark   : compile and run scalability benchmark
#   viz         : run Python visualizer (requires simulation.csv)
#   clean       : remove executables and output files

CXX = nvcc
CXXFLAGS = -O3 -arch=sm_86 --use_fast_math
LDFLAGS = 
TARGET = nbody_galaxy
SOURCES = nbody_galaxy_v13.cu

HEADLESS_RUN = ./$(TARGET) headless 16384 1000
VISUAL_RUN   = ./$(TARGET) visual   16384 1000
BENCH_RUN    = ./$(TARGET) benchmark

# Default target: compile and run headless, then visualize
all: $(TARGET)
	@echo "=== Running headless simulation (N=16384, steps=500) ==="
	$(HEADLESS_RUN)
	@echo "=== Generating visualization ==="
	python3 visualize_galaxy_v5.py

# Compile without OpenGL (headless/benchmark only)
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Compile with OpenGL support (for visual mode)
visual: CXXFLAGS += -DVISUAL_MODE
visual: LDFLAGS += -lGL -lglfw
visual: $(TARGET)
	@echo "=== Running visual simulation ==="
	$(VISUAL_RUN)

# Run benchmark
benchmark: $(TARGET)
	@echo "=== Running scalability benchmark ==="
	$(BENCH_RUN)

# Just run the Python visualizer (assumes simulation.csv exists)
viz:
	python3 visualize_galaxy_v5.py

# Clean up
clean:
	rm -f $(TARGET) simulation.csv energy.csv benchmark.csv *.gif *.mp4 *.png
	rm -rf __pycache__

.PHONY: all visual benchmark viz clean
