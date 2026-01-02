# Project Proposal: N-Body Simulation Using CUDA
## Charged Particle Dynamics

**Topic:** Topic 2 — N-Body Simulation  
**Authors:** Mohsen Mirzaei, Artin Zarei

---

## 1. Overview

GPU-accelerated N-body simulation of electrostatic interactions. Primary focus: **performance optimization** of O(N²) pairwise Coulomb force computation.

**Target Scale:** 5,000–10,000 particles

---

## 2. Interaction Model

Coulomb force:

$$\vec{F}_i = \sum_{j \neq i} k \cdot \frac{q_i \cdot q_j}{|\vec{r}_{ij}|^2 + \epsilon^2} \cdot \hat{r}_{ij}$$

**Particle Configuration:**
- 50% positive charges (+1e), 50% negative charges (-1e)
- Small number of heavy charges (±10e) as attraction/repulsion centers
- Mass proportional to charge magnitude

**Time Integration:** Velocity Verlet

---

## 3. CUDA Parallelization Strategy

**Kernel Design:** One thread per particle computes total electrostatic force.

**Shared Memory Tiling:**
- Divide particles into tiles (TILE_SIZE = 256)
- Each block loads tile into shared memory
- All threads compute interactions with tile, iterate through all tiles
- **Expected benefit:** Reduce global memory loads from O(N) to O(N/TILE_SIZE) per thread

**Memory Layout:** Structure of Arrays (SoA) for coalesced access
```
float* pos_x, pos_y, pos_z;
float* vel_x, vel_y, vel_z;
float* charge, mass;
```

**Configuration:**
- Block size: 256 threads
- Grid size: ceil(N / 256) blocks
- Shared memory: 4 KB per block

---

## 4. Expected Performance Bottlenecks

| Bottleneck | Mitigation |
|------------|------------|
| **Global memory bandwidth** | Shared memory tiling |
| **N² complexity** | Inherent to problem; optimize memory reuse |
| **Tile synchronization** | Minimize `__syncthreads()` calls |

**Primary Bottleneck:** Memory bandwidth. Naïve implementation reads 1.6 GB per timestep (N=10,000). Tiling reduces this by ~256×.

---

## 5. Performance Evaluation

**Metrics:**
- Kernel execution time (CUDA events)
- Interactions per second: $N^2 / \text{time}$
- Memory throughput (nvprof/Nsight Compute)
- GPU occupancy
- Frames per second

**Scaling Analysis:**
- Verify O(N²) time complexity (log-log plot)
- Compare against CPU baseline
- Measure effect of tile size on performance

**Correctness:**
- Energy conservation over time
- Opposite charges attract and cluster
- Like charges repel

---

## 6. Visualization

Real-time visualization showing:
- Color-coded particles by charge (red=positive, blue=negative)
- Particle size proportional to charge magnitude
- Display of FPS and particle count

---

## 7. Deliverables

1. CUDA implementation with shared memory optimization
2. Performance analysis report (timing, scaling, profiling)
3. Demo video showing charge clustering and separation
4. Final report with bottleneck analysis

---

## Summary

This project implements electrostatic N-body simulation optimized for GPU execution using shared memory tiling and coalesced memory access. The charged particle system demonstrates both attractive and repulsive forces, creating richer dynamics than gravitational systems. Real-time visualization will validate physical correctness and demonstrate GPU performance.

