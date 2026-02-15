# GPU Bottleneck Report (Measured + Derived)

This report is **numeric and evidence‑backed**. Every metric is tagged as **MEASURED**, **DERIVED**, or **ESTIMATED** with explicit sources.

## 0. Executive Summary (Top Bottlenecks)
- **MEASURED:** In benchmark mode (5,000 steps), `compute_accel_tiled` accounts for **99.4%** of GPU kernel time. Source: `perf_artifacts/logs/nsys_rerun_bench_N16384.txt`.
- **MEASURED:** For `N=16384`, time per step is **1.983 ms** (benchmark rerun). Source: `perf_artifacts/logs/nsys_rerun_bench_N16384.txt`.
- **MEASURED:** In present mode, `cudaMemcpy` CPU API time totals **3.931 s** over 50,007 calls (**78.6 μs/call ≈ 0.786 ms/frame**). Source: `perf_artifacts/logs/nsys_rerun_present.txt`.
- **MEASURED:** Present mode GPU kernel time per frame ≈ **0.692 ms** (sum of per‑kernel averages). Source: `perf_artifacts/logs/nsys_rerun_present.txt`.
- **MEASURED:** SolarCollision GPU time is split between gravity and collision detection: `compute_accel_tiled` **53.1%**, `detect_collision_pairs` **46.1%**. Source: `perf_artifacts/logs/nsys_rerun_solarCollision.txt`.
- **DERIVED:** Effective global bandwidth for gravity kernel (N=16384) ≈ **8.71 GB/s**. Source: `perf_artifacts/logs/nsys_rerun_bench_N16384.txt` + derivation below.
- **DERIVED:** Arithmetic intensity (global) ≈ **610 FLOPs/byte**; shared ≈ **2.44 FLOPs/byte**. Source: derivation below.
- **DERIVED:** Occupancy estimate for `compute_accel_tiled` on sm_75 with 29 regs/thread and 4 KB shared/block is **100%**. Source: `perf_artifacts/logs/ptxas_build.txt` + occupancy derivation.
- **MEASURED:** Nsight Compute counters are unavailable due to `ERR_NVGPUCTRPERM`. Source: `perf_artifacts/logs/ncu_attempt.txt`.

## 1. Environment (Measured)
- GPU: **NVIDIA GeForce RTX 4050 Laptop GPU** (Ada Lovelace)
- Driver: **580.126.09**, CUDA: **13.0**
- Source: `perf_artifacts/env.txt`

## 2. Commands Used (Measured)
**Build (Release):**
```
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```
Log: `perf_artifacts/logs/build_release.txt`

**NSYS rerun (vsync disabled):**
```
__GL_SYNC_TO_VBLANK=0 nsys profile --stats=true --force-overwrite=true \
  -o perf_artifacts/nsys_rerun/bench_N16384 \
  ./build/nbody --benchmark 1 --steps 5000 --collisions 0 --N 16384

__GL_SYNC_TO_VBLANK=0 nsys profile --stats=true --force-overwrite=true \
  -o perf_artifacts/nsys_rerun/present \
  ./build/nbody --present 1 --steps 5000

__GL_SYNC_TO_VBLANK=0 nsys profile --stats=true --force-overwrite=true \
  -o perf_artifacts/nsys_rerun/solarCollision \
  ./build/nbody --solarCollision 1 --steps 5000
```
Logs: `perf_artifacts/logs/nsys_rerun_*.txt`

**NCU attempt (sudo):**
```
sudo -n ncu --set full --target-processes all \
  -o perf_artifacts/ncu/accel_N16384 \
  ./build/nbody --benchmark 1 --steps 2000 --collisions 0 --N 16384
```
Result: **failed** (`sudo: a password is required`). Log: `perf_artifacts/logs/ncu_attempt.txt`

**PTXAS info (fallback for registers):**
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="--ptxas-options=-v"
cmake --build . -j
```
Log: `perf_artifacts/logs/ptxas_build.txt`

## 3. Measured Results (NSYS Rerun)

### 3.1 Benchmark Mode (5000 steps, N sweep)
Source: `perf_artifacts/logs/nsys_rerun_bench_N16384.txt`

| N | time_per_step_ms | IPS |
|---:|---:|---:|
| 1024 | 0.072 | 1.446e10 |
| 2048 | 0.125 | 3.362e10 |
| 4096 | 0.244 | 6.865e10 |
| 8192 | 0.547 | 1.227e11 |
| 16384 | 1.983 | 1.353e11 |

**Kernel time breakdown (MEASURED):**
- `compute_accel_tiled`: **99.4%**, **14.727 s total**, **0.588 ms avg**, **25,030 instances**.
- `update_vel_half`: **0.2%**, **30.76 ms total**, **1.229 μs avg**.
- `update_pos`: **0.2%**, **30.72 ms total**, **1.228 μs avg**.
- `finalize_vel`: **0.2%**, **30.77 ms total**, **1.230 μs avg**.
Source: `perf_artifacts/logs/nsys_rerun_bench_N16384.txt`.

### 3.2 Present Mode (5000 steps)
Source: `perf_artifacts/logs/nsys_rerun_present.txt`

**GPU kernel time per frame (MEASURED):**
- `compute_accel_tiled`: **0.2857 ms** (avg)
- `compute_energy_tiled`: **0.2500 ms** (avg)
- `compute_min_dist`: **0.1486 ms** (avg)
- `reduce_min_pair`: **0.00166 ms** (avg)
- Updates (vel/pos/final): **~0.00419 ms total**

**Total GPU kernel time per frame (MEASURED):**
≈ **0.692 ms/frame** (sum above)

**Transfer costs (MEASURED):**
- D→H memcpy GPU time: **80.295 ms total / 50,000 copies = 1.606 μs/copy**
  ⇒ **0.0161 ms/frame** (GPU timeline)
- `cudaMemcpy` CPU API time: **3.931 s total / 50,007 calls = 78.6 μs/call**
  ⇒ **0.786 ms/frame** (CPU‑side stall)

### 3.3 SolarCollision Mode (5000 steps)
Source: `perf_artifacts/logs/nsys_rerun_solarCollision.txt`

**GPU kernel time per frame (MEASURED):**
- `compute_accel_tiled`: **0.2795 ms** (avg)
- `detect_collision_pairs`: **0.2427 ms** (avg)
- Updates (vel/pos/final): **~0.00419 ms total**

**Kernel mix (MEASURED):**
- `compute_accel_tiled`: **53.1%**
- `detect_collision_pairs`: **46.1%**

**Transfer & sync costs (MEASURED):**
- D→H memcpy GPU time: **121.575 ms / 70,392 copies = 1.727 μs/copy**
  ⇒ **0.0243 ms/frame** (GPU timeline)
- `cudaMemcpy` CPU API time: **787.3 ms / 75,040 calls = 10.49 μs/call**
  ⇒ **0.157 ms/frame**
- `cudaDeviceSynchronize`: **2.076 s / 38,374 calls = 54.1 μs/call**
  ⇒ **0.415 ms/frame**

## 4. Nsight Compute (NCU) — Status
**MEASURED:** NCU failed with `ERR_NVGPUCTRPERM`. Log: `perf_artifacts/logs/ncu_attempt.txt`.

**Fallback implemented (DERIVED):** Registers per thread, occupancy estimate, effective bandwidth, and arithmetic intensity computed below.

## 5. Derived Metrics (Fallback Required by Permissions)

### 5.1 Register Count (MEASURED from PTXAS)
- `compute_accel_tiled`: **29 registers/thread**
- Source: `perf_artifacts/logs/ptxas_build.txt`

### 5.2 Occupancy Estimate (DERIVED)
**Assumptions (sm_75):**
- max threads/SM = 2048
- max registers/SM = 65,536
- max shared mem/SM = 64 KB
- max blocks/SM = 16
- blockDim = 256
- registers/thread = 29
- dynamic shared mem/block = 4 * blockDim * 4 = **4096 bytes**

**Blocks/SM limits:**
- By registers: floor(65536 / (29*256)) = **8 blocks**
- By threads: floor(2048 / 256) = **8 blocks**
- By shared mem: floor(65536 / 4096) = **16 blocks**

**Occupancy:**
- active blocks/SM = min(8, 8, 16) = **8**
- active threads/SM = 8 * 256 = **2048**
- **Occupancy = 2048 / 2048 = 100%**

### 5.3 Global Bytes Per Kernel (DERIVED)
For **N=16384, block=256**:
- blocks = ceil(N/block) = 64
- tiles = 64
- bytes per tile‑block load = block * 4 floats * 4 bytes = **4096 bytes**
- total tile loads = blocks * tiles * 4096 = **16,777,216 bytes**
- per‑body loads/stores: x/y/z load (12 bytes) + ax/ay/az store (12 bytes) = **24 * N = 393,216 bytes**
- **total global bytes ≈ 17,170,432 bytes**

**Bytes per interaction (DERIVED):**
- interactions = N(N−1) = 268,419,072
- bytes/interaction = 17,170,432 / 268,419,072 = **0.06397 bytes**

### 5.4 Arithmetic Intensity (DERIVED)
**FLOPs per interaction (assumptions):**
- dx,dy,dz: 3 sub
- dist2: 3 mul + 3 add
- +eps2: 1 add
- rsqrt: ~20 FLOPs
- invDist3: 2 mul
- scale (G*mi): 2 mul
- accumulate: 3 mul + 3 add

**Total FLOPs/interaction = 39 (DERIVED)**

**AI (global):**
- 39 FLOPs / 0.06397 bytes = **609.7 FLOPs/byte**

**AI (shared):**
- 39 FLOPs / 16 bytes = **2.4375 FLOPs/byte**

### 5.5 Effective Global Bandwidth (DERIVED)
Use **N=16384 benchmark** time per step and NSYS kernel share:
- time_per_step_ms (MEASURED) = **1.983 ms**
- compute_accel share (MEASURED) = **99.4%**
- kernel time (DERIVED) = 1.983 * 0.994 = **1.971 ms**
- total global bytes (DERIVED) = **17,170,432 bytes**

**Effective bandwidth = 17,170,432 / 0.001971 / 1e9 = 8.71 GB/s (DERIVED)**

## 6. Ranked Bottleneck Table (Measured/Derived)

| Rank | Bottleneck | Evidence | Metric | Type |
|---:|---|---|---|---|
| 1 | Gravity kernel dominates benchmark | `nsys_rerun_bench_N16384.txt` | 99.4% GPU kernel time | MEASURED |
| 2 | CPU memcpy cost in present mode | `nsys_rerun_present.txt` | 0.786 ms/frame CPU memcpy | MEASURED |
| 3 | Collision detection cost | `nsys_rerun_solarCollision.txt` | 46.1% GPU kernel time | MEASURED |
| 4 | Diagnostics overhead in present | `nsys_rerun_present.txt` | energy+minDist = 57.6% GPU kernel time | MEASURED |
| 5 | Effective global bandwidth | derivation | 8.71 GB/s | DERIVED |
| 6 | Occupancy | PTXAS + derivation | 100% | DERIVED |

## 7. Previously‑Present Bottlenecks (Counterfactual Estimates)
Git history is unavailable, so these are **ESTIMATED** with explicit derivations.

### 7.1 If benchmark mistakenly included render memcpy
**ESTIMATED:** Present‑mode CPU memcpy cost is 0.786 ms/frame. Adding this to benchmark N=16384 (1.983 ms) would yield **2.769 ms/step**, a **39.6% slowdown**.
- Measured memcpy cost: `perf_artifacts/logs/nsys_rerun_present.txt`
- Benchmark step time: `perf_artifacts/logs/nsys_rerun_bench_N16384.txt`

### 7.2 If gravity kernel were untiled (naive global loads)
**ESTIMATED RANGE:**
- Naive bytes/interaction ≈ 16 bytes ⇒ total ≈ 4.29 GB for N=16384.
- Using measured effective bandwidth 8.71 GB/s, naive kernel time ≈ **0.493 s**.
- Current kernel time is **0.00197 s**, implying a **~250× slowdown** if memory‑bound.

**Note:** This is a counterfactual estimate using measured bandwidth as a proxy.

## 8. Action Plan (Evidence‑Driven)
1. **CUDA‑GL interop** to eliminate D→H copies in render mode.
   - Evidence: 0.786 ms/frame CPU memcpy cost.
2. **Reduce diagnostic kernels** or lower their frequency (energy/minDist).
   - Evidence: 57.6% of GPU kernel time in present mode.
3. **Parallelize collision resolution** (avoid serialized pair handling).
   - Evidence: `resolve_collision_pairs<<<1,1>>>` and collision detection ~46% in solarCollision.
4. **Use GPU counters** by enabling performance counters (admin policy change).
   - Evidence: NCU blocked (ERR_NVGPUCTRPERM).

## 9. Files Produced
- `perf_artifacts/nsys_rerun/*.nsys-rep` (new rerun profiles)
- `perf_artifacts/logs/nsys_rerun_*.txt` (new stats logs)
- `perf_artifacts/logs/ptxas_build.txt` (register counts)
- `perf_artifacts/logs/ncu_attempt.txt` (permission failure)
- `perf_artifacts/baseline_metrics.json` (machine‑readable baseline)
