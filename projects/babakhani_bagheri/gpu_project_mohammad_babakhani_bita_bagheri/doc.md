# CUDA N-Body Project Report (Draft - Step 1)

## Scope of This Draft
This draft documents the implemented scientific model and equations used in the project.
The **CPU Methodology** and **GPU Methodology** sections are intentionally left empty for step-by-step completion.

## 1. Implemented System Overview

The project implements a 3D N-body simulation with:
- Newtonian gravity (all-to-all interactions),
- softening for near-field stability,
- Leapfrog / Velocity-Verlet time integration,
- optional collision handling (elastic or merge model),
- optional fragmentation/debris generation,
- optional visual scientific overlays (energy class, close encounters, velocity vectors, COM diagnostics).

State variables per body:
- Position: `(x_i, y_i, z_i)`
- Velocity: `(v_{x,i}, v_{y,i}, v_{z,i})`
- Mass: `m_i`
- Collision radius: `r_i` (used only in collision-enabled modes)

The gravitational solver is implemented in Structure-of-Arrays layout and computes pairwise interactions for each body.

### 1.1 Execution Path Clarification

The implemented execution path is mode-dependent:
- **Pure gravity / proposal-compliant mode (`collisions=0`)**: force computation and Leapfrog/Velocity-Verlet updates are GPU-resident for each step.
- **Elastic collision path**: collision pair detection is executed on device; collision response is applied through a dedicated collision-resolution kernel.
- **Merge/fragmentation path**: collision pairs may be copied to host and resolved/updated in host logic, followed by Host→Device state upload. This introduces Device→Host→Device round-trips in collision-enabled merge/fragmentation workflows.
- In all modes, the $O(N^2)$ gravitational interaction computation and the Leapfrog integration stages remain GPU-resident; host interaction is introduced only by optional collision-resolution/fragmentation logic in merge/fragmentation workflows.

Therefore, the statement "all physics is GPU-only" is valid only for collision-disabled gravity mode, not for all optional collision/fragmentation modes.

### 1.2 Physical vs Visual Radius

Two distinct radius concepts are implemented:
- **Physical collision radius (`rad[i]`)**: used by collision detection and collision response equations.
- **Visual radius (`vrad[i]` with render scaling)**: used for on-screen size construction through visual scale factors, world scale, pixel scaling, and minimum/maximum pixel clamps.

These quantities are not identical in general. A body's rendered size is a visualization mapping and should not be interpreted as its physical collision cross-section.

## 2. Core Physical Model and Equations

### 2.1 Newtonian Gravity with Softening

For body `i`, acceleration is computed from all `j != i`:

$$ \mathbf{a}_i = \sum_{j \ne i} G m_j \frac{\mathbf{r}_j-\mathbf{r}_i}{\left(\lVert \mathbf{r}_j-\mathbf{r}_i \rVert^2 + \epsilon^2\right)^{3/2}} $$

where:
- `G` is the gravitational constant,
- `\epsilon` is the softening parameter,
- `\mathbf{r}_i = (x_i,y_i,z_i)`.

Softening avoids singular acceleration at very small distance and improves numerical stability.

### 2.2 Pairwise Interaction Complexity

At each force evaluation:
- each body accumulates interactions from all other bodies,
- computational complexity is $O(N^2)$.

### 2.3 Time Integration (Leapfrog / Velocity-Verlet)

The implemented step order is:

1. Half-kick:
$$ \mathbf{v}\left(t+\frac{\Delta t}{2}\right) = \mathbf{v}(t)+\frac{\Delta t}{2}\mathbf{a}(t) $$

2. Drift:
$$ \mathbf{r}(t+\Delta t) = \mathbf{r}(t)+\Delta t\,\mathbf{v}\left(t+\frac{\Delta t}{2}\right) $$

3. Recompute acceleration at new positions:
$$ \mathbf{a}(t+\Delta t)=\mathbf{a}\big(\mathbf{r}(t+\Delta t)\big) $$

4. Final half-kick:
$$ \mathbf{v}(t+\Delta t) = \mathbf{v}\left(t+\frac{\Delta t}{2}\right) +\frac{\Delta t}{2}\mathbf{a}(t+\Delta t) $$

This is the exact update sequence used in simulation stepping.

## 3. Collision and Fragmentation Models

### 3.1 Collision Detection Condition

When collisions are enabled, bodies are treated as spheres.
Pair `(i,j)` is a collision if:

$$ \|\mathbf{r}_j-\mathbf{r}_i\| < r_i + r_j $$

### 3.2 Elastic Collision Resolution (Impulse Model)

Let:
- collision normal $\mathbf{n} = (\mathbf{r}_j-\mathbf{r}_i)/\|\mathbf{r}_j-\mathbf{r}_i\|$,
- relative velocity $\mathbf{v}_{rel} = \mathbf{v}_i-\mathbf{v}_j$,
- normal relative speed $v_n = \mathbf{v}_{rel}\cdot \mathbf{n}$.

If $v_n < 0$ (approaching), impulse magnitude:

$$ J = -\frac{(1+e)v_n}{1/m_i + 1/m_j} $$

where `e` is restitution.

Velocity update:

$$ \mathbf{v}_i' = \mathbf{v}_i + \frac{J}{m_i}\mathbf{n}, \quad \mathbf{v}_j' = \mathbf{v}_j - \frac{J}{m_j}\mathbf{n} $$

### 3.3 Positional Penetration Correction

For overlap depth:
$$ \delta = (r_i + r_j) - \|\mathbf{r}_j-\mathbf{r}_i\| $$
each body is shifted by $\delta/2$ along `±n` to remove interpenetration.

### 3.4 Merge Collision Model (Mass and Momentum Conserving)

In merge mode, colliding bodies can merge into COM state:

$$ M = m_i + m_j $$
$$ \mathbf{r}_{com} = \frac{m_i\mathbf{r}_i + m_j\mathbf{r}_j}{M} $$
$$ \mathbf{v}_{com} = \frac{m_i\mathbf{v}_i + m_j\mathbf{v}_j}{M} $$

The surviving body's mass and state become `(M, r_com, v_com)`.

Merged radius is volume-based:
$$ r_{new} = \sqrt[3]{r_i^3 + r_j^3} $$

### 3.5 Fragmentation Criterion

Fragmentation is optional and threshold-based.
Relative collision energy used in the implementation:

$$ E_{coll} = \frac{1}{2}\mu\|\mathbf{v}_i-\mathbf{v}_j\|^2, \quad \mu = \frac{m_i m_j}{m_i + m_j} $$

Fragmentation is triggered when:
$$ E_{coll} > E_{threshold} $$
and capacity constraints are satisfied (available inactive slots, debris caps).

### 3.6 Debris Generation Rules

For `k` fragments:
- each fragment mass:
$$ m_f = \frac{m_i + m_j}{k} $$
- positions initialized near COM with jitter scaled by `eps`,
- velocities initialized around COM velocity plus random spread.

Random jitter is mean-centered so generated perturbations are balanced around zero.

### 3.7 Alternate Fragment Spawning Path (Elastic-Collision Runtime Path)

An additional runtime path exists where fragmentation can be applied after elastic-pair detection.
In that path:
- candidate pairs come from detected collision pairs,
- COM velocity is used as the center velocity,
- spread is controlled by `fragmentSpeedSpread` and capped by `fragmentSpeedClamp`,
- source bodies are deactivated (`m=0`) and fragments are activated in free slots.

This path is physically useful for visual/event demonstrations but is less strictly tied to a single conservative collision law.

### 3.8 Fragmentation Conservation Statement

Fragmentation conservation properties in this implementation are:
- **Mass**: conserved exactly by construction during a split event, because fragment masses are explicitly partitioned from the colliding-body total mass, provided no truncation/slot-cap fallback path is triggered.
- **Linear momentum**: conserved by COM-centered velocity construction with mean-centered perturbations.
- **Practical deviations**: small momentum discrepancies can still appear from floating-point arithmetic and optional velocity clamping/spread controls.

Accordingly, the fragmentation model is an explicitly event-driven, non-Hamiltonian process. In practical runs it is intentionally non-conservative and effectively dissipative in aggregate behavior.

### 3.9 Conservation Properties by Mode

| Mode | Mass | Linear Momentum | Energy | Hamiltonian |
|---|---|---|---|---|
| Pure gravity (collisions OFF) | Conserved | Conserved up to numerical integration error | Not exactly conserved per step; energy oscillates with bounded error amplitude characteristic of symplectic integration rather than monotonic secular drift | Yes (time-independent softened-potential N-body Hamiltonian) |
| Elastic collision mode | Conserved | Conserved per impulse update up to floating-point error | Conserved only in the ideal elastic case (`e=1`) when neglecting floating-point rounding and positional penetration-correction adjustments; otherwise not conserved | No |
| Merge mode | Conserved (merge bookkeeping) | Conserved by COM merge update up to floating-point error | Not conserved (inelastic merge by design) | No |
| Fragmentation mode | Conserved exactly per split bookkeeping | Conserved by COM-centered assignment with possible small deviations (precision/clamping) | Not conserved (event-driven non-conservative model) | No |

## 4. Additional Scientific Quantities Implemented

### 4.1 Center of Mass (COM)

$$ \mathbf{R}_{com} = \frac{1}{M}\sum_i m_i \mathbf{r}_i, \quad \mathbf{V}_{com} = \frac{1}{M}\sum_i m_i \mathbf{v}_i $$

COM velocity can be removed at initialization (`zeroMomentum`) to reduce drift.

### 4.2 Total Energy Diagnostics

Kinetic energy:
$$ K = \sum_i \frac{1}{2}m_i\|\mathbf{v}_i\|^2 $$

Pairwise potential with softening:
$$ U = -\frac{1}{2}\sum_{i \ne j}\frac{G m_i m_j}{\sqrt{\|\mathbf{r}_j-\mathbf{r}_i\|^2+\epsilon^2}} $$

Total:
$$ E = K + U $$

The code reports periodic energy and relative drift.

Note:
- In pure gravity mode (no collisions) with fixed softening, the dynamics correspond to a time-independent softened N-body Hamiltonian, and the Leapfrog scheme is symplectic for this Hamiltonian.
- In this symplectic setting, energy is not conserved exactly at each step; instead, it typically exhibits bounded oscillatory error amplitude rather than monotonic secular drift that is common in non-symplectic schemes.
- Softening modifies the pair potential but does not break Hamiltonian structure in gravity-only mode.
- In collision/fragmentation modes, impulses, merges, and spawning make the system explicitly non-conservative.

### 4.3 Angular Momentum Diagnostics

$$ \mathbf{L} = \sum_i \mathbf{r}_i \times (m_i\mathbf{v}_i) $$

The code can print periodic `Lx, Ly, Lz`.

### 4.4 Close-Encounter Rule (Visualization Diagnostic)

A pair is highlighted as close interaction when:
$$ \|\mathbf{r}_j-\mathbf{r}_i\| < 3\epsilon $$

This is a diagnostic/visual threshold and not a force-law modification.

## 5. Current Data Snapshot (Available So Far)

Current extracted metric table:

| N | branch_efficiency_pct | dram_throughput_pct | sm_throughput_pct | time_us |
|---:|---:|---:|---:|---:|
| 512 | 95.08 | 0.21 | 4.21 | 44.26 |
| 1024 | 97.54 | 0.14 | 8.53 | 86.24 |
| 2048 | 98.77 | 0.12 | 17.37 | 171.52 |
| 4096 | 99.38 | 0.11 | 34.91 | 341.7 |
| 8192 | 99.69 | 0.10 | 61.62 | 771.74 |
| 16384 | 99.85 | 0.05 | 69.10 | 2750 |
![alt text](plots/branch_efficiency_vs_N.png) ![alt text](plots/correlation_matrix.png) ![alt text](plots/metrics_bar_N512.png) ![alt text](plots/metrics_bar_N1024.png) ![alt text](plots/metrics_bar_N2048.png) ![alt text](plots/metrics_bar_N4096.png) ![alt text](plots/metrics_bar_N8192.png) ![alt text](plots/metrics_bar_N16384.png) ![alt text](plots/sm_dram_throughput_vs_N_symlog.png) ![alt text](plots/sm_dram_throughput_vs_N.png) ![alt text](plots/time_per_interaction_vs_N.png) ![alt text](plots/time_vs_dram_scatter.png) ![alt text](plots/time_vs_N.png) ![alt text](plots/time_vs_sm_scatter.png)
These values are treated as preliminary Nsight Compute extracts from the currently available profiling artifacts. Final kernel filtering, metric-to-kernel mapping, and reproducibility criteria are deferred to the Methodology section.

Potential causes of suspiciously uniform or abrupt throughput patterns include: kernel aggregation ambiguity across multiple launches, incomplete kernel filtering, metric normalization behavior in Nsight Compute, occupancy/regime transitions at larger $N$, short profiling windows, and scenario-dependent kernel execution differences between runs.

Important note for later sections:
- this table should be interpreted carefully until final methodology and metric extraction protocol are fully documented.
- **IMPLEMENTED:** The current summary set spans $N=512$ through $N=16384$.

## 6. Methodology

### 6.1 CPU Methodology

#### 6.1.1 Scope and Baseline Definition

- **DERIVED:** CPU baseline is defined as a compute-only gravity reference configuration with `collisions=0`, `fragmentation=0`, and rendering/presentation disabled.
- **DERIVED:** CPU and GPU baseline comparisons must use matched `dt`, `eps`, initialization scenario, and random seed.
- **DERIVED:** CPU baseline is intended for gravity-kernel benchmarking only; it is not a proxy for presentation-heavy FPS behavior or collision/fragment end-to-end workflows unless those modes are separately profiled.
- **CODE-VERIFY-NEEDED:** A dedicated CPU gravity execution path is not identified in the inspected source set (`app/main.cpp`, `app/bench.cpp`, `app/sim.cpp`, `src/*`). If a CPU solver exists in another module, cite that path before claiming implemented CPU benchmark results.

#### 6.1.2 CPU Implementation Details

- **DERIVED:** Target CPU reference algorithm is all-pairs gravity accumulation ($O(N^2)$) with the same Leapfrog/Velocity-Verlet update order documented in Section 2.3.
- **DERIVED:** For fairness with GPU SoA kernels, CPU baseline should use SoA-style arrays (`x`,`y`,`z`,`vx`,`vy`,`vz`,`m`) where possible.
- **CODE-VERIFY-NEEDED:** If a CPU implementation exists, verify data layout directly in its source file and record whether it is SoA or AoS.
- **CODE-VERIFY-NEEDED:** Numeric type (`float` vs `double`) for a CPU baseline must be verified in implementation source and documented explicitly.
- **IMPLEMENTED:** No OpenMP pragmas are present in currently inspected compute files (`app/*`, `src/*`).
- **CODE-VERIFY-NEEDED:** If CPU parallelization exists in non-inspected files or build-time options, cite those paths and flags before claiming multithreaded CPU baseline behavior.

#### 6.1.3 Build and Compilation Reproducibility (CPU Baseline)

Environment record (verbatim):
- CPU: GenuineIntel, 13th Gen Intel(R) Core(TM) i7-13620H, 10 cores, cache 24576 KB
- GPU: NVIDIA GeForce RTX 4050
- CUDA Toolkit Version: CUDA 13.0
- OS: Ubuntu 22.04
- Compiler: nvcc (record nvcc --version in artifacts)

Build-policy requirements:
- **MEASUREMENT-NEEDED:** Record exact compiler/version, optimization flags, and build type for CPU baseline artifacts.
- **DERIVED:** Recommended CPU baseline policy is to document optimization flags (for example `-O3`, architecture flags) and keep them fixed across runs.
- **DERIVED:** Any use of `fast-math` must be explicitly disclosed because it can alter numerical trajectories.
- **CODE-VERIFY-NEEDED:** Actual CPU-compile flags for a CPU baseline cannot be confirmed from current source because a dedicated CPU baseline target is not identified.

#### 6.1.4 Timing Protocol (CPU)

- **DERIVED:** CPU timing should measure compute-only step time and exclude rendering, file I/O, allocations, initialization, and diagnostic printing.
- **DERIVED:** Use warm-up iterations that are not included in reported timing statistics.
- **DERIVED:** Report repeated measurements with a fixed protocol (for example median plus dispersion metric) for reproducibility.
- **DERIVED:** Preferred timer for CPU baseline is `std::chrono::steady_clock`.
- **MEASUREMENT-NEEDED:** CPU frequency/power policy (governor/turbo behavior) should be recorded because clock variability can affect wall-time comparability.
- **CODE-VERIFY-NEEDED:** If an existing CPU timing harness already exists, cite its implementation and replace this protocol-only text with implementation-grounded details.

#### 6.1.5 Correctness and Validation Protocol

- **DERIVED:** Validate CPU baseline with trajectory diagnostics already used in the report: total energy trend, center-of-mass drift, and angular momentum trend.
- **DERIVED:** CPU-vs-GPU validation should use tolerance-based comparison after fixed step counts, not bitwise equality.
- **DERIVED:** Small divergence is expected under floating-point arithmetic and differing operation ordering.
- **MEASUREMENT-NEEDED:** Publish validation tolerances and comparison checkpoints alongside benchmark artifacts for auditability.

### 6.2 GPU Methodology

#### 6.2.1 Scope and Modes

Runtime commands used in this report:
- `./build/nbody --solarCollision 1 --fragmentation 1`
- `./build/nbody --present 1`
- `./build/nbody --realSolarSystem 1 --collisions 0`

Mode intent:
- **IMPLEMENTED/MODE-DEPENDENT:** `--realSolarSystem --collisions 0` is the closest mode to compute-focused gravity benchmarking.
- **IMPLEMENTED/MODE-DEPENDENT:** `--present` is visualization-heavy and includes rendering/presentation overheads.
- **IMPLEMENTED/MODE-DEPENDENT:** `--solarCollision --fragmentation` is event-driven and includes additional collision/fragment overhead paths.

#### 6.2.2 GPU Implementation Summary

- **IMPLEMENTED:** Gravity computation uses one-thread-per-body accumulation in `compute_accel_tiled`.
- **IMPLEMENTED:** Timestep integration follows staged Leapfrog/Velocity-Verlet kernels (`update_vel_half`, `update_pos`, recompute acceleration, `finalize_vel`).
- **IMPLEMENTED:** Shared-memory tiling stages j-body `x,y,z,m` for reuse.
- **MODE-DEPENDENT:** Collision/fragment workflows can introduce additional kernels and host-assisted update paths (`stepSimulationMerge`).

Evidence:
- `src/kernels.cu`, `compute_accel_tiled`, `update_vel_half`, `update_pos`, `finalize_vel`, `detect_collision_pairs`, `resolve_collision_pairs`.
- `src/integrator.cu`, launch wrappers and dynamic shared-memory launch sizing.
- `app/sim.cpp`, `stepSimulation` and `stepSimulationMerge`.

#### 6.2.3 Tooling and Measurement Plan

Environment record (verbatim):
- CPU: GenuineIntel, 13th Gen Intel(R) Core(TM) i7-13620H, 10 cores, cache 24576 KB
- GPU: NVIDIA GeForce RTX 4050
- CUDA Toolkit Version: CUDA 13.0
- OS: Ubuntu 22.04
- Compiler: nvcc (record nvcc --version in artifacts)

Profiling tools:
- **DERIVED:** Use Nsight Systems (`nsys`) for timeline attribution, kernel/API breakdown, memcpy behavior, and synchronization analysis.
- **DERIVED:** Use Nsight Compute (`ncu`) for kernel counters (occupancy, memory behavior, instruction mix, branch behavior, SFU sensitivity).
- **MEASUREMENT-NEEDED:** Counter availability may depend on permissions/policy; record any collection limitations in artifacts.

#### 6.2.4 Timing Protocol (GPU)

- **IMPLEMENTED:** In benchmark mode (`cfg.benchmark`), timing is performed with CUDA events in `app/bench.cpp` and reported as `time_per_step_ms` and `IPS`.
- **DERIVED:** Compute-focused timing should exclude rendering/presentation overhead and use fixed-step repeated runs with warm-up excluded from reported stats.
- **MEASUREMENT-NEEDED:** For non-benchmark interactive modes (`--present`, collision-heavy demos), isolate compute vs render costs with timeline profiling before comparing step-time trends.
- **CODE-VERIFY-NEEDED:** If additional timing paths exist outside `app/bench.cpp`, they should be cited explicitly before being used as primary methodology evidence.

#### 6.2.5 Kernel Filtering and Reproducibility

- **DERIVED:** Maintain stable kernel-name filtering when extracting Nsight metrics to avoid mixed-kernel aggregation artifacts.
- **DERIVED:** Keep launch configuration fixed per experiment (`blockSize`, step count, mode, `dt`, `eps`, seed/init scenario).
- **DERIVED:** Separate compute-focused runs from visualization-heavy or event-driven runs in reporting tables and plots.
- **MEASUREMENT-NEEDED:** Publish profiler command lines, filters, and raw artifacts for audit replay.

#### 6.2.6 Data Products and Interpretation Scope

- **IMPLEMENTED:** Plot generation pipeline reads `ncu_metrics_summary.csv` via `plot.py` and writes figures into `plots/`.
- **IMPLEMENTED:** Existing figure references in this report are limited to those generated in the current `plots/` directory.
- **MEASUREMENT-NEEDED:** Preliminary summary plots should not be over-interpreted without kernel-isolated filtering and mode tags.

## 7. Next Step Placeholder

In the next revision, we will add:
- executed CPU baseline artifacts following Section 6.1 protocol,
- mode-tagged GPU profiling artifacts following Section 6.2 protocol,
- finalized kernel-filtered metric extraction and validation criteria.




## 8. CUDA Computational Strategy Analysis

### 8.0 Scope and Terms

This section analyzes CUDA computational strategy using code-consistent statements only.

- **IMPLEMENTED:** Proposal-compliant gravity mode refers to runs where collisions are disabled. In this mode, the gravity force evaluation and integrator kernels execute on the GPU for each timestep.
- **MODE-DEPENDENT:** Extended collision/fragment mode refers to runs with collision and fragmentation features enabled. In these modes, collision and fragmentation handling may include device-to-host and host-to-device transfers depending on runtime path.
- **IMPLEMENTED:** Kernel roles are separated at a high level into acceleration evaluation, velocity update kernels, and position update kernel.
- **IMPLEMENTED:** A tile is a shared-memory staging block for j-body attributes, storing position and mass fields in Structure-of-Arrays form (`x`, `y`, `z`, `m`) to increase data reuse.

### 8.1 One-Thread-Per-Body Strategy (IMPLEMENTED + implications)

- **IMPLEMENTED:** The force kernel maps one CUDA thread to one target body. Each thread accumulates contributions to its own acceleration output.
- **IMPLEMENTED:** This mapping avoids inter-thread write conflicts on acceleration arrays because each thread owns its accumulator state.

Warp divergence characteristics:
- **EXPECTED:** Divergence is limited by a boundary guard and a self-interaction skip condition.
- **EXPECTED:** The main interaction loop is otherwise structurally uniform, so control-flow irregularity should be limited in gravity-only force accumulation.
- **MEASUREMENT-NEEDED:** Branch efficiency and warp execution efficiency should be verified with Nsight Compute counters.

Register-pressure and loop-lifetime implications:
- **IMPLEMENTED:** A thread keeps invariant body state, running acceleration accumulators, and per-interaction temporaries live across a long inner loop.
- **EXPECTED:** Long live ranges increase register lifetime pressure and can reduce effective occupancy when combined with block-level resource usage.
- **MEASUREMENT-NEEDED:** Validate with register count (PTXAS), achieved occupancy, and scheduler statistics.

Alternative mappings:
- **EXPECTED:** Pairwise-thread and grid-based pair mappings require reductions or atomics to combine many interaction contributions into per-body outputs.
- **EXPECTED:** Those mappings increase synchronization complexity and can introduce additional contention; therefore, the implemented mapping is a pragmatic choice for correctness and simplicity.

### 8.2 Kernel Structure and Integrator Synchronization (IMPLEMENTED)

- **IMPLEMENTED:** The timestep follows Leapfrog/Velocity-Verlet staging, with velocity advanced as two half-kicks around a position drift.
- **IMPLEMENTED:** Acceleration must be recomputed after positions are updated because the force field depends on the full set of current body positions.
- **IMPLEMENTED:** Kernel boundaries provide explicit synchronization points between force evaluation and state updates.

Performance implications:
- **EXPECTED:** Launch overhead is most visible when problem size is small and arithmetic work per launch is limited.
- **EXPECTED:** At larger problem sizes, acceleration work typically dominates because pairwise interaction cost grows quadratically with body count.
- **MEASUREMENT-NEEDED:** Confirm with per-kernel time breakdown and launch/API timeline traces.

### 8.3 Shared-Memory Tiling (IMPLEMENTED + qualitative traffic model)

Purpose and mechanism:
- **IMPLEMENTED:** Shared-memory tiling is used to reduce redundant global loads of j-body data in all-to-all force evaluation.
- **IMPLEMENTED:** Each block stages a tile of j-body attributes (`x`, `y`, `z`, `m`) and reuses them across many interaction computations.
- **IMPLEMENTED:** Structure-of-Arrays layout supports coalesced global loads into shared memory.

Symbolic traffic model (no numeric substitution):
$$ \text{blocks} = \left\lceil \frac{N}{B} \right\rceil, \quad \text{tiles} = \left\lceil \frac{N}{B} \right\rceil $$
$$ \text{global bytes per tile per block} \propto B \cdot \text{sizeof}((x,y,z,m)_{j\text{-body}}) $$
$$ \text{total global bytes} \propto \text{blocks} \cdot \text{tiles} \cdot B \cdot \text{sizeof}((x,y,z,m)_{j\text{-body}}) $$
$$ \text{total interactions} \in O(N\cdot N) $$
$$ \text{bytes per interaction} = \frac{\text{total global bytes}}{\text{total interactions}} $$

Resource tradeoff:
- **EXPECTED:** Larger tile size increases reuse and reduces redundant global traffic per interaction.
- **EXPECTED:** Larger tile size can also increase resource pressure (registers/shared memory), which can reduce occupancy.
- **MEASUREMENT-NEEDED:** Validate tradeoff using achieved occupancy, memory-throughput counters, and stall breakdown.

### 8.4 Mode-Specific Bottlenecks

#### 8.4.A Mode: realSolarSystem, collisions=0
Command: `./build/nbody --realSolarSystem 1 --collisions 0`

Scope statement:
- **MODE-DEPENDENT:** This command is typically used as an interactive run, so rendering may be active unless explicitly disabled by a headless/benchmark path.
- **IMPLEMENTED:** Physics timestep in this mode remains GPU-resident for gravity and integrator kernels.

Expected bottlenecks (ranked):
- **Highest expected contributor:** Quadratic all-to-all acceleration workload in the gravity kernel.
- **High expected contributor:** Memory hierarchy behavior inside the acceleration kernel, including cache/shared-memory effectiveness.
- **High expected contributor:** Special-function latency in inverse-distance evaluation and its interaction with warp scheduling.
- **Moderate expected contributor:** Occupancy limits from register/shared-memory pressure.
- **Mode-dependent contributor:** CPU-GPU visualization transfer and render synchronization when visualization is enabled.
- **Regime-dependent contributor:** Kernel launch overhead relative to useful arithmetic work in small problem regimes.

Measurement checklist (MEASUREMENT-NEEDED):
- `nsys`: kernel time breakdown, CUDA API timeline, memcpy events, synchronization calls.
- `ncu`: achieved occupancy, SM efficiency, memory throughput, instruction mix, branch efficiency, special-function utilization.
- Counter access caveat: hardware counters may be blocked by permission policy (for example NVGPUCTRPERM).
- Fallback if counters are blocked: PTXAS register report plus occupancy estimation from launch configuration and kernel resource usage.

#### 8.4.B Mode: present
Command: `./build/nbody --present 1`

Scope statement:
- **MODE-DEPENDENT:** This mode emphasizes visualization and frame presentation, so graphics pipeline behavior can be co-dominant with CUDA kernels.

Expected bottlenecks (ranked):
- **Highest expected contributor:** Rendering pipeline cost, including draw submission and buffer update behavior.
- **High expected contributor:** CUDA-graphics synchronization or interop stalls when compute and rendering exchange state each frame.
- **Mode-dependent contributor:** CPU-GPU transfer overhead for visualization buffers or diagnostic overlays.
- **Mode-dependent contributor:** Frame pacing effects such as presentation synchronization, which can mask pure compute throughput.
- **Persistent compute contributor:** Gravity acceleration kernel still contributes a core compute cost, especially as body count grows.

Measurement checklist (MEASUREMENT-NEEDED):
- `nsys`: overlap and ordering of CUDA kernels, graphics API activity, memcpy activity, synchronization points.
- `ncu`: per-kernel occupancy, execution efficiency, memory behavior, instruction profile.
- Counter access caveat: hardware counter permissions may restrict Nsight Compute data collection.
- Fallback if counters are blocked: PTXAS register count and occupancy estimate; rely on timeline-level evidence from `nsys`.

#### 8.4.C Mode: solarCollision with fragmentation
Command: `./build/nbody --solarCollision 1 --fragmentation 1`

Scope statement:
- **MODE-DEPENDENT:** This mode adds collision and fragmentation paths beyond proposal-compliant gravity-only execution.
- **MODE-DEPENDENT:** Collision resolution and fragmentation may involve device-only handling or device-host-device round-trips depending on runtime path.

Expected bottlenecks (ranked):
- **Highest expected contributor:** Collision pair detection and associated traversal overhead on top of gravity computation.
- **Mode-dependent contributor:** Collision resolution path cost, including any host round-trip and re-upload overhead when host-side handling is used.
- **High expected contributor:** Fragmentation slot management and activation/deactivation bookkeeping.
- **High expected contributor:** Additional memory traffic for debris/body state arrays and associated metadata.
- **High expected contributor:** Higher branch divergence risk in collision/fragment kernels due to event-driven, non-uniform work.
- **High expected contributor:** Additional synchronization points across detection, resolution, and state-compaction/spawn paths.

Stability and robustness requirements:
- **IMPLEMENTED/EXPECTED:** Guarding against invalid floating-point states is required when collision impulses, merges, and fragment spawning alter state abruptly.
- **IMPLEMENTED/EXPECTED:** Fixed-capacity arrays and rendering buffers require strict bounds handling and overflow-safe activation policies.
- **EXPECTED:** Event-driven burst behavior can produce frame-to-frame work imbalance and should be profiled as workload variability, not only average cost.

Measurement checklist (MEASUREMENT-NEEDED):
- `nsys`: mode-specific kernel breakdown, host callbacks, memcpy bursts, synchronization hotspots.
- `ncu`: branch behavior, memory throughput, occupancy changes between gravity and collision kernels, instruction mix differences.
- Counter access caveat: Nsight Compute counters may require elevated permissions.
- Fallback if counters are blocked: PTXAS register reports, occupancy estimates, and detailed timeline attribution from `nsys`.

### 8.5 Next Step: Improvements (PLACEHOLDER)

The next revision will address the following headings:
- confirming one-thread-per-body mapping efficiency,
- validating the three-kernel design overhead,
- validating shared-memory tiling effectiveness,
- collision/fragmentation stability versus performance tradeoffs.

Solutions and optimizations will be added in the next revision.



## 9. Performance Improvements and Optimization Strategy

### 9.1 Shared Memory Tiling Improvement

Implementation evidence:
- **IMPLEMENTED:** The acceleration path uses `compute_accel_tiled` with shared-memory staging of j-body fields (`x`, `y`, `z`, `m`) into per-block tile buffers (`sx`, `sy`, `sz`, `sm`), with synchronization barriers around load/compute phases.
- **IMPLEMENTED:** Launch configuration allocates dynamic shared memory for four float arrays per block in the tiled force kernel.
- **IMPLEMENTED:** Compared with a naive global-only access pattern, tiled staging converts repeated j-body fetches into tile-level global loads followed by intra-block reuse.

Validation evidence from `plots/`:
- **MEASURED:** `plots/time_vs_N.png` shows monotonic runtime growth across the sampled N range ($512 \rightarrow 16384$).
- **MEASURED:** `plots/time_per_interaction_vs_N.png` shows a decreasing per-interaction trend over the sampled range ($512 \rightarrow 16384$).
- **DERIVED:** These trends are aligned with stronger amortization as workload size grows.
- **EXPECTED:** Shared-memory reuse is a plausible contributor to the observed trend.
- **MEASUREMENT-NEEDED:** Causal attribution requires controlled A/B profiling against a non-tiled baseline.
- **DERIVED:** Interpretation of these compute trends follows Section 6 timing boundaries: compute-focused runs exclude render/present overhead, and mode-heavy paths must be analyzed separately.

![Kernel time scaling](plots/time_vs_N.png)
![Time per interaction scaling](plots/time_per_interaction_vs_N.png)

Block-size note:
- **IMPLEMENTED:** Launch wrappers accept `blockSize` as a runtime parameter.
- **MEASUREMENT-NEEDED:** The current `plots/` set does not contain a block-size sweep figure; block-size sensitivity is therefore not directly validated by existing plots.

### 9.2 Block Size Tuning Strategy

- **EXPECTED:** Block size controls a reuse-versus-resources tradeoff: larger blocks increase tile reuse opportunity while increasing shared-memory/register pressure and affecting occupancy.
- **IMPLEMENTED:** Benchmark and launch interfaces are parameterized by `blockSize`.
- **IMPLEMENTED:** The current main execution path deploys a fixed block size, so existing plots represent one deployed configuration rather than a multi-configuration sweep.

Validation status:
- **MEASUREMENT-NEEDED:** A comparison across candidate block sizes (for example `64/128/256/512`) is not present in `plots/`.
- **MEASUREMENT-NEEDED:** Without a block-size comparison plot, no data-backed optimum can be claimed.

Evidence gap note:
- **MEASUREMENT-NEEDED:** Add a dedicated block-size sweep figure before claiming a selected optimum.

### 9.3 Kernel Decomposition and Synchronization Strategy

Implementation evidence:
- **IMPLEMENTED:** The timestep is decomposed into velocity half-update, position update, acceleration recomputation, and velocity finalization around the force kernel.
- **IMPLEMENTED:** Full fusion is not used because the second force evaluation depends on globally updated positions.
- **IMPLEMENTED:** Collision-enabled paths insert additional detection/resolution stages and synchronization points between drift and force recomputation.

- **DERIVED:** Step decomposition model:
$$ T_{\text{step}} = T_{\text{accel}} + T_{\text{updates}} + T_{\text{launch}} $$

Interpretation using available plots:
- **MEASURED:** `plots/time_per_interaction_vs_N.png` is consistent with fixed-overhead amortization as N increases.
- **EXPECTED:** Smaller N regimes are more sensitive to launch/synchronization overhead.
- **EXPECTED:** Larger N regimes are increasingly dominated by acceleration workload.
- **MEASUREMENT-NEEDED:** Per-kernel timeline decomposition from `nsys` is required to validate term-wise dominance directly.

![Time per interaction trend for overhead interpretation](plots/time_per_interaction_vs_N.png)

#### 9.3.1 Baseline Mock (Derived) Without a Naive Kernel

Definitions:
- **IMPLEMENTED (optimized path):** The current acceleration kernel is tiled/shared-memory (`compute_accel_tiled`), uses SoA body arrays (`x`,`y`,`z`,`m`), and one-thread-per-body accumulation.
- **DERIVED (baseline mock):** A hypothetical naive global-only kernel loads j-body fields (`x`,`y`,`z`,`m`) from global memory per interaction with no tile reuse.
- **DERIVED:** The baseline mock is a reference traffic model only; it is not executed in current artifacts.
- **DERIVED:** Baseline-mock interpretation is constrained by Section 6 methodology: it applies to compute-focused gravity timing and not to render-dominated or event-driven end-to-end timings.

Traffic derivation:
- **DERIVED:** Let `B` be block size.
- **DERIVED:** $\text{blocks}=\lceil N/B \rceil$, $\text{tiles}=\lceil N/B \rceil$.

Tiled model:
- **DERIVED:** Global bytes per tile per block for j-body fields: $16B$ bytes.
- **DERIVED:** $\text{bytes}_{\text{tiled}} \approx \text{blocks}\cdot\text{tiles}\cdot16B$.
- **DERIVED:** $\text{interactions} \approx N(N-1)$.
- **DERIVED:** $\text{bytes/interaction}_{\text{tiled}} = \text{bytes}_{\text{tiled}} / \text{interactions}$.

Naive mock model:
- **DERIVED:** $\text{bytes/interaction}_{\text{naive}} \approx 16$ bytes for j-body fields as a lower-bound reference.
- **EXPECTED:** Real naive traffic can be higher depending on cache behavior and implementation details.

Traffic reduction factor:
- **DERIVED:**
$$ \text{reduction\_factor}= \frac{\text{bytes/interaction}_{\text{naive}}}{\text{bytes/interaction}_{\text{tiled}}} $$
- **DERIVED (asymptotic):** For large N, $\text{bytes/interaction}_{\text{tiled}} \approx 16/B$, so $\text{reduction\_factor} \approx B$.

Scope and limitation of the reduction factor:
- **DERIVED:** The reduction factor refers specifically to streamed j-body global traffic.
- **DERIVED:** The model excludes per-thread i-body loads, global result stores, shared-memory traffic volume, cache hit/miss effects, and other memory streams.
- **DERIVED:** The factor is therefore an upper bound on this traffic component, not a direct runtime-speedup claim.
- **EXPECTED:** True runtime impact depends on whether the dominant bottleneck is memory, compute throughput, or special-function latency.

Assumptions and limitations:
- **DERIVED (assumption):** SoA layout and ideal coalescing for modeled global j-body loads.
- **DERIVED (assumption):** Uniform interaction traversal with no control-path perturbations in the traffic counting model.
- **MEASUREMENT-NEEDED (limitation):** No executed naive kernel exists for direct A/B timing attribution.
- **MEASUREMENT-NEEDED (limitation):** Runtime effects from register pressure, cache behavior, and scheduler stalls require profiler counters/timelines.
- **MEASUREMENT-NEEDED (limitation):** Counter-permission constraints can limit direct metric access in some environments.

Interpretation bounds:
- **EXPECTED:** Under a memory-bound regime, `reduction_factor` provides an approximate upper bound on potential time improvement, but real hardware behavior may shift the bottleneck toward compute or special-function units.
- **MEASUREMENT-NEEDED:** Without direct naive-vs-tiled A/B profiling, numeric runtime improvement cannot be asserted.

#### 9.3.2 What Changes With N (Regimes)

- **EXPECTED — Small-N Regime:** Launch/synchronization overhead and limited warp supply are relatively more visible; latency hiding is weaker.
- **MEASURED/EXPECTED — Mid-N Regime:** The decreasing per-interaction trend in `plots/time_per_interaction_vs_N.png` is consistent with overhead amortization and increasing relative dominance of core acceleration work.
- **EXPECTED — Large-N Regime:** Asymptotic $O(N^2)$ interaction cost dominates; resource pressure (register/shared-memory) and special-function latency become increasingly important constraints.

Plot linkage and causality scope:
- **MEASURED:** `plots/time_vs_N.png` and `plots/time_per_interaction_vs_N.png` provide qualitative evidence consistent with regime transitions.
- **DERIVED:** The observed decreasing per-interaction trend is consistent with amortization of fixed costs and increasing relative dominance of tiled reuse.
- **MEASUREMENT-NEEDED:** Plot trends alone do not prove mechanism; controlled profiling is required for causal attribution.

### 9.4 Mode-Specific Improvements

#### 9.4.A Mode: `--realSolarSystem --collisions 0`
Command: `./build/nbody --realSolarSystem 1 --collisions 0`

- **IMPLEMENTED optimizations that apply:** Shared-memory tiled acceleration, SoA layout, one-thread-per-body mapping, and staged Leapfrog integration.
- **DERIVED:** This mode is the best case for baseline-mock interpretation because collision/fragmentation overhead paths are absent.
- **MODE-DEPENDENT:** If rendering is enabled, render-path and presentation costs are separate from core gravity-kernel compute cost.
- **MEASURED:** `plots/time_vs_N.png` and `plots/time_per_interaction_vs_N.png` are qualitatively consistent with expected scaling and amortization behavior in the sampled runs.
- **MEASUREMENT-NEEDED:** Mode-isolated profiling remains necessary to separate compute-only behavior from render-side overhead when visualization is active.

#### 9.4.B Mode: `--present`
Command: `./build/nbody --present 1`

- **IMPLEMENTED optimizations that apply:** Tiled gravity kernel, SoA layout, one-thread-per-body mapping, staged integration.
- **MODE-DEPENDENT:** Visualization pipeline work can dominate or mask pure compute behavior.
- **MODE-DEPENDENT:** Presentation synchronization (including vsync/frame pacing effects) can introduce stalls unrelated to raw kernel throughput.
- **MODE-DEPENDENT:** GPU/CPU synchronization overhead can increase if graphics interop or frequent visualization-side updates are active.
- **DERIVED:** Baseline-mock traffic improvement still applies to the gravity kernel component, but does not model render/present bottlenecks.
- **MEASUREMENT-NEEDED:** Current `plots/` are not present-mode-isolated; defensible attribution requires `nsys` timeline analysis.

#### 9.4.C Mode: `--solarCollision --fragmentation`
Command: `./build/nbody --solarCollision 1 --fragmentation 1`

- **IMPLEMENTED optimizations that apply:** Tiled acceleration remains active in the gravity phase.
- **IMPLEMENTED path detail:** Collision/fragment workflows include explicit collision-pair detection kernels and host-assisted merge/fragment paths in applicable runtime branches.
- **IMPLEMENTED path detail:** Host-side merge/fragment handling may copy collision pairs and body state to host, perform update/spawn logic, and upload modified state back to device.
- **EXPECTED performance cost:** Conditional/event-driven collision logic can increase branch divergence relative to gravity-only flow.
- **EXPECTED performance cost:** Dynamic activation/deactivation of bodies and spawn bookkeeping can increase synchronization and memory-traffic pressure.
- **EXPECTED performance cost:** Event-driven workload variability can produce non-uniform per-step cost and transient overhead spikes.
- **DERIVED:** Baseline-mock gravity traffic reduction remains valid for the gravity component only; total step behavior includes additional non-gravity costs.
- **MEASUREMENT-NEEDED:** Dedicated collision/fragment traces are needed to quantify overhead composition and variability.

### 9.5 Summary of Observed Improvements

| Optimization | Applied In | Observed Effect | Evidence Source |
|---|---|---|---|
| Shared-memory tiled acceleration (`compute_accel_tiled`) | All modes | **IMPLEMENTED:** tiled reuse exists; **MEASURED:** `time_per_interaction_vs_N` trend is consistent with amortization; **MEASUREMENT-NEEDED:** causal proof versus naive requires A/B | `src/kernels.cu`, `src/integrator.cu`, `plots/time_per_interaction_vs_N.png` |
| SoA layout for body fields (`x`,`y`,`z`,`m`) | All modes | **IMPLEMENTED:** field-wise contiguous access is present; **EXPECTED:** compatible with coalesced load behavior | `src/kernels.cu` |
| Kernel decomposition for Leapfrog/Velocity-Verlet | All modes | **IMPLEMENTED:** staged updates and force recomputation; **MEASURED/EXPECTED:** trend aligns with overhead amortization across N; **MEASUREMENT-NEEDED:** term-wise attribution requires timeline profiling | `app/sim.cpp`, `src/integrator.cu`, `plots/time_per_interaction_vs_N.png` |
| Baseline Mock traffic model (naive global-only reference) | Gravity-dominant analysis | **DERIVED:** provides a traffic-reduction bound for j-body streamed global traffic only; **MEASUREMENT-NEEDED:** no numeric runtime claim without A/B | Section 9.3.1 derivation |
| Collision/fragment host-assisted path | Collision/fragment mode | **IMPLEMENTED:** host round-trip path exists in merge/fragment workflows; **EXPECTED:** may increase overhead and variability beyond gravity-kernel scaling | `app/sim.cpp` |
| Block-size tuning infrastructure | All modes (infrastructure) | **IMPLEMENTED:** parameterized launch interfaces exist; **MEASUREMENT-NEEDED:** no block-size sweep plot currently validates an optimum | `app/bench.cpp`, `src/integrator.cu`, evidence gap in `plots/` |

### 9.6 Assumptions and Limitations (Section-Level)

- **DERIVED:** The baseline-mock model assumes SoA indexing and idealized j-body global traffic accounting for tiled versus naive reference paths.
- **DERIVED:** The model does not capture full-memory-stream behavior, cache dynamics, register-spill effects, or detailed scheduler stall composition.
- **MEASUREMENT-NEEDED:** Causal confirmation of mechanism-level claims requires controlled tiled-vs-naive A/B profiling and per-kernel counter/timeline evidence.
- **MEASUREMENT-NEEDED:** Where profiler counter access is restricted, timeline evidence plus compiler resource reports should be treated as partial, not final, attribution evidence.

## 10. Scalability Analysis (Formal)

### 10.1 Scope Statement

- **IMPLEMENTED:** This section analyzes scalability of the implemented CUDA N-body pipeline with problem size $N$, separating gravity-only and collision/fragmentation-enhanced execution paths.
- **MEASURED:** Qualitative trend evidence is taken from `plots/time_vs_N.png` and `plots/time_per_interaction_vs_N.png` over the current sampled range $N=512..16384$.
- **MEASUREMENT-NEEDED:** Mode-isolated scalability decomposition for render-heavy and collision-heavy workflows requires dedicated profiling traces.
- **DERIVED:** Unless explicitly mode-tagged otherwise, scalability plots are interpreted under the compute-focused methodology in Section 6.2, and are not automatically representative of present-mode or collision/fragment end-to-end behavior.

### 10.2 Definitions and Complexity Decomposition

Definitions:
- **DERIVED:** $N$ is the body-count parameter passed to simulation kernels.
- **DERIVED:** Per-step wall time is $T_{\text{step}}$.
- **DERIVED:** Gravity interaction count is approximated as $N(N-1)$.

Complexity by stage:
- **IMPLEMENTED/DERIVED:** Gravity force recomputation is $O(N^2)$ per call.
- **IMPLEMENTED/DERIVED:** Velocity and position updates are $O(N)$ per update kernel.
- **IMPLEMENTED/DERIVED:** Collision pair detection is $O(N^2)$ worst-case because `detect_collision_pairs` iterates `i in [0,n)` and `j in (i,n)` without an active-body prefilter in loop bounds.
- **EXPECTED:** Effective detected-collision volume depends on runtime body state (for example zero radii or deactivated masses), but pair traversal cost is still driven by full `n` in the current kernel.
- **CODE-VERIFY-NEEDED:** If an alternate branch or future revision introduces explicit inactive-body early-skip checks inside the loop body, that would change effective work but not the worst-case $O(N^2)$ traversal bound.
- **MODE-DEPENDENT/DERIVED:** Collision resolution and fragmentation overhead scale with detected events and selected runtime path.

Evidence:
- `src/kernels.cu`, `compute_accel_tiled`: tile-loop plus inner interaction loop confirms dense pair accumulation.
- `src/kernels.cu`, `update_vel_half` / `update_pos` / `finalize_vel`: one-thread-per-body linear update kernels.
- `src/kernels.cu`, `detect_collision_pairs`: `for (int j = i + 1; j < n; ++j)` confirms pairwise collision search.
- `app/sim.cpp`, `stepSimulation` and `stepSimulationMerge`: mode-dependent step orchestration and collision/fragment paths.

### 10.3 Scaling Metrics

- **DERIVED:** `time_per_step` denotes end-to-end step cost under a fixed mode/configuration.
- **DERIVED:** Gravity-normalized interaction cost:
$$ \text{time\_per\_interaction} = \frac{T_{\text{step,gravity-dominant}}}{N(N-1)} $$
- **DERIVED:** Interactions per second:
$$ \text{IPS} = \frac{N(N-1)}{T_{\text{step}}} $$
- **IMPLEMENTED:** Benchmark output prints a table header containing `time_per_step_ms` and `IPS`, then prints rows as `N | dt | eps | blockSize | time_per_step_ms | IPS`.
- **IMPLEMENTED:** In the current code path, `IPS` is computed in `app/bench.cpp` from measured step time (`timePerStepSec`); `plot.py` derives `time_per_interaction_ns` and does not print benchmark `IPS` rows.

Evidence:
- `app/bench.cpp`, `runBenchmark`: `std::cout << "N | dt | eps | blockSize | time_per_step_ms | IPS\n";` plus per-row print formatting confirm field names and table order.
- `plot.py`, `main`: computes `pair_interactions` and `time_per_interaction_ns` for scaling plots.

### 10.4 N-Regime Interpretation

- **EXPECTED — Small-N regime:** Launch/synchronization overhead and limited warp supply are relatively more visible; latency hiding is weaker.
- **MEASURED/EXPECTED — Mid-N regime:** The observed per-interaction trend is consistent with overhead amortization and stronger utilization.
- **EXPECTED — Large-N regime:** Quadratic interaction cost dominates; resource pressure and special-function latency become more relevant.

Plot support:
- **MEASURED:** `plots/time_vs_N.png` shows monotonic step-time growth over sampled N.
![Scalability trend: step time vs N](plots/time_vs_N.png)
*Caption: Qualitative evidence that step time increases with N; this aligns with dense pairwise workload growth.*

- **MEASURED:** `plots/time_per_interaction_vs_N.png` shows decreasing normalized interaction cost in sampled data.
![Scalability trend: time per interaction vs N](plots/time_per_interaction_vs_N.png)
*Caption: Qualitative evidence consistent with fixed-cost amortization; mechanism attribution still requires profiling confirmation.*

### 10.5 Mode-Specific Scalability Context

Command context:
- `./build/nbody --realSolarSystem 1 --collisions 0`
- `./build/nbody --present 1`
- `./build/nbody --solarCollision 1 --fragmentation 1`

Interpretation:
- **MODE-DEPENDENT:** Gravity-only mode best reflects compute-kernel scalability without collision/fragment event overhead.
- **MODE-DEPENDENT:** `--present` includes rendering/presentation costs that can decouple frame behavior from pure compute scaling.
- **MODE-DEPENDENT:** `--solarCollision --fragmentation` adds event-driven collision/debris work that alters scaling shape.

Evidence:
- `app/main.cpp`: mode presets and runtime branching (`present`, `solarCollision`, collision model selection).
- `app/sim.cpp`, `stepSimulationMerge`: host-assisted merge/fragment path and state upload behavior.

### 10.6 Limitations and Measurement Needs

- **MEASUREMENT-NEEDED:** Current plots are not mode-isolated and do not provide per-kernel share decomposition.
- **MEASUREMENT-NEEDED:** Definitive scalability decomposition requires `nsys` timeline attribution per mode.
- **MEASUREMENT-NEEDED:** Counter-level validation requires `ncu` metrics with stable kernel filtering.

## 11. Roofline-Oriented Classification (Qualitative + Derivation)

### 11.1 Scope Statement

- **DERIVED:** This section provides a roofline-oriented classification framework for the implemented tiled gravity kernel (`compute_accel_tiled`) without asserting unmeasured absolute bottleneck labels.
- **MEASURED:** Available evidence is trend-level and comes from plots generated by `plot.py` from `ncu_metrics_summary.csv` (default input path).
- **CODE-VERIFY-NEEDED:** Input-source override paths should be rechecked when external wrappers are used; in the inspected script, CSV override is available via CLI (`--csv`) and no environment-variable override is visible.
- **MEASUREMENT-NEEDED:** Final roofline placement requires kernel-isolated counters plus hardware peak references captured in reproducible profiling artifacts.

Evidence:
- `src/kernels.cu`, `compute_accel_tiled`: confirms tiled interaction loop, shared-memory staging, and `rsqrtf`-based inverse-distance evaluation.
- `src/integrator.cu`, `launchComputeAccel`: confirms dynamic shared-memory launch (`shmem = 4 * blockSize * sizeof(float)`) for tiled gravity execution.
- `plot.py`, module-level `CSV_PATH` and `main`: confirms default CSV input (`ncu_metrics_summary.csv`) and output files under `plots/`.

### 11.2 Definitions

- **DERIVED:** Arithmetic intensity for global memory perspective:
$$ AI_{\text{global}} = \frac{\text{FLOPs per interaction}}{\text{global bytes per interaction}} $$
- **DERIVED:** For tiled j-body streaming:
$$ \text{bytes}_{\text{tiled}} \propto \left\lceil \frac{N}{B} \right\rceil \left\lceil \frac{N}{B} \right\rceil 16B,\quad \text{interactions} \approx N(N-1) $$
$$ \text{bytes/interaction}_{\text{tiled}} = \frac{\text{bytes}_{\text{tiled}}}{\text{interactions}} $$
- **DERIVED (asymptotic):**
$$ \text{bytes/interaction}_{\text{tiled}} \sim \frac{16}{B} $$

FLOP accounting band:
- **DERIVED:** Let $F_{\text{int}}$ denote FLOPs-per-interaction under an explicit counting convention.
- **DERIVED:** $F_{\text{int}}$ is convention-dependent because FMA accounting and `rsqrtf` weighting can differ across methodology choices.
- **DERIVED:** Therefore:
$$ AI_{\text{global}} = \frac{F_{\text{int}}}{\text{bytes/interaction}_{\text{tiled}}} $$
should be interpreted as a band unless the report fixes one formal FLOP convention.

Evidence:
- `src/kernels.cu`, `compute_accel_tiled`: arithmetic path includes subtract/multiply/add, `rsqrtf`, inverse-cube scaling, and accumulation.
- `src/integrator.cu`, `launchComputeAccel`: block-size-dependent shared-memory footprint for four tiled arrays (`x`,`y`,`z`,`m`).

### 11.3 Qualitative Classification Logic

- **DERIVED/EXPECTED:** Tiling reduces modeled streamed global bytes per interaction asymptotically with increasing block reuse.
- **EXPECTED:** Lower streamed global traffic can shift limiting factors toward arithmetic/SFU throughput and scheduler latency-hiding limits.
- **MEASUREMENT-NEEDED:** Whether the implementation is memory-bound, compute-bound, or transitional at a specific $N$ requires measured stall and utilization counters on the same kernel invocation set.

### 11.4 Plot-Linked Observations and Limits

- **MEASURED:** `plots/sm_dram_throughput_vs_N.png` and `plots/sm_dram_throughput_vs_N_symlog.png` show mixed throughput trends across sampled N values in the current summary dataset ($N=512..16384$).
![SM and DRAM throughput trends](plots/sm_dram_throughput_vs_N.png)
*Caption: Throughput trend view from current summary CSV; suitable for qualitative context but not for kernel-level causal attribution.*

![SM and DRAM throughput trends (symlog)](plots/sm_dram_throughput_vs_N_symlog.png)
*Caption: Symlog view emphasizes mixed-scale behavior; interpretation remains preliminary without strict kernel filtering.*

- **MEASURED:** `plots/time_vs_sm_scatter.png` and `plots/time_vs_dram_scatter.png` show correlation structure between summary metrics and time.
- **MEASUREMENT-NEEDED:** These figures are not sufficient to establish causal bottleneck direction because they aggregate precomputed summary values rather than kernel-isolated counter traces.

Evidence:
- `plot.py`, save calls in `main`: confirms exact output filenames `sm_dram_throughput_vs_N.png`, `sm_dram_throughput_vs_N_symlog.png`, `time_vs_sm_scatter.png`, `time_vs_dram_scatter.png`.

### 11.5 Required Measurements for Final Roofline Classification

- **MEASUREMENT-NEEDED:** Memory subsystem counters (DRAM throughput, cache behavior, memory-related stalls) for `compute_accel_tiled`.
- **MEASUREMENT-NEEDED:** Compute/scheduling counters (SM activity, issue efficiency, eligible warps, instruction mix, SFU utilization) for the same kernel and launch geometry.
- **MEASUREMENT-NEEDED:** Stable kernel filtering and mode-specific reproducible runs to avoid mixed-kernel aggregation.
- **MEASUREMENT-NEEDED:** Hardware peak ceilings for roofline placement must be taken from captured profiler/system artifacts in this workflow; do not assume public-spec peak values.
- **IMPLEMENTED:** Environment identity for this study is recorded in Section 6 (CPU: GenuineIntel 13th Gen Intel(R) Core(TM) i7-13620H; GPU: NVIDIA GeForce RTX 4050; CUDA Toolkit: 13.0; OS: Ubuntu 22.04; compiler path includes `nvcc` artifact capture).

## 12. Collision + Fragmentation Numerical Stability

### 12.1 Scope Statement

- **IMPLEMENTED:** This section covers numerical-stability-relevant behavior visible in gravity-only and collision/fragmentation-enabled execution paths.
- **EXPECTED:** Collision, merge, and fragmentation events can degrade long-horizon conservation relative to gravity-only stepping.
- **MEASUREMENT-NEEDED:** Stability characterization still requires controlled parameter sweeps and long-horizon monitoring runs.

### 12.2 Stability Drivers

- **IMPLEMENTED:** Softening enters gravity distance denominator as $r^2+\epsilon^2$, regularizing near-field acceleration.
- **IMPLEMENTED:** Timestep $\Delta t$ controls integration stability through repeated half-kick/drift/finalize updates.
- **IMPLEMENTED:** Collision response uses impulse-style updates plus penetration correction.
- **IMPLEMENTED:** Fragmentation paths assign debris offsets/velocities using jitter plus spread/clamp controls.
- **EXPECTED:** Large $\Delta t$, insufficient softening, and event-driven impulses/spawns can amplify numerical sensitivity.

Evidence:
- `src/kernels.cu`, `compute_accel_tiled` and `compute_energy_tiled`: both use softened distance terms (`dist2 + eps2`) and reciprocal/sqrt evaluation.
- `app/sim.cpp`, `stepSimulation`: confirms Leapfrog order and collision-resolve insertion between drift and acceleration recomputation.
- `app/sim.cpp`, `stepSimulationMerge`: confirms host-side merge/fragment logic with energy threshold, jitter, and speed-clamp paths.
- `app/config.h`, `Config` fields: contains `dt`, `eps`, `fragmentEnergyThreshold`, `fragmentSpeedSpread`, `fragmentSpeedClamp`, and debris/collision control parameters.

### 12.3 Conservation and Mode-Dependent Invariants

- **DERIVED/IMPLEMENTED:** Gravity-only stepping uses a Leapfrog/Velocity-Verlet sequence with softened interactions; exact per-step energy conservation is not enforced in finite precision.
- **IMPLEMENTED/EXPECTED:** Elastic-collision updates include penetration correction and floating-point arithmetic; strict total-energy invariance is therefore not guaranteed in practice.
- **IMPLEMENTED:** Merge mode explicitly replaces two bodies with a COM state, which is inelastic and not energy-conservative by design.
- **IMPLEMENTED:** Fragmentation is event-driven with explicit mass repartition and debris spawning logic; Hamiltonian conservation is not enforced.

Evidence:
- `app/sim.cpp`, `resolveElasticHost` and merge branches inside `stepSimulationMerge`: confirms impulse path, merge COM updates, and non-conservative event handling.
- `src/kernels.cu`, `resolve_collision_pairs`: confirms impulse-based collision updates plus positional penetration correction on device path.
- `app/sim.cpp`, fragmentation branches (`doFragment` logic): confirms threshold-triggered debris spawning and source-body deactivation.

### 12.4 Numerical Failure Modes and Guards

Potential failure modes:
- **EXPECTED:** Near-zero distance singular behavior without adequate regularization.
- **EXPECTED:** Timestep-induced integration blow-up under aggressive stepping.
- **EXPECTED:** Event bursts (many collisions/spawns) creating transient instability and work imbalance.

Implemented guards and diagnostics:
- **IMPLEMENTED:** Softened denominator in force and potential computations.
- **IMPLEMENTED:** Collision-pair capacity limits and truncation safeguards.
- **IMPLEMENTED:** Debris activation limits and slot-availability checks for fragmentation.
- **IMPLEMENTED:** Runtime sanity checks for invalid states are available when `cfg.sanityChecks` is enabled; this path checks finite positions and applies a reset/deactivation policy on detected failures.
- **IMPLEMENTED:** Periodic diagnostics for total energy, angular momentum, and COM are available when corresponding print flags are enabled.
- **CODE-VERIFY-NEEDED:** If additional NaN/Inf guards exist outside `app/main.cpp` and `app/sim.cpp`, they should be cited explicitly before broadening this claim.

Evidence:
- `src/integrator.cu`, `allocateDeviceArrays`: `maxPairs` allocation and pair-buffer capacity management.
- `app/sim.cpp`, `stepSimulation` and `stepSimulationMerge`: pair-count clipping, free-slot checks, debris caps, and host-device state transfer guards.
- `app/main.cpp`, `cfg.sanityChecks` block: finite-value checks, failure logging, reset/deactivation writes, and state re-upload.
- `app/main.cpp`, `computeTotalEnergy` / `computeAngularMomentum` / `computeCOM` call sites: periodic diagnostics under flag-controlled conditions.

### 12.5 Measurement Needs for Stability Validation

- **MEASUREMENT-NEEDED:** Energy drift versus step count under fixed mode/configuration.
- **MEASUREMENT-NEEDED:** Sensitivity sweeps over $\Delta t$ and $\epsilon$ with consistent initialization.
- **MEASUREMENT-NEEDED:** Collision/fragment event-rate versus stability diagnostics under mode-controlled workloads.
- **MEASUREMENT-NEEDED:** Separate reporting for gravity-only and collision/fragment modes to avoid mixing conservative and non-conservative regimes.

## 13. Visualization / Rendering Implementation Overview

### 13.1 Scope Statement

- **IMPLEMENTED:** This section documents the rendering/dataflow path visible in current source files and how it interacts with CUDA simulation state.
- **MODE-DEPENDENT:** Rendering overhead relevance depends on run mode and enabled overlays.
- **MEASUREMENT-NEEDED:** End-to-end frame-time attribution still requires graphics-aware timeline profiling.
- **DERIVED:** Mode definitions and timing boundaries follow Section 6.2: compute-focused benchmark timing is separate from present-mode frame pacing and visualization overhead.

### 13.2 Rendering Stack and Frame Loop

- **IMPLEMENTED:** Rendering uses OpenGL with GLFW and GLEW, managed by the `Renderer` module.
- **IMPLEMENTED:** Main loop performs simulation update(s), then visualization data preparation, then GPU buffer updates, then draw/present.
- **IMPLEMENTED:** Frame presentation uses swap-buffer flow via GLFW.

Evidence:
- `CMakeLists.txt`, `find_package(OpenGL/glfw3/GLEW)` and `target_link_libraries`: confirms linked graphics stack.
- `vis/renderer.cpp`, includes `<GL/glew.h>` and `<GLFW/glfw3.h>`, plus `glfwInit`, `glewInit`, `glfwSwapBuffers`: confirms runtime backend.
- `app/main.cpp`, main render loop (`while (!renderer.shouldClose())`): confirms ordering of step, data updates, and `renderer.render()`.

### 13.3 CUDA–Visualization Data Flow

- **IMPLEMENTED:** Current pipeline uses explicit CUDA device-to-host copies for visualization state (positions and optional velocities/energies), followed by CPU-side packing/interleaving and OpenGL buffer updates.
- **IMPLEMENTED:** When `cfg.render` is enabled (render-enabled main loop), position arrays (`d.x`, `d.y`, `d.z`) are copied device-to-host every frame before `renderer.updatePositions`.
- **IMPLEMENTED:** Velocity arrays (`d.vx`, `d.vy`, `d.vz`) are copied only when `needVelCopy` is true (`showVelocityVectors`, `showEnergyColor`, `demoSlingshot`, `printCOM`, or `cometTail`).
- **IMPLEMENTED:** Energy buffer (`d.energy`) is copied only when energy coloring is enabled (`needEnergy` / `showEnergyColor` path).
- **IMPLEMENTED:** OpenGL buffers are updated through `glBufferSubData` in renderer update functions.
- **DERIVED:** In the inspected source set, the visualization path is copy-based rather than direct CUDA-graphics interop.
- **CODE-VERIFY-NEEDED:** To rule out hidden interop paths outside the inspected files, verify absence/presence of `cudaGraphics*`/`cudaGL*` usage in any additional modules.
- **MODE-DEPENDENT:** Copy volume and update frequency increase when overlays requiring additional state are enabled.

Evidence:
- `app/main.cpp`, render loop copy/update blocks: unconditional per-frame `cudaMemcpy` for `d.x/d.y/d.z`; conditional `needVelCopy` block for `d.v*`; conditional `needEnergy` block for `d.energy`; followed by `renderer.update*`.
- `app/render_utils.cpp`, `interleavePositions` and color/size helpers: confirms CPU-side packing prior to renderer buffer updates.
- `vis/renderer.cpp`, `updatePositions/updateColors/updateSizes/updateTrails/updateVelocityLines/updateCloseLines/updateAuxLines`: confirms `glBufferSubData` upload path.

### 13.4 Visual Features and Their Runtime Interaction

- **IMPLEMENTED:** Visual features include trails (`cfg.trails`), velocity-vector overlays (`cfg.showVelocityVectors`), close-encounter highlight lines (`cfg.highlightCloseEncounters`), comet-tail auxiliary lines (`cfg.cometTail`), energy-based coloring (`cfg.showEnergyColor`), collision-flash scaling (`cfg.collisionFlashScale`), and dynamic size updates.
- **MODE-DEPENDENT:** Enabling these features introduces extra CPU-side preprocessing and additional buffer updates.
- **EXPECTED:** Increased overlay complexity can reduce frame-rate scaling sensitivity to pure CUDA kernel improvements.

Evidence:
- `app/main.cpp`, feature-conditional blocks (`showVelocityVectors`, `showEnergyColor`, `highlightCloseEncounters`, collision flash, comet tail, trail updates): confirms per-feature runtime work.
- `app/config.h`, visualization controls (`render`, `trails`, `showEnergyColor`, `showVelocityVectors`, `highlightCloseEncounters`, bloom/alpha/size parameters): confirms configuration interface.
- `vis/renderer.cpp`, `render` draw sequence and blend-state methods (`setAdditiveBlend`, `drawTrails`, `drawLines`, point draw): confirms presentation-side behavior.

### 13.5 Mode Implications

- **MODE-DEPENDENT:** `--present` emphasizes visualization path costs and presentation pacing behavior.
- **MODE-DEPENDENT:** `--realSolarSystem --collisions 0` can still include rendering overhead unless rendering is disabled.
- **MODE-DEPENDENT:** `--solarCollision --fragmentation` adds collision/fragment event handling plus optional event overlays on top of baseline render work.
- **DERIVED:** These mode distinctions align with Section 6.2 methodology classification (benchmark-suitable gravity mode vs visualization-heavy mode vs event-driven collision/fragment mode).

Evidence:
- `app/cli.cpp`, argument parsing for `--present`, `--realSolarSystem`, `--solarCollision`, `--fragmentation`, and `--render`.
- `app/main.cpp`, mode-preset branches and render/simulation feature toggles.

### 13.6 Limitations and Measurement Needs

- **MEASUREMENT-NEEDED:** Quantify split between simulation kernels, host-device copies, buffer updates, and presentation waits using graphics-aware `nsys` traces.
- **MEASUREMENT-NEEDED:** Validate whether frame pacing/vsync or synchronization dominates end-to-end frame time in presentation-focused modes.
- **MEASUREMENT-NEEDED:** Evaluate how overlay toggles change transfer/update load and frame stability.
- **CODE-VERIFY-NEEDED:** If future branches introduce CUDA-graphics interop, this section should be revised with explicit interop call-path evidence.
- **DERIVED:** Any comparison against compute-only performance sections must respect Section 6 timing boundaries and avoid mixing present-mode frame time with kernel step-time metrics.

## 14. Final Conclusions

### 14.1 Impact of Shared-Memory Tiling

- **IMPLEMENTED:** The gravity kernel `compute_accel_tiled` stages j-body state (`x`,`y`,`z`,`m`) in shared memory and reuses each tile across threads in the block.
- **DERIVED:** Relative to a naive global-only reference model, streamed global j-body traffic per interaction is reduced asymptotically by a factor on the order of $B$ (block size), as documented by the baseline-mock derivation.
- **MEASURED:** `plots/time_per_interaction_vs_N.png` shows a decreasing normalized interaction cost trend over sampled $N$, consistent with amortization effects in the implemented tiled path.
- **DERIVED:** The baseline-mock interpretation is a traffic-reduction upper-bound model, not a direct measured runtime-speedup claim.
- **MEASUREMENT-NEEDED:** Causal runtime improvement versus a naive kernel requires controlled tiled-vs-naive A/B execution under matched methodology.

### 14.2 Scalability Behavior of Gravity-Only Mode

- **IMPLEMENTED/DERIVED:** Gravity-only execution is dominated by dense all-pairs force evaluation with $O(N^2)$ interaction count per recomputation.
- **MEASURED:** `plots/time_vs_N.png` shows superlinear growth with $N$ over the sampled set ($512..16384$), qualitatively consistent with quadratic interaction scaling.
- **MEASURED/DERIVED:** `plots/time_per_interaction_vs_N.png` indicates normalized-cost amortization from small to larger $N$ regimes in the same sampled set.
- **DERIVED:** The three-regime interpretation remains: small-$N$ overhead sensitivity, mid-$N$ overhead amortization, and large-$N$ dominance of interaction workload and resource constraints.
- **DERIVED:** Constant-factor optimizations improve efficiency but do not alter the asymptotic $O(N^2)$ growth law.

### 14.3 Overhead Characteristics of Collision/Fragment Modes

- **IMPLEMENTED/DERIVED:** Collision pair detection in current kernels remains $O(N^2)$ worst-case traversal.
- **IMPLEMENTED:** Fragmentation and merge workflows introduce event-driven, non-Hamiltonian processing beyond gravity integration.
- **MODE-DEPENDENT:** Host-assisted merge/fragment branches can introduce Device→Host→Device transfers and additional synchronization points.
- **DERIVED:** These effects change scaling shape relative to gravity-only mode and can increase runtime variability.
- **EXPECTED:** Event-driven collision/fragment bursts can produce step-to-step runtime instability even under fixed global configuration.

### 14.4 Roofline-Oriented Interpretation

- **DERIVED:** Shared-memory tiling reduces streamed global bytes per interaction for j-body data.
- **EXPECTED:** Reduced streamed traffic can shift limiting behavior away from strict DRAM-bandwidth dominance toward arithmetic/SFU/scheduling sensitivity.
- **MEASURED:** Current summary plots provide qualitative throughput and correlation context but do not isolate kernel-level causality.
- **MEASUREMENT-NEEDED:** Definitive memory-bound vs compute-bound classification requires kernel-isolated Nsight Compute counters and hardware peak ceilings captured in reproducible artifacts.
- **DERIVED:** Based on available evidence, the tiled gravity kernel does not exhibit clear DRAM-saturation behavior across sampled points; however, final roofline placement requires controlled, kernel-filtered profiling.

### 14.5 Overall System-Level Assessment

- **IMPLEMENTED:** The system applies coherent constant-factor efficiency mechanisms through shared-memory tiling and SoA-oriented data organization.
- **DERIVED:** Asymptotic computational complexity of direct gravity remains quadratic.
- **MODE-DEPENDENT:** Visualization and collision/fragment modes introduce additional non-gravity overhead paths that must be separated from compute-only interpretation.
- **IMPLEMENTED:** The codebase is architecturally consistent with staged profiling and reproducibility workflows defined in the methodology sections.
- **MEASUREMENT-NEEDED:** Priority next steps are controlled naive-vs-tiled A/B experiments, block-size sweep validation, full roofline placement with kernel-filtered counters, and long-horizon stability sweeps.

In conclusion, the current implementation establishes a technically sound and profiler-ready foundation: it demonstrates defensible constant-factor optimization structure, preserves a clear interpretation of asymptotic limits, and frames all unresolved performance classifications as measurement-dependent rather than assumed.

## 15. Execution Commands and NCU Collection Workflow

### 15.1 Runtime Commands Used in This Report

- **IMPLEMENTED:** Gravity-focused mode:
```bash
./build/nbody --realSolarSystem 1 --collisions 0
```
- **IMPLEMENTED:** Visualization-focused mode:
```bash
./build/nbody --present 1
```
- **IMPLEMENTED:** Collision/fragmentation event-driven mode:
```bash
./build/nbody --solarCollision 1 --fragmentation 1
```

### 15.2 Nsight Compute Profiling Command (Full Set)

- **IMPLEMENTED:** NCU collection is executed with the following loop over multiple $N$ values:
```bash
mkdir -p perf_ncu_full

for N in 512 1024 2048 4096 8192 16384; do
  echo "===== Profiling N=$N ====="
  sudo $(which ncu) --set full \
    --launch-count 1 \
    -o perf_ncu_full/profile_N${N} \
    ./build/nbody --solarCollision 1 --fragmentation 1 --steps 2 --N $N
done
```

### 15.3 Data and Reporting Update Note

- **IMPLEMENTED:** The NCU summary CSV now includes `N=16384`.
- **IMPLEMENTED:** The Section 5 snapshot table is synchronized with the updated `ncu_metrics_summary.csv`.
- **IMPLEMENTED:** Plot artifacts were regenerated from the updated CSV using `plot.py`, and now include the `N=16384` point (for example `metrics_bar_N16384.png`).
- **MEASUREMENT-NEEDED:** Final causal interpretation still requires stable kernel filtering and mode-tagged profiling separation as defined in Section 6.2.

