/*
 * High-Performance N-Body Simulation of Colliding Galaxies (CUDA)
 * Final Version – with naive vs tiled kernel comparison.
 * 
 * Compilation (headless):
 *   nvcc -o nbody_galaxy nbody_galaxy_v12.cu -O3 -arch=sm_86 --use_fast_math
 *
 * Compilation (with OpenGL visualisation):
 *   nvcc -o nbody_galaxy nbody_galaxy_v12.cu -O3 -arch=sm_86 --use_fast_math -lGL -lglfw -DVISUAL_MODE
 *
 * Usage:
 *   ./nbody_galaxy headless [N] [steps]        # run without graphics, save CSV
 *   ./nbody_galaxy visual   [N] [steps]        # run with OpenGL window (requires VISUAL_MODE)
 *   ./nbody_galaxy benchmark                   # run scalability benchmark (saves benchmark.csv)
 *   ./nbody_galaxy compare                      # compare naive vs tiled kernels (saves kernel_comparison.csv)
 */

#ifdef VISUAL_MODE
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#endif

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <iomanip>

// -------------------- Configuration --------------------
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#define SOFTENING 1e-4f          // ε² (softening length squared)
#define TIME_STEP 0.001f
#define ENERGY_INTERVAL 100       // steps between energy prints/saves

// -------------------- Structures --------------------
struct GalaxySim {
    float4* d_pos;          // positions + mass (device)
    float3* d_vel;          // velocities (device)
    float3* d_acc;          // accelerations (device)
    float*  d_totalU;       // total potential energy (device)
    float*  d_totalK;       // total kinetic energy (device)

#ifdef VISUAL_MODE
    GLuint vbo;
    struct cudaGraphicsResource* cuda_vbo_resource;
#endif

    int n_bodies;
};

// -------------------- Error Checking --------------------
#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__   \
                  << " - " << cudaGetErrorString(err) << std::endl;    \
        exit(EXIT_FAILURE);                                            \
    } } while(0)

#define CUDA_LAST_CHECK() CUDA_CHECK(cudaGetLastError())

// -------------------- Kernels --------------------

// Naive force calculation (no shared memory)
__global__ void computeForces_naive(const float4* __restrict__ p,
                                    float3* __restrict__ a,
                                    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float4 myPos = p[idx];
    float3 acc = {0.0f, 0.0f, 0.0f};

    for (int j = 0; j < n; ++j) {
        if (j == idx) continue;               // skip self
        float4 other = p[j];
        float3 r;
        r.x = other.x - myPos.x;
        r.y = other.y - myPos.y;
        r.z = other.z - myPos.z;

        float distSqr = r.x*r.x + r.y*r.y + r.z*r.z + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;
        float s = other.w * invDist3;          // G = 1

        acc.x += r.x * s;
        acc.y += r.y * s;
        acc.z += r.z * s;
    }
    a[idx] = acc;
}

// Tile-based force calculation (uses shared memory) – self-interaction skipped
__global__ void computeForces_tiled(const float4* __restrict__ p,
                                    float3* __restrict__ a,
                                    int n) {
    extern __shared__ float4 tile[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float4 myPos = p[idx];
    float3 acc = {0.0f, 0.0f, 0.0f};

    for (int tileIdx = 0; tileIdx < gridDim.x; ++tileIdx) {
        int loadIdx = tileIdx * blockDim.x + threadIdx.x;
        if (loadIdx < n)
            tile[threadIdx.x] = p[loadIdx];
        else
            tile[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f); // dummy

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < blockDim.x; ++i) {
            int otherGlobalIdx = tileIdx * blockDim.x + i;
            //if (otherGlobalIdx == idx) continue;          // skip self

            float4 other = tile[i];
            if (other.w == 0.0f) continue;                // skip dummy

            float3 r;
            r.x = other.x - myPos.x;
            r.y = other.y - myPos.y;
            r.z = other.z - myPos.z;

            float distSqr = r.x*r.x + r.y*r.y + r.z*r.z + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;
            float s = other.w * invDist3;     // G = 1

            acc.x += r.x * s;
            acc.y += r.y * s;
            acc.z += r.z * s;
        }
        __syncthreads();
    }
    a[idx] = acc;
}

// Velocity Verlet step 1: v(t+½Δt) and r(t+Δt)
__global__ void integrate_step1(float4* p, float3* v, const float3* a,
                                float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    v[i].x += 0.5f * a[i].x * dt;
    v[i].y += 0.5f * a[i].y * dt;
    v[i].z += 0.5f * a[i].z * dt;

    p[i].x += v[i].x * dt;
    p[i].y += v[i].y * dt;
    p[i].z += v[i].z * dt;
}

// Velocity Verlet step 2: v(t+Δt)
__global__ void integrate_step2(float3* v, const float3* a, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    v[i].x += 0.5f * a[i].x * dt;
    v[i].y += 0.5f * a[i].y * dt;
    v[i].z += 0.5f * a[i].z * dt;
}

// Compute total kinetic and potential energy (with block reduction)
__global__ void computeTotalEnergy(const float4* __restrict__ p,
                                   const float3* __restrict__ v,
                                   float* d_totalU,
                                   float* d_totalK,
                                   int n) {
    extern __shared__ float4 tile[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float4 myPos = p[idx];
    float3 myVel = v[idx];
    float myMass = myPos.w;

    float v2 = myVel.x*myVel.x + myVel.y*myVel.y + myVel.z*myVel.z;
    float kinetic = 0.5f * myMass * v2;

    float potential = 0.0f;
    for (int tileIdx = 0; tileIdx < gridDim.x; ++tileIdx) {
        int loadIdx = tileIdx * blockDim.x + threadIdx.x;
        if (loadIdx < n)
            tile[threadIdx.x] = p[loadIdx];
        else
            tile[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < blockDim.x; ++i) {
            int otherGlobalIdx = tileIdx * blockDim.x + i;
            if (otherGlobalIdx == idx) continue;          // skip self

            float4 other = tile[i];
            if (other.w == 0.0f) continue;

            float dx = other.x - myPos.x;
            float dy = other.y - myPos.y;
            float dz = other.z - myPos.z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            potential += other.w * invDist;
        }
        __syncthreads();
    }

    potential = -0.5f * myMass * potential;

    // Block reduction for potential energy
    float* shMem = (float*)tile;
    shMem[threadIdx.x] = potential;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            shMem[threadIdx.x] += shMem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(d_totalU, shMem[0]);

    // Block reduction for kinetic energy
    shMem[threadIdx.x] = kinetic;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            shMem[threadIdx.x] += shMem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(d_totalK, shMem[0]);
}

// Reset energy accumulators
__global__ void resetEnergy(float* d_totalU, float* d_totalK) {
    if (threadIdx.x == 0) {
        *d_totalU = 0.0f;
        *d_totalK = 0.0f;
    }
}

// -------------------- Host Helpers --------------------

// Initialize two colliding spiral galaxies (particles 0..n/2-1 = galaxy A, n/2..n-1 = galaxy B)
void initGalaxies(std::vector<float4>& pos_mass,
                  std::vector<float3>& vel,
                  int n_bodies) {
    std::default_random_engine gen(42);
    std::uniform_real_distribution<float> uniform(0.0, 1.0);

    pos_mass.resize(n_bodies);
    vel.resize(n_bodies);

    for (int i = 0; i < n_bodies; ++i) {
        bool second = (i >= n_bodies / 2);
        float cx = second ? 0.5f : -0.5f;
        float cy = 0.0f;
        float cz = 0.0f;
        float vx_base = second ? -0.2f : 0.2f;
        float vy_base = second ? -0.1f : 0.1f;

        float angle = uniform(gen) * 2.0f * M_PI;
        float radius = 0.1f + 0.4f * uniform(gen);

        pos_mass[i].x = cx + radius * cos(angle);
        pos_mass[i].y = cy + radius * sin(angle) * 0.8f;
        pos_mass[i].z = cz + (uniform(gen) - 0.5f) * 0.05f;
        pos_mass[i].w = 1.0f / n_bodies;   // mass

        float vel_mag = sqrtf(pos_mass[i].w / radius) * 0.5f;
        vel[i].x = vx_base - vel_mag * sin(angle);
        vel[i].y = vy_base + vel_mag * cos(angle);
        vel[i].z = 0.0f;
    }
}

// Compute and print total energy drift, and optionally save to CSV
void saveEnergy(GalaxySim& sim, int step, float dt, std::ofstream& energy_csv) {
    resetEnergy<<<1, 1>>>(sim.d_totalU, sim.d_totalK);
    CUDA_LAST_CHECK();

    int grid = (sim.n_bodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t shmem = BLOCK_SIZE * sizeof(float4);
    computeTotalEnergy<<<grid, BLOCK_SIZE, shmem>>>(
        sim.d_pos, sim.d_vel, sim.d_totalU, sim.d_totalK, sim.n_bodies);
    CUDA_LAST_CHECK();

    float h_totalU, h_totalK;
    CUDA_CHECK(cudaMemcpy(&h_totalU, sim.d_totalU, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_totalK, sim.d_totalK, sizeof(float), cudaMemcpyDeviceToHost));

    float totalE = h_totalU + h_totalK;
    static float initialE = 0.0f;
    if (step == 0) initialE = totalE;
    float drift = (totalE - initialE) / initialE;

    printf("Step %5d | K = %8.3e | U = %8.3e | E = %8.3e | drift = %6.4e\n",
           step, h_totalK, h_totalU, totalE, drift);

    if (energy_csv.is_open()) {
        energy_csv << step << "," << h_totalK << "," << h_totalU << ","
                   << totalE << "," << drift << "\n";
    }
}

// -------------------- Headless Simulation (CSV output) --------------------
void runHeadlessCSV(int n_bodies, int max_steps) {
    std::cout << "=== Headless CSV Generation (N=" << n_bodies << ", steps=" << max_steps << ") ===\n";

    GalaxySim sim;
    sim.n_bodies = n_bodies;
    size_t bytes_pos = n_bodies * sizeof(float4);
    size_t bytes_vec = n_bodies * sizeof(float3);

    std::vector<float4> h_pos;
    std::vector<float3> h_vel;
    initGalaxies(h_pos, h_vel, n_bodies);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&sim.d_pos, bytes_pos));
    CUDA_CHECK(cudaMalloc(&sim.d_vel, bytes_vec));
    CUDA_CHECK(cudaMalloc(&sim.d_acc, bytes_vec));
    CUDA_CHECK(cudaMalloc(&sim.d_totalU, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sim.d_totalK, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(sim.d_pos, h_pos.data(), bytes_pos, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.d_vel, h_vel.data(), bytes_vec, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(sim.d_acc, 0, bytes_vec));

    int grid = (n_bodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int shared_mem_size = BLOCK_SIZE * sizeof(float4);
    computeForces_tiled<<<grid, BLOCK_SIZE, shared_mem_size>>>(sim.d_pos, sim.d_acc, n_bodies);
    CUDA_LAST_CHECK();
    cudaDeviceSynchronize();

    // CSV files
    std::ofstream pos_csv("simulation.csv");
    pos_csv << "step,particle_id,x,y,z,mass,galaxy\n";   // added galaxy id (0 or 1)
    std::ofstream energy_csv("energy.csv");
    energy_csv << "step,kinetic,potential,total,drift\n";

    int step = 0;
    while (step < max_steps) {
        // Physics steps
        integrate_step1<<<grid, BLOCK_SIZE>>>(sim.d_pos, sim.d_vel, sim.d_acc,
                                              TIME_STEP, n_bodies);
        computeForces_tiled<<<grid, BLOCK_SIZE, shared_mem_size>>>(
            sim.d_pos, sim.d_acc, n_bodies);
        integrate_step2<<<grid, BLOCK_SIZE>>>(sim.d_vel, sim.d_acc, TIME_STEP, n_bodies);
        CUDA_LAST_CHECK();

        // Energy tracking (every ENERGY_INTERVAL steps)
        if (step % ENERGY_INTERVAL == 0)
            saveEnergy(sim, step, TIME_STEP, energy_csv);

        // Dump positions every 10 steps
        if (step % 10 == 0) {
            std::vector<float4> h_pos_dump(n_bodies);
            CUDA_CHECK(cudaMemcpy(h_pos_dump.data(), sim.d_pos, bytes_pos,
                                  cudaMemcpyDeviceToHost));
            for (int i = 0; i < n_bodies; ++i) {
                int galaxy = (i < n_bodies/2) ? 0 : 1;
                pos_csv << step << "," << i << ","
                        << h_pos_dump[i].x << "," << h_pos_dump[i].y << ","
                        << h_pos_dump[i].z << "," << h_pos_dump[i].w << ","
                        << galaxy << "\n";
            }
        }

        step++;
    }

    pos_csv.close();
    energy_csv.close();

    // Cleanup
    CUDA_CHECK(cudaFree(sim.d_pos));
    CUDA_CHECK(cudaFree(sim.d_vel));
    CUDA_CHECK(cudaFree(sim.d_acc));
    CUDA_CHECK(cudaFree(sim.d_totalU));
    CUDA_CHECK(cudaFree(sim.d_totalK));

    std::cout << "Simulation finished. Data saved to simulation.csv and energy.csv\n";
}

// -------------------- Visual Simulation (OpenGL) --------------------
#ifdef VISUAL_MODE
void runVisualSimulation(int n_bodies, int max_steps) {
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return;
    }
    GLFWwindow* window = glfwCreateWindow(1024, 1024,
                                          "N-Body Galaxy Collision (RTX 3060)",
                                          nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);

    GalaxySim sim;
    sim.n_bodies = n_bodies;
    size_t bytes_pos = n_bodies * sizeof(float4);
    size_t bytes_vec = n_bodies * sizeof(float3);

    std::vector<float4> h_pos;
    std::vector<float3> h_vel;
    initGalaxies(h_pos, h_vel, n_bodies);

    CUDA_CHECK(cudaMalloc(&sim.d_vel, bytes_vec));
    CUDA_CHECK(cudaMalloc(&sim.d_acc, bytes_vec));
    CUDA_CHECK(cudaMalloc(&sim.d_totalU, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sim.d_totalK, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(sim.d_vel, h_vel.data(), bytes_vec, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(sim.d_acc, 0, bytes_vec));

    // OpenGL interop
    glGenBuffers(1, &sim.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, sim.vbo);
    glBufferData(GL_ARRAY_BUFFER, bytes_pos, h_pos.data(), GL_DYNAMIC_DRAW);
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&sim.cuda_vbo_resource,
                                            sim.vbo,
                                            cudaGraphicsMapFlagsNone));

    int grid = (n_bodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int shared_mem_size = BLOCK_SIZE * sizeof(float4);

    // Optional CSV output (uncomment if you want CSV even in visual mode)
    // std::ofstream pos_csv("simulation.csv");
    // std::ofstream energy_csv("energy.csv");

    int step = 0;
    double lastTime = glfwGetTime();
    int frames = 0;

    while (!glfwWindowShouldClose(window) && step < max_steps) {
        // Map OpenGL buffer
        CUDA_CHECK(cudaGraphicsMapResources(1, &sim.cuda_vbo_resource, 0));
        size_t num_bytes;
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&sim.d_pos,
                                                        &num_bytes,
                                                        sim.cuda_vbo_resource));

        // Physics
        integrate_step1<<<grid, BLOCK_SIZE>>>(sim.d_pos, sim.d_vel, sim.d_acc,
                                              TIME_STEP, n_bodies);
        computeForces_tiled<<<grid, BLOCK_SIZE, shared_mem_size>>>(
            sim.d_pos, sim.d_acc, n_bodies);
        integrate_step2<<<grid, BLOCK_SIZE>>>(sim.d_vel, sim.d_acc, TIME_STEP, n_bodies);
        CUDA_LAST_CHECK();

        // Energy (optional)
        // if (step % ENERGY_INTERVAL == 0) saveEnergy(sim, step, TIME_STEP, energy_csv);

        // Unmap
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &sim.cuda_vbo_resource, 0));

        // Render
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glBindBuffer(GL_ARRAY_BUFFER, sim.vbo);
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);
        glColor4f(0.6f, 0.8f, 1.0f, 0.4f);
        glPointSize(1.5f);
        glDrawArrays(GL_POINTS, 0, n_bodies);
        glDisableClientState(GL_VERTEX_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();

        frames++;
        if (glfwGetTime() - lastTime >= 1.0) {
            double gips = (double)n_bodies * n_bodies * frames / 1e9;
            char title[256];
            snprintf(title, sizeof(title),
                     "N-Body GPU | N=%d | FPS: %d | GIPS: %.2f",
                     n_bodies, frames, gips);
            glfwSetWindowTitle(window, title);
            frames = 0;
            lastTime = glfwGetTime();
        }
        step++;
    }

    // Cleanup
    CUDA_CHECK(cudaGraphicsUnregisterResource(sim.cuda_vbo_resource));
    glDeleteBuffers(1, &sim.vbo);
    CUDA_CHECK(cudaFree(sim.d_vel));
    CUDA_CHECK(cudaFree(sim.d_acc));
    CUDA_CHECK(cudaFree(sim.d_totalU));
    CUDA_CHECK(cudaFree(sim.d_totalK));
    glfwTerminate();

    std::cout << "Visual simulation finished after " << step << " steps.\n";
}
#endif

// -------------------- Headless Benchmark (Tiled only) --------------------
void runBenchmark() {
    const int n_values[] = { 1024, 2048, 4096, 8192, 16384, 32768 };
    const int n_tests = sizeof(n_values) / sizeof(n_values[0]);
    const int STEPS = 100;

    std::cout << "=== N-Body Scalability Benchmark (Tiled kernel) ===\n";
    std::cout << "BLOCK_SIZE = " << BLOCK_SIZE << ", STEPS = " << STEPS << "\n\n";

    std::ofstream bench_csv("benchmark.csv");
    bench_csv << "N,grid,steps,gips,avg_ms_per_step\n";

    for (int test = 0; test < n_tests; ++test) {
        int n = n_values[test];
        int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        float4* d_pos;
        float3* d_vel;
        float3* d_acc;

        size_t bytes_pos = n * sizeof(float4);
        size_t bytes_vec = n * sizeof(float3);

        CUDA_CHECK(cudaMalloc(&d_pos, bytes_pos));
        CUDA_CHECK(cudaMalloc(&d_vel, bytes_vec));
        CUDA_CHECK(cudaMalloc(&d_acc, bytes_vec));

        std::vector<float4> h_pos;
        std::vector<float3> h_vel;
        initGalaxies(h_pos, h_vel, n);
        CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(), bytes_pos, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vel, h_vel.data(), bytes_vec, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_acc, 0, bytes_vec));

        // Warm‑up
        for (int i = 0; i < 10; ++i) {
            integrate_step1<<<grid, BLOCK_SIZE>>>(d_pos, d_vel, d_acc, TIME_STEP, n);
            computeForces_tiled<<<grid, BLOCK_SIZE, BLOCK_SIZE*sizeof(float4)>>>(
                d_pos, d_acc, n);
            integrate_step2<<<grid, BLOCK_SIZE>>>(d_vel, d_acc, TIME_STEP, n);
        }
        CUDA_LAST_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed steps
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < STEPS; ++i) {
            integrate_step1<<<grid, BLOCK_SIZE>>>(d_pos, d_vel, d_acc, TIME_STEP, n);
            computeForces_tiled<<<grid, BLOCK_SIZE, BLOCK_SIZE*sizeof(float4)>>>(
                d_pos, d_acc, n);
            integrate_step2<<<grid, BLOCK_SIZE>>>(d_vel, d_acc, TIME_STEP, n);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        float avg_ms = elapsed_ms / STEPS;
        double gips = (double)n * n / (avg_ms * 1e6);

        printf("%8d   %4d   %6.2f   %8.3f\n", n, grid, gips, avg_ms);
        bench_csv << n << "," << grid << "," << STEPS << "," << gips << "," << avg_ms << "\n";

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_pos));
        CUDA_CHECK(cudaFree(d_vel));
        CUDA_CHECK(cudaFree(d_acc));
    }
    bench_csv.close();
    std::cout << "Benchmark results saved to benchmark.csv\n";
}

// -------------------- Kernel Comparison (Naive vs Tiled) --------------------
void runKernelComparison() {
    const int n_values[] = { 1024, 2048, 4096, 8192, 16384, 32768 };
    const int n_tests = sizeof(n_values) / sizeof(n_values[0]);
    const int STEPS = 100;

    std::cout << "=== Naive vs Tiled Kernel Comparison ===\n";
    std::cout << "BLOCK_SIZE = " << BLOCK_SIZE << ", STEPS = " << STEPS << "\n\n";

    std::ofstream comp_csv("kernel_comparison.csv");
    comp_csv << "N,kernel,avg_time_ms,gips\n";

    for (int test = 0; test < n_tests; ++test) {
        int n = n_values[test];
        int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Prepare data once
        float4* d_pos;
        float3* d_vel;
        float3* d_acc_naive;
        float3* d_acc_tiled;

        size_t bytes_pos = n * sizeof(float4);
        size_t bytes_vec = n * sizeof(float3);

        CUDA_CHECK(cudaMalloc(&d_pos, bytes_pos));
        CUDA_CHECK(cudaMalloc(&d_vel, bytes_vec));
        CUDA_CHECK(cudaMalloc(&d_acc_naive, bytes_vec));
        CUDA_CHECK(cudaMalloc(&d_acc_tiled, bytes_vec));

        std::vector<float4> h_pos;
        std::vector<float3> h_vel;
        initGalaxies(h_pos, h_vel, n);
        CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(), bytes_pos, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vel, h_vel.data(), bytes_vec, cudaMemcpyHostToDevice));

        // ----- Naive kernel -----
        CUDA_CHECK(cudaMemset(d_acc_naive, 0, bytes_vec));
        // Warm‑up
        for (int i = 0; i < 10; ++i) {
            computeForces_naive<<<grid, BLOCK_SIZE>>>(d_pos, d_acc_naive, n);
        }
        CUDA_LAST_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < STEPS; ++i) {
            computeForces_naive<<<grid, BLOCK_SIZE>>>(d_pos, d_acc_naive, n);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_naive_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_naive_ms, start, stop));
        float avg_naive_ms = elapsed_naive_ms / STEPS;
        double gips_naive = (double)n * n / (avg_naive_ms * 1e6);
        printf("N=%6d Naive: %8.3f ms, %6.2f GIPS\n", n, avg_naive_ms, gips_naive);
        comp_csv << n << ",Naive," << avg_naive_ms << "," << gips_naive << "\n";

        // ----- Tiled kernel -----
        CUDA_CHECK(cudaMemset(d_acc_tiled, 0, bytes_vec));
        // Warm‑up
        for (int i = 0; i < 10; ++i) {
            computeForces_tiled<<<grid, BLOCK_SIZE, BLOCK_SIZE*sizeof(float4)>>>(
                d_pos, d_acc_tiled, n);
        }
        CUDA_LAST_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < STEPS; ++i) {
            computeForces_tiled<<<grid, BLOCK_SIZE, BLOCK_SIZE*sizeof(float4)>>>(
                d_pos, d_acc_tiled, n);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_tiled_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_tiled_ms, start, stop));
        float avg_tiled_ms = elapsed_tiled_ms / STEPS;
        double gips_tiled = (double)n * n / (avg_tiled_ms * 1e6);
        printf("N=%6d Tiled: %8.3f ms, %6.2f GIPS (speedup = %.2fx)\n",
               n, avg_tiled_ms, gips_tiled, avg_naive_ms / avg_tiled_ms);
        comp_csv << n << ",Tiled," << avg_tiled_ms << "," << gips_tiled << "\n";

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_pos));
        CUDA_CHECK(cudaFree(d_vel));
        CUDA_CHECK(cudaFree(d_acc_naive));
        CUDA_CHECK(cudaFree(d_acc_tiled));
    }
    comp_csv.close();
    std::cout << "\nKernel comparison saved to kernel_comparison.csv\n";
}

// -------------------- Main --------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [visual|headless|benchmark|compare] [N] [steps]\n";
        return 1;
    }

    std::string mode = argv[1];
    int n = (argc >= 3) ? std::atoi(argv[2]) : 32768;
    int steps = (argc >= 4) ? std::atoi(argv[3]) : 500;

    if (mode == "visual") {
#ifdef VISUAL_MODE
        runVisualSimulation(n, steps);
#else
        std::cerr << "Visual mode not compiled (define VISUAL_MODE and link with -lGL -lglfw).\n";
#endif
    } else if (mode == "headless") {
        runHeadlessCSV(n, steps);
    } else if (mode == "benchmark") {
        runBenchmark();
    } else if (mode == "compare") {
        runKernelComparison();
    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        return 1;
    }

    return 0;
}
