#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

// Define M_PI if not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Simulation parameters
// ============================================================================
#define G           6.67430e-11f
#define SOFTENING   1e9f
#define DT          3600.0f
#define NSTEPS      8760
#define BLOCK_SIZE  256
#define OUTPUT_INTERVAL 24

#define AU          1.496e11f
#define SOLAR_MASS  1.989e30f
#define EARTH_MASS  5.972e24f

// ============================================================================
// Data structures
// ============================================================================
typedef struct {
    float x, y, z;
    float vx, vy, vz;
    float mass;
    float _pad;
} BodyAoS;

typedef struct {
    float x, y, z;
} Accel;

// ============================================================================
// Helper functions
// ============================================================================
void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// ============================================================================
// KERNEL 1: NAIVE (baseline)
// ============================================================================
__global__ void compute_accel_naive(BodyAoS* bodies, Accel* acc, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    float4 my = make_float4(bodies[idx].x, bodies[idx].y, bodies[idx].z, bodies[idx].mass);

    for (int j = 0; j < N; ++j) {
        float4 other = make_float4(bodies[j].x, bodies[j].y, bodies[j].z, bodies[j].mass);
        float dx = other.x - my.x;
        float dy = other.y - my.y;
        float dz = other.z - my.z;
        float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;
        float force = G * other.w * invDist3;
        ax += dx * force;
        ay += dy * force;
        az += dz * force;
    }
    acc[idx].x = ax;
    acc[idx].y = ay;
    acc[idx].z = az;
}

// ============================================================================
// KERNEL 2: TILED (optimized)
// ============================================================================
__global__ void compute_accel_tiled(BodyAoS* bodies, Accel* acc, int N) {
    extern __shared__ float4 sharedPos[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float4 my;
    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    if (idx < N) {
        my = make_float4(bodies[idx].x, bodies[idx].y, bodies[idx].z, bodies[idx].mass);
    }

    int numTiles = (N + blockDim.x - 1) / blockDim.x;
    for (int tile = 0; tile < numTiles; ++tile) {
        int tileIdx = tile * blockDim.x + threadIdx.x;
        if (tileIdx < N) {
            sharedPos[threadIdx.x] = make_float4(
                bodies[tileIdx].x,
                bodies[tileIdx].y,
                bodies[tileIdx].z,
                bodies[tileIdx].mass
            );
        }
        else {
            sharedPos[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        __syncthreads();

        if (idx < N) {
#pragma unroll 8
            for (int i = 0; i < blockDim.x; ++i) {
                float4 other = sharedPos[i];
                float dx = other.x - my.x;
                float dy = other.y - my.y;
                float dz = other.z - my.z;
                float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;
                float force = G * other.w * invDist3;
                ax += dx * force;
                ay += dy * force;
                az += dz * force;
            }
        }
        __syncthreads();
    }

    if (idx < N) {
        acc[idx].x = ax;
        acc[idx].y = ay;
        acc[idx].z = az;
    }
}

// ============================================================================
// Integration kernels
// ============================================================================
__global__ void update_positions_vel_firsthalf(BodyAoS* bodies, Accel* acc, int N, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    bodies[idx].vx += 0.5f * acc[idx].x * dt;
    bodies[idx].vy += 0.5f * acc[idx].y * dt;
    bodies[idx].vz += 0.5f * acc[idx].z * dt;

    bodies[idx].x += bodies[idx].vx * dt;
    bodies[idx].y += bodies[idx].vy * dt;
    bodies[idx].z += bodies[idx].vz * dt;
}

__global__ void update_velocity_secondhalf(BodyAoS* bodies, Accel* acc, int N, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    bodies[idx].vx += 0.5f * acc[idx].x * dt;
    bodies[idx].vy += 0.5f * acc[idx].y * dt;
    bodies[idx].vz += 0.5f * acc[idx].z * dt;
}

// ============================================================================
// Initialize solar system with chaotic comet
// ============================================================================
void init_solar_system_with_comet(BodyAoS* h_bodies, int N) {
    printf("Initializing solar system with chaotic comet (%d bodies)...\n", N);

    // Central star
    h_bodies[0].x = 0.0f;
    h_bodies[0].y = 0.0f;
    h_bodies[0].z = 0.0f;
    h_bodies[0].vx = 0.0f;
    h_bodies[0].vy = 0.0f;
    h_bodies[0].vz = 0.0f;
    h_bodies[0].mass = SOLAR_MASS;

    srand(42);

    // Major planets (1-8)
    for (int i = 1; i <= 8 && i < N; ++i) {
        float semi_major_axis = (0.4f + 0.6f * i) * AU;
        float eccentricity = 0.01f + (rand() / (float)RAND_MAX) * 0.05f;
        float inclination = (rand() / (float)RAND_MAX) * 0.05f;
        float true_anomaly = (rand() / (float)RAND_MAX) * 2.0f * M_PI;

        float r = semi_major_axis * (1.0f - eccentricity * eccentricity) /
            (1.0f + eccentricity * cosf(true_anomaly));

        float x_orbit = r * cosf(true_anomaly);
        float y_orbit = r * sinf(true_anomaly);

        h_bodies[i].x = x_orbit;
        h_bodies[i].y = y_orbit * cosf(inclination);
        h_bodies[i].z = y_orbit * sinf(inclination);

        h_bodies[i].mass = EARTH_MASS * (300.0f / powf(2.0f, i - 1));

        float v_mag = sqrtf(G * SOLAR_MASS / r);
        float vx_orbit = -v_mag * sinf(true_anomaly);
        float vy_orbit = v_mag * cosf(true_anomaly);

        h_bodies[i].vx = vx_orbit;
        h_bodies[i].vy = vy_orbit * cosf(inclination);
        h_bodies[i].vz = vy_orbit * sinf(inclination);
    }

    // CHAOTIC COMET (body 9)
    if (N > 9) {
        float perihelion = 0.3f * AU;
        float aphelion = 10.0f * AU;
        float ecc = (aphelion - perihelion) / (aphelion + perihelion);
        float semi_major = (perihelion + aphelion) / 2.0f;

        float true_anomaly = 0.1f;
        float r = semi_major * (1.0f - ecc * ecc) / (1.0f + ecc * cosf(true_anomaly));

        h_bodies[9].x = r * cosf(true_anomaly);
        h_bodies[9].y = r * sinf(true_anomaly);
        h_bodies[9].z = 0.5f * AU;
        h_bodies[9].mass = EARTH_MASS * 0.0001f;

        float v_mag = sqrtf(G * SOLAR_MASS * (2.0f / r - 1.0f / semi_major));
        h_bodies[9].vx = -v_mag * sinf(true_anomaly);
        h_bodies[9].vy = v_mag * cosf(true_anomaly);
        h_bodies[9].vz = v_mag * 0.3f;

        printf("  → Chaotic comet: e=%.3f, perihelion=%.2f AU\n", ecc, perihelion / AU);
    }

    // Asteroids
    for (int i = 10; i < N; ++i) {
        float semi_major_axis = 0.4f * AU + (rand() / (float)RAND_MAX) * 8.0f * AU;
        float eccentricity = (rand() / (float)RAND_MAX) * 0.3f;
        float inclination = (rand() / (float)RAND_MAX) * 0.15f;
        float true_anomaly = (rand() / (float)RAND_MAX) * 2.0f * M_PI;

        float r = semi_major_axis * (1.0f - eccentricity * eccentricity) /
            (1.0f + eccentricity * cosf(true_anomaly));

        float x_orbit = r * cosf(true_anomaly);
        float y_orbit = r * sinf(true_anomaly);

        h_bodies[i].x = x_orbit;
        h_bodies[i].y = y_orbit * cosf(inclination);
        h_bodies[i].z = y_orbit * sinf(inclination);
        h_bodies[i].mass = EARTH_MASS * (0.0001f + (rand() / (float)RAND_MAX) * 0.01f);

        float v_mag = sqrtf(G * SOLAR_MASS / r);
        float vx_orbit = -v_mag * sinf(true_anomaly);
        float vy_orbit = v_mag * cosf(true_anomaly);

        h_bodies[i].vx = vx_orbit;
        h_bodies[i].vy = vy_orbit * cosf(inclination);
        h_bodies[i].vz = vy_orbit * sinf(inclination);
    }

    printf("Init: 1 star + 8 planets + 1 comet + %d asteroids\n", N - 10);
}

// ============================================================================
// Benchmark kernel
// ============================================================================
typedef struct {
    const char* name;
    float avg_time_ms;
    double interactions_per_sec;
    double gflops;
} BenchmarkResult;

BenchmarkResult benchmark_kernel(int use_tiled, int N, int iterations) {
    BenchmarkResult result;
    result.name = use_tiled ? "Tiled" : "Naive";

    BodyAoS* h_bodies = (BodyAoS*)malloc(N * sizeof(BodyAoS));
    init_solar_system_with_comet(h_bodies, N);

    BodyAoS* d_bodies;
    Accel* d_acc;

    checkCuda(cudaMalloc(&d_bodies, N * sizeof(BodyAoS)), "malloc bodies");
    checkCuda(cudaMalloc(&d_acc, N * sizeof(Accel)), "malloc accel");
    checkCuda(cudaMemcpy(d_bodies, h_bodies, N * sizeof(BodyAoS), cudaMemcpyHostToDevice), "memcpy");

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        if (use_tiled) {
            size_t shared_mem = BLOCK_SIZE * sizeof(float4);
            compute_accel_tiled << <grid, block, shared_mem >> > (d_bodies, d_acc, N);
        }
        else {
            compute_accel_naive << <grid, block >> > (d_bodies, d_acc, N);
        }
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        if (use_tiled) {
            size_t shared_mem = BLOCK_SIZE * sizeof(float4);
            compute_accel_tiled << <grid, block, shared_mem >> > (d_bodies, d_acc, N);
        }
        else {
            compute_accel_naive << <grid, block >> > (d_bodies, d_acc, N);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);

    result.avg_time_ms = total_time / iterations;

    long long interactions = (long long)N * (long long)N * (long long)iterations;
    result.interactions_per_sec = interactions / (total_time / 1000.0);
    result.gflops = (result.interactions_per_sec * 20.0) / 1e9;

    cudaFree(d_bodies);
    cudaFree(d_acc);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_bodies);

    return result;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    printf("========================================================================\n");
    printf("GPU-Accelerated N-Body Simulation - COMPREHENSIVE\n");
    printf("========================================================================\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Compute: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Shared mem/block: %zu KB\n\n", prop.sharedMemPerBlock / 1024);

    int N = 32768;
    int run_benchmarks = 1;

    if (argc > 1) {
        N = atoi(argv[1]);
        if (N < 2 || N > 65536) {
            fprintf(stderr, "N must be 2-65536\n");
            return EXIT_FAILURE;
        }
    }
    if (argc > 2) run_benchmarks = atoi(argv[2]);

#ifdef _WIN32
    system("if not exist output mkdir output");
#else
    system("mkdir -p output");
#endif

    // ========================================================================
    // BENCHMARKING
    // ========================================================================
    if (run_benchmarks) {
        printf("========================================================================\n");
        printf("KERNEL BENCHMARKING (Naive vs Tiled)\n");
        printf("========================================================================\n\n");

        FILE* bench_file = fopen("output/kernel_benchmark.csv", "w");
        fprintf(bench_file, "N,kernel,avg_time_ms,interactions_per_sec,gflops,speedup\n");

        int N_values[] = { 1024, 2048, 4096, 8192, 16384, 32768 };
        int num_N = 6;

        for (int i = 0; i < num_N; ++i) {
            int test_N = N_values[i];
            printf("N = %d:\n", test_N);

            BenchmarkResult naive = benchmark_kernel(0, test_N, 50);
            BenchmarkResult tiled = benchmark_kernel(1, test_N, 50);

            float speedup = naive.avg_time_ms / tiled.avg_time_ms;

            printf("  Naive: %.3f ms, %.2e inter/s, %.2f GFLOPS\n",
                naive.avg_time_ms, naive.interactions_per_sec, naive.gflops);
            printf("  Tiled: %.3f ms, %.2e inter/s, %.2f GFLOPS\n",
                tiled.avg_time_ms, tiled.interactions_per_sec, tiled.gflops);
            printf("  Speedup: %.2fx\n\n", speedup);

            fprintf(bench_file, "%d,Naive,%.6f,%.6e,%.6f,1.00\n",
                test_N, naive.avg_time_ms, naive.interactions_per_sec, naive.gflops);
            fprintf(bench_file, "%d,Tiled,%.6f,%.6e,%.6f,%.6f\n",
                test_N, tiled.avg_time_ms, tiled.interactions_per_sec, tiled.gflops, speedup);
        }

        fclose(bench_file);
        printf("Benchmark saved: output/kernel_benchmark.csv\n\n");
    }

    // ========================================================================
    // FULL SIMULATION
    // ========================================================================
    printf("========================================================================\n");
    printf("PHYSICS SIMULATION\n");
    printf("========================================================================\n\n");

    printf("Parameters:\n");
    printf("  N: %d\n", N);
    printf("  Time step: %.0f sec\n", DT);
    printf("  Steps: %d\n", NSTEPS);
    printf("  Duration: %.1f days\n\n", NSTEPS * DT / 86400.0f);

    BodyAoS* h_bodies = (BodyAoS*)malloc(N * sizeof(BodyAoS));
    init_solar_system_with_comet(h_bodies, N);

    BodyAoS* d_bodies;
    Accel* d_acc;

    checkCuda(cudaMalloc(&d_bodies, N * sizeof(BodyAoS)), "malloc");
    checkCuda(cudaMalloc(&d_acc, N * sizeof(Accel)), "malloc");
    checkCuda(cudaMemcpy(d_bodies, h_bodies, N * sizeof(BodyAoS), cudaMemcpyHostToDevice), "memcpy");

    FILE* pos_file = fopen("output/positions.csv", "w");
    FILE* energy_file = fopen("output/energy.csv", "w");
    FILE* perf_file = fopen("output/performance.csv", "w");
    FILE* comet_file = fopen("output/comet_track.csv", "w");

    fprintf(pos_file, "time,body_id,x,y,z,vx,vy,vz,mass\n");
    fprintf(energy_file, "time,total_energy,kinetic,potential\n");
    fprintf(perf_file, "step,kernel_time_ms,interactions_per_sec,gflops\n");
    fprintf(comet_file, "time,x,y,z,vx,vy,vz,speed,distance\n");

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    size_t shared_mem = BLOCK_SIZE * sizeof(float4);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initial acceleration
    compute_accel_tiled << <grid, block, shared_mem >> > (d_bodies, d_acc, N);
    cudaDeviceSynchronize();

    printf("Running...\n[");
    fflush(stdout);

    int progress_step = NSTEPS / 50;
    float total_kernel_time = 0.0f;

    for (int step = 0; step < NSTEPS; ++step) {
        update_positions_vel_firsthalf << <grid, block >> > (d_bodies, d_acc, N, DT);

        cudaEventRecord(start);
        compute_accel_tiled << <grid, block, shared_mem >> > (d_bodies, d_acc, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float kernel_time;
        cudaEventElapsedTime(&kernel_time, start, stop);
        total_kernel_time += kernel_time;

        update_velocity_secondhalf << <grid, block >> > (d_bodies, d_acc, N, DT);
        cudaDeviceSynchronize();

        if (step % OUTPUT_INTERVAL == 0) {
            checkCuda(cudaMemcpy(h_bodies, d_bodies, N * sizeof(BodyAoS), cudaMemcpyDeviceToHost), "memcpy");

            float time = step * DT;

            double ke = 0.0, pe = 0.0;
            for (int i = 0; i < N; ++i) {
                ke += 0.5 * h_bodies[i].mass * (h_bodies[i].vx * h_bodies[i].vx +
                    h_bodies[i].vy * h_bodies[i].vy +
                    h_bodies[i].vz * h_bodies[i].vz);
                for (int j = i + 1; j < N; ++j) {
                    float dx = h_bodies[j].x - h_bodies[i].x;
                    float dy = h_bodies[j].y - h_bodies[i].y;
                    float dz = h_bodies[j].z - h_bodies[i].z;
                    float dist = sqrtf(dx * dx + dy * dy + dz * dz + SOFTENING);
                    pe += (G * h_bodies[i].mass * h_bodies[j].mass) / dist;
                }
            }
            fprintf(energy_file, "%.1f,%.12e,%.12e,%.12e\n", time, ke - pe, ke, pe);

            for (int i = 0; i < N; ++i) {
                fprintf(pos_file, "%.1f,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                    time, i, h_bodies[i].x, h_bodies[i].y, h_bodies[i].z,
                    h_bodies[i].vx, h_bodies[i].vy, h_bodies[i].vz, h_bodies[i].mass);
            }

            if (N > 9) {
                float speed = sqrtf(h_bodies[9].vx * h_bodies[9].vx +
                    h_bodies[9].vy * h_bodies[9].vy +
                    h_bodies[9].vz * h_bodies[9].vz);
                float dist = sqrtf(h_bodies[9].x * h_bodies[9].x +
                    h_bodies[9].y * h_bodies[9].y +
                    h_bodies[9].z * h_bodies[9].z);
                fprintf(comet_file, "%.1f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                    time, h_bodies[9].x, h_bodies[9].y, h_bodies[9].z,
                    h_bodies[9].vx, h_bodies[9].vy, h_bodies[9].vz, speed, dist);
            }

            fflush(pos_file);
            fflush(energy_file);
            fflush(comet_file);
        }

        if (step % 100 == 0) {
            long long interactions = (long long)N * (long long)N;
            double inter_per_sec = interactions / (kernel_time / 1000.0);
            double gflops = (inter_per_sec * 20.0) / 1e9;
            fprintf(perf_file, "%d,%.3f,%.3e,%.3f\n", step, kernel_time, inter_per_sec, gflops);
        }

        if (step % progress_step == 0) {
            printf("=");
            fflush(stdout);
        }
    }

    printf("] Done!\n\n");

    fclose(pos_file);
    fclose(energy_file);
    fclose(perf_file);
    fclose(comet_file);

    float avg_time = total_kernel_time / NSTEPS;
    long long total_inter = (long long)N * (long long)N * (long long)NSTEPS;
    double inter_per_sec = total_inter / (total_kernel_time / 1000.0);
    double gflops = (inter_per_sec * 20.0) / 1e9;

    printf("Performance:\n");
    printf("  Avg kernel: %.3f ms\n", avg_time);
    printf("  Total: %.2f sec\n", total_kernel_time / 1000.0);
    printf("  Inter/sec: %.3e\n", inter_per_sec);
    printf("  GFLOPS: %.2f\n\n", gflops);

    printf("Output files created. Run visualize_nbody_advanced.py\n");

    cudaFree(d_bodies);
    cudaFree(d_acc);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_bodies);

    return 0;
}
