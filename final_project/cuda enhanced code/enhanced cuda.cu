#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

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
#define COLLISION_THRESHOLD 1e9f  // 1 million km

// ============================================================================
// Data structures
// ============================================================================

// Array of Structures (AoS)
typedef struct {
    float x, y, z;
    float vx, vy, vz;
    float mass;
    int active;  // 0 = merged/destroyed, 1 = active
} BodyAoS;

// Structure of Arrays (SoA)
typedef struct {
    float* x;
    float* y;
    float* z;
    float* vx;
    float* vy;
    float* vz;
    float* mass;
    int* active;
} BodiesSoA;

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
// KERNEL 1: NAIVE AoS (baseline)
// ============================================================================
__global__ void compute_accel_naive_aos(BodyAoS* bodies, Accel* acc, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N || !bodies[idx].active) return;

    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    float4 my = make_float4(bodies[idx].x, bodies[idx].y, bodies[idx].z, bodies[idx].mass);

    for (int j = 0; j < N; ++j) {
        if (!bodies[j].active) continue;
        
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
// KERNEL 2: TILED AoS (optimized)
// ============================================================================
__global__ void compute_accel_tiled_aos(BodyAoS* bodies, Accel* acc, int N) {
    extern __shared__ float4 sharedPos[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float4 my;
    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    if (idx < N && bodies[idx].active) {
        my = make_float4(bodies[idx].x, bodies[idx].y, bodies[idx].z, bodies[idx].mass);
    }

    int numTiles = (N + blockDim.x - 1) / blockDim.x;
    for (int tile = 0; tile < numTiles; ++tile) {
        int tileIdx = tile * blockDim.x + threadIdx.x;
        if (tileIdx < N && bodies[tileIdx].active) {
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

        if (idx < N && bodies[idx].active) {
#pragma unroll 8
            for (int i = 0; i < blockDim.x; ++i) {
                float4 other = sharedPos[i];
                if (other.w == 0.0f) continue;
                
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

    if (idx < N && bodies[idx].active) {
        acc[idx].x = ax;
        acc[idx].y = ay;
        acc[idx].z = az;
    }
}

// ============================================================================
// KERNEL 3: NAIVE SoA (for comparison)
// ============================================================================
__global__ void compute_accel_naive_soa(
    float* x, float* y, float* z, float* mass, int* active,
    Accel* acc, int N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N || !active[idx]) return;

    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    float my_x = x[idx];
    float my_y = y[idx];
    float my_z = z[idx];

    for (int j = 0; j < N; ++j) {
        if (!active[j]) continue;
        
        float dx = x[j] - my_x;
        float dy = y[j] - my_y;
        float dz = z[j] - my_z;
        float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;
        float force = G * mass[j] * invDist3;
        ax += dx * force;
        ay += dy * force;
        az += dz * force;
    }
    acc[idx].x = ax;
    acc[idx].y = ay;
    acc[idx].z = az;
}

// ============================================================================
// KERNEL 4: TILED SoA (optimized)
// ============================================================================
__global__ void compute_accel_tiled_soa(
    float* x, float* y, float* z, float* mass, int* active,
    Accel* acc, int N)
{
    extern __shared__ float4 sharedPos[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float my_x, my_y, my_z;
    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    if (idx < N && active[idx]) {
        my_x = x[idx];
        my_y = y[idx];
        my_z = z[idx];
    }

    int numTiles = (N + blockDim.x - 1) / blockDim.x;
    for (int tile = 0; tile < numTiles; ++tile) {
        int tileIdx = tile * blockDim.x + threadIdx.x;
        if (tileIdx < N && active[tileIdx]) {
            sharedPos[threadIdx.x] = make_float4(
                x[tileIdx], y[tileIdx], z[tileIdx], mass[tileIdx]
            );
        }
        else {
            sharedPos[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        __syncthreads();

        if (idx < N && active[idx]) {
#pragma unroll 8
            for (int i = 0; i < blockDim.x; ++i) {
                float4 other = sharedPos[i];
                if (other.w == 0.0f) continue;
                
                float dx = other.x - my_x;
                float dy = other.y - my_y;
                float dz = other.z - my_z;
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

    if (idx < N && active[idx]) {
        acc[idx].x = ax;
        acc[idx].y = ay;
        acc[idx].z = az;
    }
}

// ============================================================================
// COLLISION DETECTION KERNEL
// ============================================================================
__global__ void detect_collisions(BodyAoS* bodies, int* collision_pairs, int* num_collisions, int N, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N || !bodies[idx].active) return;

    for (int j = idx + 1; j < N; ++j) {
        if (!bodies[j].active) continue;
        
        float dx = bodies[j].x - bodies[idx].x;
        float dy = bodies[j].y - bodies[idx].y;
        float dz = bodies[j].z - bodies[idx].z;
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);

        if (dist < threshold) {
            int col_idx = atomicAdd(num_collisions, 1);
            if (col_idx < 1000) {  // Max 1000 collisions per step
                collision_pairs[col_idx * 2] = idx;
                collision_pairs[col_idx * 2 + 1] = j;
            }
        }
    }
}

// ============================================================================
// MERGE BODIES KERNEL (Inelastic collision)
// ============================================================================
__global__ void merge_bodies(BodyAoS* bodies, int* collision_pairs, int num_collisions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_collisions) return;

    int i = collision_pairs[idx * 2];
    int j = collision_pairs[idx * 2 + 1];

    if (!bodies[i].active || !bodies[j].active) return;

    // Conservation of momentum
    float total_mass = bodies[i].mass + bodies[j].mass;
    float new_vx = (bodies[i].mass * bodies[i].vx + bodies[j].mass * bodies[j].vx) / total_mass;
    float new_vy = (bodies[i].mass * bodies[i].vy + bodies[j].mass * bodies[j].vy) / total_mass;
    float new_vz = (bodies[i].mass * bodies[i].vz + bodies[j].mass * bodies[j].vz) / total_mass;

    // Center of mass position
    float new_x = (bodies[i].mass * bodies[i].x + bodies[j].mass * bodies[j].x) / total_mass;
    float new_y = (bodies[i].mass * bodies[i].y + bodies[j].mass * bodies[j].y) / total_mass;
    float new_z = (bodies[i].mass * bodies[i].z + bodies[j].mass * bodies[j].z) / total_mass;

    // Merge into larger body
    if (bodies[i].mass >= bodies[j].mass) {
        bodies[i].x = new_x;
        bodies[i].y = new_y;
        bodies[i].z = new_z;
        bodies[i].vx = new_vx;
        bodies[i].vy = new_vy;
        bodies[i].vz = new_vz;
        bodies[i].mass = total_mass;
        bodies[j].active = 0;  // Deactivate smaller body
    } else {
        bodies[j].x = new_x;
        bodies[j].y = new_y;
        bodies[j].z = new_z;
        bodies[j].vx = new_vx;
        bodies[j].vy = new_vy;
        bodies[j].vz = new_vz;
        bodies[j].mass = total_mass;
        bodies[i].active = 0;  // Deactivate smaller body
    }
}

// ============================================================================
// Integration kernels
// ============================================================================
__global__ void update_positions_vel_firsthalf(BodyAoS* bodies, Accel* acc, int N, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N || !bodies[idx].active) return;

    bodies[idx].vx += 0.5f * acc[idx].x * dt;
    bodies[idx].vy += 0.5f * acc[idx].y * dt;
    bodies[idx].vz += 0.5f * acc[idx].z * dt;

    bodies[idx].x += bodies[idx].vx * dt;
    bodies[idx].y += bodies[idx].vy * dt;
    bodies[idx].z += bodies[idx].vz * dt;
}

__global__ void update_velocity_secondhalf(BodyAoS* bodies, Accel* acc, int N, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N || !bodies[idx].active) return;

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
    h_bodies[0].active = 1;

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
        h_bodies[i].active = 1;

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
        h_bodies[9].active = 1;

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
        h_bodies[i].active = 1;

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

BenchmarkResult benchmark_kernel(int kernel_type, int N, int iterations) {
    // kernel_type: 0=Naive AoS, 1=Tiled AoS, 2=Naive SoA, 3=Tiled SoA
    
    BenchmarkResult result;
    const char* names[] = {"Naive_AoS", "Tiled_AoS", "Naive_SoA", "Tiled_SoA"};
    result.name = names[kernel_type];

    BodyAoS* h_bodies = (BodyAoS*)malloc(N * sizeof(BodyAoS));
    init_solar_system_with_comet(h_bodies, N);

    void* d_data;
    Accel* d_acc;

    if (kernel_type < 2) {
        // AoS
        checkCuda(cudaMalloc(&d_data, N * sizeof(BodyAoS)), "malloc");
        checkCuda(cudaMemcpy(d_data, h_bodies, N * sizeof(BodyAoS), cudaMemcpyHostToDevice), "memcpy");
    } else {
        // SoA
        BodiesSoA* d_soa = (BodiesSoA*)malloc(sizeof(BodiesSoA));
        checkCuda(cudaMalloc(&d_soa->x, N * sizeof(float)), "malloc");
        checkCuda(cudaMalloc(&d_soa->y, N * sizeof(float)), "malloc");
        checkCuda(cudaMalloc(&d_soa->z, N * sizeof(float)), "malloc");
        checkCuda(cudaMalloc(&d_soa->vx, N * sizeof(float)), "malloc");
        checkCuda(cudaMalloc(&d_soa->vy, N * sizeof(float)), "malloc");
        checkCuda(cudaMalloc(&d_soa->vz, N * sizeof(float)), "malloc");
        checkCuda(cudaMalloc(&d_soa->mass, N * sizeof(float)), "malloc");
        checkCuda(cudaMalloc(&d_soa->active, N * sizeof(int)), "malloc");

        // Copy data
        float* temp_x = (float*)malloc(N * sizeof(float));
        float* temp_y = (float*)malloc(N * sizeof(float));
        float* temp_z = (float*)malloc(N * sizeof(float));
        float* temp_vx = (float*)malloc(N * sizeof(float));
        float* temp_vy = (float*)malloc(N * sizeof(float));
        float* temp_vz = (float*)malloc(N * sizeof(float));
        float* temp_mass = (float*)malloc(N * sizeof(float));
        int* temp_active = (int*)malloc(N * sizeof(int));

        for (int i = 0; i < N; i++) {
            temp_x[i] = h_bodies[i].x;
            temp_y[i] = h_bodies[i].y;
            temp_z[i] = h_bodies[i].z;
            temp_vx[i] = h_bodies[i].vx;
            temp_vy[i] = h_bodies[i].vy;
            temp_vz[i] = h_bodies[i].vz;
            temp_mass[i] = h_bodies[i].mass;
            temp_active[i] = h_bodies[i].active;
        }

        checkCuda(cudaMemcpy(d_soa->x, temp_x, N * sizeof(float), cudaMemcpyHostToDevice), "memcpy");
        checkCuda(cudaMemcpy(d_soa->y, temp_y, N * sizeof(float), cudaMemcpyHostToDevice), "memcpy");
        checkCuda(cudaMemcpy(d_soa->z, temp_z, N * sizeof(float), cudaMemcpyHostToDevice), "memcpy");
        checkCuda(cudaMemcpy(d_soa->vx, temp_vx, N * sizeof(float), cudaMemcpyHostToDevice), "memcpy");
        checkCuda(cudaMemcpy(d_soa->vy, temp_vy, N * sizeof(float), cudaMemcpyHostToDevice), "memcpy");
        checkCuda(cudaMemcpy(d_soa->vz, temp_vz, N * sizeof(float), cudaMemcpyHostToDevice), "memcpy");
        checkCuda(cudaMemcpy(d_soa->mass, temp_mass, N * sizeof(float), cudaMemcpyHostToDevice), "memcpy");
        checkCuda(cudaMemcpy(d_soa->active, temp_active, N * sizeof(int), cudaMemcpyHostToDevice), "memcpy");

        d_data = d_soa;

        free(temp_x); free(temp_y); free(temp_z);
        free(temp_vx); free(temp_vy); free(temp_vz);
        free(temp_mass); free(temp_active);
    }

    checkCuda(cudaMalloc(&d_acc, N * sizeof(Accel)), "malloc accel");

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        if (kernel_type == 0) {
            compute_accel_naive_aos<<<grid, block>>>((BodyAoS*)d_data, d_acc, N);
        } else if (kernel_type == 1) {
            size_t shared_mem = BLOCK_SIZE * sizeof(float4);
            compute_accel_tiled_aos<<<grid, block, shared_mem>>>((BodyAoS*)d_data, d_acc, N);
        } else if (kernel_type == 2) {
            BodiesSoA* soa = (BodiesSoA*)d_data;
            compute_accel_naive_soa<<<grid, block>>>(
                soa->x, soa->y, soa->z, soa->mass, soa->active, d_acc, N);
        } else {
            BodiesSoA* soa = (BodiesSoA*)d_data;
            size_t shared_mem = BLOCK_SIZE * sizeof(float4);
            compute_accel_tiled_soa<<<grid, block, shared_mem>>>(
                soa->x, soa->y, soa->z, soa->mass, soa->active, d_acc, N);
        }
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        if (kernel_type == 0) {
            compute_accel_naive_aos<<<grid, block>>>((BodyAoS*)d_data, d_acc, N);
        } else if (kernel_type == 1) {
            size_t shared_mem = BLOCK_SIZE * sizeof(float4);
            compute_accel_tiled_aos<<<grid, block, shared_mem>>>((BodyAoS*)d_data, d_acc, N);
        } else if (kernel_type == 2) {
            BodiesSoA* soa = (BodiesSoA*)d_data;
            compute_accel_naive_soa<<<grid, block>>>(
                soa->x, soa->y, soa->z, soa->mass, soa->active, d_acc, N);
        } else {
            BodiesSoA* soa = (BodiesSoA*)d_data;
            size_t shared_mem = BLOCK_SIZE * sizeof(float4);
            compute_accel_tiled_soa<<<grid, block, shared_mem>>>(
                soa->x, soa->y, soa->z, soa->mass, soa->active, d_acc, N);
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

    // Cleanup
    if (kernel_type < 2) {
        cudaFree(d_data);
    } else {
        BodiesSoA* soa = (BodiesSoA*)d_data;
        cudaFree(soa->x);
        cudaFree(soa->y);
        cudaFree(soa->z);
        cudaFree(soa->vx);
        cudaFree(soa->vy);
        cudaFree(soa->vz);
        cudaFree(soa->mass);
        cudaFree(soa->active);
        free(soa);
    }
    
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
    printf("GPU-Accelerated N-Body Simulation - CLEAN VERSION\n");
    printf("========================================================================\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Compute: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Shared mem/block: %zu KB\n\n", prop.sharedMemPerBlock / 1024);

    int N = 1024;
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
    // COMPREHENSIVE BENCHMARKING (AoS vs SoA)
    // ========================================================================
    if (run_benchmarks) {
        printf("========================================================================\n");
        printf("COMPREHENSIVE KERNEL BENCHMARKING (AoS vs SoA)\n");
        printf("========================================================================\n\n");

        FILE* bench_file = fopen("output/kernel_benchmark_comprehensive.csv", "w");
        fprintf(bench_file, "N,kernel,data_structure,avg_time_ms,interactions_per_sec,gflops,speedup_vs_naive_aos\n");

        int N_values[] = { 1024, 2048, 4096, 8192, 16384 };
        int num_N = 5;

        for (int i = 0; i < num_N; ++i) {
            int test_N = N_values[i];
            printf("N = %d:\n", test_N);

            BenchmarkResult naive_aos = benchmark_kernel(0, test_N, 50);
            BenchmarkResult tiled_aos = benchmark_kernel(1, test_N, 50);
            BenchmarkResult naive_soa = benchmark_kernel(2, test_N, 50);
            BenchmarkResult tiled_soa = benchmark_kernel(3, test_N, 50);

            printf("  Naive AoS: %.3f ms, %.2e inter/s, %.2f GFLOPS\n",
                naive_aos.avg_time_ms, naive_aos.interactions_per_sec, naive_aos.gflops);
            printf("  Tiled AoS: %.3f ms, %.2e inter/s, %.2f GFLOPS (%.2fx)\n",
                tiled_aos.avg_time_ms, tiled_aos.interactions_per_sec, tiled_aos.gflops,
                naive_aos.avg_time_ms / tiled_aos.avg_time_ms);
            printf("  Naive SoA: %.3f ms, %.2e inter/s, %.2f GFLOPS (%.2fx)\n",
                naive_soa.avg_time_ms, naive_soa.interactions_per_sec, naive_soa.gflops,
                naive_aos.avg_time_ms / naive_soa.avg_time_ms);
            printf("  Tiled SoA: %.3f ms, %.2e inter/s, %.2f GFLOPS (%.2fx)\n\n",
                tiled_soa.avg_time_ms, tiled_soa.interactions_per_sec, tiled_soa.gflops,
                naive_aos.avg_time_ms / tiled_soa.avg_time_ms);

            fprintf(bench_file, "%d,Naive,AoS,%.6f,%.6e,%.6f,1.00\n",
                test_N, naive_aos.avg_time_ms, naive_aos.interactions_per_sec, naive_aos.gflops);
            fprintf(bench_file, "%d,Tiled,AoS,%.6f,%.6e,%.6f,%.6f\n",
                test_N, tiled_aos.avg_time_ms, tiled_aos.interactions_per_sec, tiled_aos.gflops,
                naive_aos.avg_time_ms / tiled_aos.avg_time_ms);
            fprintf(bench_file, "%d,Naive,SoA,%.6f,%.6e,%.6f,%.6f\n",
                test_N, naive_soa.avg_time_ms, naive_soa.interactions_per_sec, naive_soa.gflops,
                naive_aos.avg_time_ms / naive_soa.avg_time_ms);
            fprintf(bench_file, "%d,Tiled,SoA,%.6f,%.6e,%.6f,%.6f\n",
                test_N, tiled_soa.avg_time_ms, tiled_soa.interactions_per_sec, tiled_soa.gflops,
                naive_aos.avg_time_ms / tiled_soa.avg_time_ms);
        }

        fclose(bench_file);
        printf("Benchmark saved: output/kernel_benchmark_comprehensive.csv\n\n");
    }

    // ========================================================================
    // FULL SIMULATION WITH COLLISION DETECTION
    // ========================================================================
    printf("========================================================================\n");
    printf("PHYSICS SIMULATION WITH COLLISION DETECTION\n");
    printf("========================================================================\n\n");

    printf("Parameters:\n");
    printf("  N: %d\n", N);
    printf("  Time step: %.0f sec\n", DT);
    printf("  Steps: %d\n", NSTEPS);
    printf("  Duration: %.1f days\n", NSTEPS * DT / 86400.0f);
    printf("  Collision threshold: %.2e m\n\n", COLLISION_THRESHOLD);

    BodyAoS* h_bodies = (BodyAoS*)malloc(N * sizeof(BodyAoS));
    init_solar_system_with_comet(h_bodies, N);

    BodyAoS* d_bodies;
    Accel* d_acc;
    int* d_collision_pairs;
    int* d_num_collisions;
    int* h_num_collisions = (int*)malloc(sizeof(int));

    checkCuda(cudaMalloc(&d_bodies, N * sizeof(BodyAoS)), "malloc");
    checkCuda(cudaMalloc(&d_acc, N * sizeof(Accel)), "malloc");
    checkCuda(cudaMalloc(&d_collision_pairs, 2000 * sizeof(int)), "malloc");
    checkCuda(cudaMalloc(&d_num_collisions, sizeof(int)), "malloc");
    checkCuda(cudaMemcpy(d_bodies, h_bodies, N * sizeof(BodyAoS), cudaMemcpyHostToDevice), "memcpy");

    FILE* pos_file = fopen("output/positions.csv", "w");
    FILE* energy_file = fopen("output/energy.csv", "w");
    FILE* perf_file = fopen("output/performance.csv", "w");
    FILE* comet_file = fopen("output/comet_track.csv", "w");
    FILE* collision_file = fopen("output/collisions.csv", "w");
    FILE* active_bodies_file = fopen("output/active_bodies.csv", "w");

    fprintf(pos_file, "time,body_id,x,y,z,vx,vy,vz,mass,active\n");
    fprintf(energy_file, "time,total_energy,kinetic,potential\n");
    fprintf(perf_file, "step,kernel_time_ms,interactions_per_sec,gflops\n");
    fprintf(comet_file, "time,x,y,z,vx,vy,vz,speed,distance\n");
    fprintf(collision_file, "time,body1_id,body2_id,distance,merged_mass\n");
    fprintf(active_bodies_file, "time,active_count\n");

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    size_t shared_mem = BLOCK_SIZE * sizeof(float4);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initial acceleration
    compute_accel_tiled_aos<<<grid, block, shared_mem>>>(d_bodies, d_acc, N);
    cudaDeviceSynchronize();

    printf("Running...\n[");
    fflush(stdout);

    int progress_step = NSTEPS / 50;
    float total_kernel_time = 0.0f;
    int total_collisions = 0;

    for (int step = 0; step < NSTEPS; ++step) {
        // Collision detection
        *h_num_collisions = 0;
        checkCuda(cudaMemcpy(d_num_collisions, h_num_collisions, sizeof(int), cudaMemcpyHostToDevice), "memcpy");
        
        detect_collisions<<<grid, block>>>(d_bodies, d_collision_pairs, d_num_collisions, N, COLLISION_THRESHOLD);
        cudaDeviceSynchronize();

        checkCuda(cudaMemcpy(h_num_collisions, d_num_collisions, sizeof(int), cudaMemcpyDeviceToHost), "memcpy");

        if (*h_num_collisions > 0) {
            int* h_collision_pairs = (int*)malloc((*h_num_collisions) * 2 * sizeof(int));
            checkCuda(cudaMemcpy(h_collision_pairs, d_collision_pairs, (*h_num_collisions) * 2 * sizeof(int), cudaMemcpyDeviceToHost), "memcpy");

            // Merge bodies
            dim3 merge_grid((*h_num_collisions + 255) / 256);
            merge_bodies<<<merge_grid, block>>>(d_bodies, d_collision_pairs, *h_num_collisions);
            cudaDeviceSynchronize();

            // Log collisions
            checkCuda(cudaMemcpy(h_bodies, d_bodies, N * sizeof(BodyAoS), cudaMemcpyDeviceToHost), "memcpy");
            for (int c = 0; c < *h_num_collisions; c++) {
                int i = h_collision_pairs[c * 2];
                int j = h_collision_pairs[c * 2 + 1];
                float dx = h_bodies[j].x - h_bodies[i].x;
                float dy = h_bodies[j].y - h_bodies[i].y;
                float dz = h_bodies[j].z - h_bodies[i].z;
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                float merged_mass = h_bodies[i].active ? h_bodies[i].mass : h_bodies[j].mass;
                fprintf(collision_file, "%.1f,%d,%d,%.6e,%.6e\n", 
                    step * DT, i, j, dist, merged_mass);
            }
            total_collisions += *h_num_collisions;

            free(h_collision_pairs);
        }

        // Physics integration
        update_positions_vel_firsthalf<<<grid, block>>>(d_bodies, d_acc, N, DT);

        cudaEventRecord(start);
        compute_accel_tiled_aos<<<grid, block, shared_mem>>>(d_bodies, d_acc, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float kernel_time;
        cudaEventElapsedTime(&kernel_time, start, stop);
        total_kernel_time += kernel_time;

        update_velocity_secondhalf<<<grid, block>>>(d_bodies, d_acc, N, DT);
        cudaDeviceSynchronize();

        // Output data
        if (step % OUTPUT_INTERVAL == 0) {
            checkCuda(cudaMemcpy(h_bodies, d_bodies, N * sizeof(BodyAoS), cudaMemcpyDeviceToHost), "memcpy");

            float time = step * DT;
            int active_count = 0;

            double ke = 0.0, pe = 0.0;
            for (int i = 0; i < N; ++i) {
                if (!h_bodies[i].active) continue;
                active_count++;

                ke += 0.5 * h_bodies[i].mass * (h_bodies[i].vx * h_bodies[i].vx +
                    h_bodies[i].vy * h_bodies[i].vy +
                    h_bodies[i].vz * h_bodies[i].vz);
                for (int j = i + 1; j < N; ++j) {
                    if (!h_bodies[j].active) continue;
                    float dx = h_bodies[j].x - h_bodies[i].x;
                    float dy = h_bodies[j].y - h_bodies[i].y;
                    float dz = h_bodies[j].z - h_bodies[i].z;
                    float dist = sqrtf(dx * dx + dy * dy + dz * dz + SOFTENING);
                    pe += (G * h_bodies[i].mass * h_bodies[j].mass) / dist;
                }
            }
            fprintf(energy_file, "%.1f,%.12e,%.12e,%.12e\n", time, ke - pe, ke, pe);
            fprintf(active_bodies_file, "%.1f,%d\n", time, active_count);

            for (int i = 0; i < N; ++i) {
                fprintf(pos_file, "%.1f,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%d\n",
                    time, i, h_bodies[i].x, h_bodies[i].y, h_bodies[i].z,
                    h_bodies[i].vx, h_bodies[i].vy, h_bodies[i].vz, h_bodies[i].mass,
                    h_bodies[i].active);
            }

            if (N > 9 && h_bodies[9].active) {
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
            fflush(collision_file);
            fflush(active_bodies_file);
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
    fclose(collision_file);
    fclose(active_bodies_file);

    float avg_time = total_kernel_time / NSTEPS;
    long long total_inter = (long long)N * (long long)N * (long long)NSTEPS;
    double inter_per_sec = total_inter / (total_kernel_time / 1000.0);
    double gflops = (inter_per_sec * 20.0) / 1e9;

    printf("Performance:\n");
    printf("  Avg kernel: %.3f ms\n", avg_time);
    printf("  Total: %.2f sec\n", total_kernel_time / 1000.0);
    printf("  Inter/sec: %.3e\n", inter_per_sec);
    printf("  GFLOPS: %.2f\n\n", gflops);

    printf("Collision Statistics:\n");
    printf("  Total collisions: %d\n\n", total_collisions);

    printf("Output files created in output/ directory\n");
    printf("Run: python3 visualize_complete.py\n");

    cudaFree(d_bodies);
    cudaFree(d_acc);
    cudaFree(d_collision_pairs);
    cudaFree(d_num_collisions);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_bodies);
    free(h_num_collisions);

    return 0;
}
