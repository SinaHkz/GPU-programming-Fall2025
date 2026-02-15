#include "physics.h"

// Naive
__global__ void bodyForceKernelNaive(float4* p, float3* v, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;

    // Math
    for (int j = 0; j < n; j++) {
        float dx = p[j].x - p[i].x;
        float dy = p[j].y - p[i].y;
        float dz = p[j].z - p[i].z;
        
        float distSqr = dx*dx + dy*dy + dz*dz + 1e-9f; 
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        float F = p[j].w * invDist3;
        
        fx += dx * F;
        fy += dy * F;
        fz += dz * F;
    }

    // Velocity
    v[i].x += dt * fx;
    v[i].y += dt * fy;
    v[i].z += dt * fz;
}

// Tiled
__global__ void bodyForceKernelTiled(float4* p, float3* v, float dt, int n) {
    extern __shared__ float4 sh_p[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = threadIdx.x;
    
    float4 my_p;
    if (i < n) my_p = p[i];

    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;

    // Tiles
    for (int t = 0; t < gridDim.x; t++) {
        int idx = t * blockDim.x + ti;
        
        // Load
        if (idx < n) {
            sh_p[ti] = p[idx];
        } else {
            sh_p[ti] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
        __syncthreads();

        // Math
        if (i < n) {
            #pragma unroll
            for (int j = 0; j < blockDim.x; j++) {
                float dx = sh_p[j].x - my_p.x;
                float dy = sh_p[j].y - my_p.y;
                float dz = sh_p[j].z - my_p.z;
                
                float distSqr = dx*dx + dy*dy + dz*dz + 1e-9f; 
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;

                float F = sh_p[j].w * invDist3;
                
                fx += dx * F;
                fy += dy * F;
                fz += dz * F;
            }
        }
        __syncthreads();
    }

    // Velocity
    if (i < n) {
        v[i].x += dt * fx;
        v[i].y += dt * fy;
        v[i].z += dt * fz;
    }
}

// Update
__global__ void integratePositionsKernel(float4* p, float3* v, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i].x += v[i].x * dt;
        p[i].y += v[i].y * dt;
        p[i].z += v[i].z * dt;
    }
}

// RunNaive
void runSimulationGPU_Naive(float4* h_p, float3* h_v, float dt, int n, int iters) {
    int b_p = n * sizeof(float4);
    int b_v = n * sizeof(float3);

    float4 *d_p;
    float3 *d_v;

    // Mem
    cudaMalloc(&d_p, b_p);
    cudaMalloc(&d_v, b_v);
    cudaMemcpy(d_p, h_p, b_p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, b_v, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n + block - 1) / block;

    // Loop
    for (int step = 0; step < iters; step++) {
        bodyForceKernelNaive<<<grid, block>>>(d_p, d_v, dt, n);
        integratePositionsKernel<<<grid, block>>>(d_p, d_v, dt, n);
    }
    
    // Return
    cudaMemcpy(h_p, d_p, b_p, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v, d_v, b_v, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_p);
    cudaFree(d_v);
}

// RunTiled
void runSimulationGPU_Tiled(float4* h_p, float3* h_v, float dt, int n, int iters) {
    int b_p = n * sizeof(float4);
    int b_v = n * sizeof(float3);

    float4 *d_p;
    float3 *d_v;

    // Mem
    cudaMalloc(&d_p, b_p);
    cudaMalloc(&d_v, b_v);
    cudaMemcpy(d_p, h_p, b_p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, b_v, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n + block - 1) / block;
    int sh_mem = block * sizeof(float4);

    // Loop
    for (int step = 0; step < iters; step++) {
        bodyForceKernelTiled<<<grid, block, sh_mem>>>(d_p, d_v, dt, n);
        integratePositionsKernel<<<grid, block>>>(d_p, d_v, dt, n);
    }
    
    // Return
    cudaMemcpy(h_p, d_p, b_p, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v, d_v, b_v, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_p);
    cudaFree(d_v);
}

// RunStream
void runSimulationGPU_Streamed(float4* h_p, float3* h_v, float dt, int n, int iters) {
    int b_p = n * sizeof(float4);
    int b_v = n * sizeof(float3);

    float4 *d_p;
    float3 *d_v;

    // Mem
    cudaMalloc(&d_p, b_p);
    cudaMalloc(&d_v, b_v);

    // Stream
    cudaStream_t s;
    cudaStreamCreate(&s);

    // Async
    cudaMemcpyAsync(d_p, h_p, b_p, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_v, h_v, b_v, cudaMemcpyHostToDevice, s);

    int block = 256;
    int grid = (n + block - 1) / block;
    int sh_mem = block * sizeof(float4);

    // Loop
    for (int step = 0; step < iters; step++) {
        bodyForceKernelTiled<<<grid, block, sh_mem, s>>>(d_p, d_v, dt, n);
        integratePositionsKernel<<<grid, block, 0, s>>>(d_p, d_v, dt, n);
    }
    
    // Return
    cudaMemcpyAsync(h_p, d_p, b_p, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_v, d_v, b_v, cudaMemcpyDeviceToHost, s);

    // Sync
    cudaStreamSynchronize(s);

    // Cleanup
    cudaStreamDestroy(s);
    cudaFree(d_p);
    cudaFree(d_v);
}