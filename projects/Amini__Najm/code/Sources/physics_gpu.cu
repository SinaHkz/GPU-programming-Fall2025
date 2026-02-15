#include "types.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

const float G = 1.0f;
const float SOFTENING = 1e-9f;



__global__ void bodyForceKernelNaive(float4* p, float3* v, float dt, int n) {
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;
    
    
    float4 myPos = p[i]; 

    for (int j = 0; j < n; j++) {
        
        float4 otherPos = p[j]; 
        
        float dx = otherPos.x - myPos.x;
        float dy = otherPos.y - myPos.y;
        float dz = otherPos.z - myPos.z;

        float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        

        float invDist = rsqrtf(distSqr); 
        float invDist3 = invDist * invDist * invDist;

        float force_magnitude = G * otherPos.w * invDist3;

        Fx += dx * force_magnitude;
        Fy += dy * force_magnitude;
        Fz += dz * force_magnitude;
    }


    v[i].x += Fx * dt;
    v[i].y += Fy * dt;
    v[i].z += Fz * dt;
}

__global__ void integratePositionsKernel(float4* p, float3* v, float dt, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    p[i].x += v[i].x * dt;
    p[i].y += v[i].y * dt;
    p[i].z += v[i].z * dt;
}



void runSimulationGPU_Naive(float4* h_p, float3* h_v, float dt, int nBodies, int nIters) {
    int bytes_pos = nBodies * sizeof(float4);
    int bytes_vel = nBodies * sizeof(float3);

    float4 *d_p;
    float3 *d_v;

    
    cudaMalloc(&d_p, bytes_pos);
    cudaMalloc(&d_v, bytes_vel);

    
    cudaMemcpy(d_p, h_p, bytes_pos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, bytes_vel, cudaMemcpyHostToDevice);

    
    int blockSize = 256;
    int gridSize = (nBodies + blockSize - 1) / blockSize;

    
    for (int step = 0; step < nIters; step++) {
        bodyForceKernelNaive<<<gridSize, blockSize>>>(d_p, d_v, dt, nBodies);
        integratePositionsKernel<<<gridSize, blockSize>>>(d_p, d_v, dt, nBodies);
    }
    
   
    cudaDeviceSynchronize();

  
    cudaMemcpy(h_p, d_p, bytes_pos, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v, d_v, bytes_vel, cudaMemcpyDeviceToHost);

   
    cudaFree(d_p);
    cudaFree(d_v);
}



__global__ void bodyForceKernelTiled(float4* p, float3* v, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
   
    __shared__ float4 sharedPos[256]; 
    
    float4 myPos;
    float3 accel = {0.0f, 0.0f, 0.0f};

    if (i < n) {
        myPos = p[i];
    }

    
    for (int tile = 0; tile < gridDim.x; tile++) {
        
        // 1. COOPERATIVE LOAD: Each thread loads exactly one body into the shared tile
        int idx = tile * blockDim.x + threadIdx.x;
        if (idx < n) {
            sharedPos[threadIdx.x] = p[idx];
        } else {
            
            sharedPos[threadIdx.x] = {0.0f, 0.0f, 0.0f, 0.0f}; 
        }

       
        __syncthreads();

        // 2. COMPUTE: Now all threads calculate forces against the fast shared memory
        if (i < n) {
            
            #pragma unroll 32
            for (int j = 0; j < blockDim.x; j++) {
                float4 otherPos = sharedPos[j];
                
                float dx = otherPos.x - myPos.x;
                float dy = otherPos.y - myPos.y;
                float dz = otherPos.z - myPos.z;

                float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;

                float force_magnitude = G * otherPos.w * invDist3;

                accel.x += dx * force_magnitude;
                accel.y += dy * force_magnitude;
                accel.z += dz * force_magnitude;
            }
        }

        
        __syncthreads();
    }

    // 3. UPDATE: Write the final velocity back to global memory
    if (i < n) {
        v[i].x += accel.x * dt;
        v[i].y += accel.y * dt;
        v[i].z += accel.z * dt;
    }
}


void runSimulationGPU_Tiled(float4* h_p, float3* h_v, float dt, int nBodies, int nIters) {
    int bytes_pos = nBodies * sizeof(float4);
    int bytes_vel = nBodies * sizeof(float3);

    float4 *d_p;
    float3 *d_v;

    cudaMalloc(&d_p, bytes_pos);
    cudaMalloc(&d_v, bytes_vel);

    cudaMemcpy(d_p, h_p, bytes_pos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, bytes_vel, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (nBodies + blockSize - 1) / blockSize;

    for (int step = 0; step < nIters; step++) {
        bodyForceKernelTiled<<<gridSize, blockSize>>>(d_p, d_v, dt, nBodies);
        integratePositionsKernel<<<gridSize, blockSize>>>(d_p, d_v, dt, nBodies);
    }
    
    cudaDeviceSynchronize();

    cudaMemcpy(h_p, d_p, bytes_pos, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v, d_v, bytes_vel, cudaMemcpyDeviceToHost);

    cudaFree(d_p);
    cudaFree(d_v);
}


void runSimulationGPU_Streamed(float4* h_p, float3* h_v, float dt, int nBodies, int nIters) {
    int bytes_pos = nBodies * sizeof(float4);
    int bytes_vel = nBodies * sizeof(float3);

    float4 *d_p;
    float3 *d_v;

    cudaMalloc(&d_p, bytes_pos);
    cudaMalloc(&d_v, bytes_vel);

    // 1. Initialize the Custom CUDA Stream
    cudaStream_t compute_stream;
    cudaStreamCreate(&compute_stream);

    // 2. Perform Asynchronous Memory Transfers

    cudaMemcpyAsync(d_p, h_p, bytes_pos, cudaMemcpyHostToDevice, compute_stream);
    cudaMemcpyAsync(d_v, h_v, bytes_vel, cudaMemcpyHostToDevice, compute_stream);

    int blockSize = 256;
    int gridSize = (nBodies + blockSize - 1) / blockSize;

    // 3. Launch the kernels directly into our custom stream
 
    for (int step = 0; step < nIters; step++) {
        bodyForceKernelTiled<<<gridSize, blockSize, 0, compute_stream>>>(d_p, d_v, dt, nBodies);
        integratePositionsKernel<<<gridSize, blockSize, 0, compute_stream>>>(d_p, d_v, dt, nBodies);
    }
    
    // 4. Asynchronously copy the final results back to the Host
    cudaMemcpyAsync(h_p, d_p, bytes_pos, cudaMemcpyDeviceToHost, compute_stream);
    cudaMemcpyAsync(h_v, d_v, bytes_vel, cudaMemcpyDeviceToHost, compute_stream);

    // 5. Block the CPU *only* at the very end to ensure all stream operations are finished
    cudaStreamSynchronize(compute_stream);

    // 6. Clean up
    cudaStreamDestroy(compute_stream);
    cudaFree(d_p);
    cudaFree(d_v);
}