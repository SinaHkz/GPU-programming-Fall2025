#include <cuda_runtime.h>
#include <stdio.h>
#include "../../include/utils.h"

void print_device_info() {
    int device_id;
    cudaGetDevice(&device_id);

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));

    printf("--- GPU Device Info ---\n");
    printf("Name:                  %s\n", prop.name);
    printf("Compute Capability:    %d.%d\n", prop.major, prop.minor);
    printf("Global Memory:         %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("SM Count:              %d\n", prop.multiProcessorCount);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per SM:    %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warp Size:             %d\n", prop.warpSize);
    printf("L2 Cache Size:         %d bytes\n", prop.l2CacheSize);
    printf("-----------------------\n\n");
}

void init_cuda_device() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA capable devices found.\n");
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA(cudaSetDevice(0));
}