#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Macro to wrap CUDA API calls.
 * * Checks the return status of CUDA functions (e.g., cudaMalloc, cudaMemcpy).
 * * If the call fails, it prints the error name, string description, file, and line number,
 * then terminates the program.
 * * Usage: CHECK_CUDA(cudaMalloc(&ptr, size));
 */
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error:\n"); \
        fprintf(stderr, "    File:       %s\n", __FILE__); \
        fprintf(stderr, "    Line:       %d\n", __LINE__); \
        fprintf(stderr, "    Error Code: %d\n", err); \
        fprintf(stderr, "    Error Text: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

/**
 * @brief Utility function to check for errors after a kernel launch.
 * * Unlike API calls, kernel launches are asynchronous and don't return a status directly.
 * * This function checks cudaGetLastError() to catch launch failures (e.g., invalid grid args)
 * and cudaDeviceSynchronize() to catch execution failures (e.g., illegal memory access),
 * though the latter is slow and should be used primarily for debugging.
 * * @param kernel_name A string identifier for the kernel being checked (for logging).
 */
inline void check_kernel_launch(const char* kernel_name) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel Launch Error [%s]: %s\n", kernel_name, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Uncomment the following line for strict debugging (but it kills performance due to sync)
    // err = cudaDeviceSynchronize();
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "Kernel Execution Error [%s]: %s\n", kernel_name, cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
}

#endif // UTILS_H