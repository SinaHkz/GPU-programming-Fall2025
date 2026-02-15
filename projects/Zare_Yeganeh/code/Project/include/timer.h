#ifndef TIMER_H
#define TIMER_H

#include <cuda_runtime.h>

/**
 * @brief C-Style GPU Timer Struct using CUDA Events.
 * * Used to measure kernel execution time precisely on the GPU.
 */
struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;
};

// Functions to manage the timer
void timer_init(GpuTimer* timer);
void timer_start(GpuTimer* timer);
void timer_stop(GpuTimer* timer);
float timer_elapsed_ms(GpuTimer* timer);
void timer_destroy(GpuTimer* timer);

#endif // TIMER_H