#include <stdio.h>
#include "../../include/timer.h"
#include "../../include/utils.h"


void timer_init(GpuTimer* timer) {
    if (timer) {
        CHECK_CUDA(cudaEventCreate(&timer->start));
        CHECK_CUDA(cudaEventCreate(&timer->stop));
    }
}


void timer_start(GpuTimer* timer) {
    if (timer) {
        CHECK_CUDA(cudaEventRecord(timer->start, 0));
    }
}


void timer_stop(GpuTimer* timer) {
    if (timer) {
        CHECK_CUDA(cudaEventRecord(timer->stop, 0));
    }
}


float timer_elapsed_ms(GpuTimer* timer) {
    if (!timer) return 0.0f;

    CHECK_CUDA(cudaEventSynchronize(timer->stop));

    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, timer->start, timer->stop));
    
    return milliseconds;
}


void timer_destroy(GpuTimer* timer) {
    if (timer) {
        CHECK_CUDA(cudaEventDestroy(timer->start));
        CHECK_CUDA(cudaEventDestroy(timer->stop));
    }
}