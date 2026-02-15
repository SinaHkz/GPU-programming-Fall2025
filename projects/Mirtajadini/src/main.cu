#include "config.h"
#include "utils.h"
#include "inference.cuh"
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <cuda_runtime.h>
#include <string>


//get CUDA Cores per SM based on Architecture
int getCudaCoresPerSM(int major, int minor) {
    // Maxwell
    if (major == 5) {
        if (minor == 0 || minor == 2) return 128;
        if (minor == 3) return 128;
    }
    // Pascal
    if (major == 6) {
        if (minor == 0) return 64;  
        if (minor == 1) return 128;
        if (minor == 2) return 128;
    }
    // Volta
    if (major == 7) {
        if (minor == 0) return 64; // V100
        // Turing
        if (minor == 5) return 64; // RTX 20 series
    }
    // Ampere
    if (major == 8) {
        if (minor == 0) return 64;  // A100
        if (minor == 6) return 128; // RTX 30 series (GA102)
        if (minor == 9) return 128; // Ada Lovelace (RTX 40 series)
    }
    // Hopper
    if (major == 9) {
        if (minor == 0) return 128; // H100
    }

    return 0;
}

//get Tensor Cores per SM based on Architecture
int getTensorCoresPerSM(int major, int minor) {
    // Volta (1st Gen Tensor Cores)
    if (major == 7 && minor == 0) return 8;

    // Turing (2nd Gen Tensor Cores)
    if (major == 7 && minor == 5) return 8;

    // Ampere (3rd Gen Tensor Cores)
    if (major == 8) return 4;

    // Hopper (4th Gen Tensor Cores)
    if (major == 9) return 4;

    // Blackwell (5th Gen) - Preliminary
    if (major == 10) return 4;

    return 0;
}

void printDeviceInfo() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
            (int)error_id, cudaGetErrorString(error_id));
        return;
    }

    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    }
    else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        int major = deviceProp.major;
        int minor = deviceProp.minor;

        // Calculations
        int cudaCoresPerSM = getCudaCoresPerSM(major, minor);
        int tensorCoresPerSM = getTensorCoresPerSM(major, minor);
        long totalCudaCores = (long)cudaCoresPerSM * deviceProp.multiProcessorCount;
        long totalTensorCores = (long)tensorCoresPerSM * deviceProp.multiProcessorCount;

        printf("\n------------------------------------------------------------\n");
        printf("Setup and Device %d: \"%s\"\n", dev, deviceProp.name);
        printf("------------------------------------------------------------\n");

        printf("  Compute Capability:            %d.%d\n", major, minor);
        printf("  Total Global Memory:           %.0f MB\n",
            (float)deviceProp.totalGlobalMem / 1048576.0f);
        
        printf("  Memory Bus Width:              %d-bit\n", deviceProp.memoryBusWidth);
        printf("  L2 Cache Size:                 %d bytes\n", deviceProp.l2CacheSize);

        //SM and Thread Info
        printf("\n  [Streaming Multiprocessors (SMs)]\n");
        printf("  Number of SMs:                 %d\n", deviceProp.multiProcessorCount);
        printf("  Max Threads per SM:            %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Max Threads per Block:         %d\n", deviceProp.maxThreadsPerBlock);

        //Warp Info
        printf("\n  [Warp Information]\n");
        printf("  Warp Size:                     %d\n", deviceProp.warpSize);
        //max warps per SM:
        printf("  Max Warps per SM:              %d\n",
            deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize);

        //Core Counts
        printf("\n  [Core Counts]\n");
        if (cudaCoresPerSM > 0) {
            printf("  CUDA Cores per SM:             %d\n", cudaCoresPerSM);
            printf("  Total CUDA Cores:              %ld\n", totalCudaCores);
        }
        else {
            printf("  CUDA Cores:                    (Unknown Architecture)\n");
        }

        //Tensor Cores 
        if (tensorCoresPerSM > 0) {
            printf("  Tensor Cores per SM:           %d\n", tensorCoresPerSM);
            printf("  Total Tensor Cores:            %ld\n", totalTensorCores);
        }
        else {
            printf("  Tensor Cores:                  (Not Supported or Unknown)\n");
        }

        printf("------------------------------------------------------------\n\n");
    }
}

void print_metric_row(const char* name, float time_ms, int batch_size, double baseline_time = -1.0) {
    double time_sec = time_ms / 1000.0;
    double throughput = (double)batch_size / time_sec;

    // Ops = 2 * (M*K*N) + Bias Adds
    double ops_layer1 = 2.0 * (double)batch_size * (double)INPUT_DIM * (double)HIDDEN_DIM;
    double ops_layer2 = 2.0 * (double)batch_size * (double)HIDDEN_DIM * (double)OUTPUT_DIM;
    double tflops = ((ops_layer1 + ops_layer2) / 1e12) / (time_sec);

    printf("%-30s | %-10.3f | %-12.2f | %-10.4f", name, time_ms, throughput, tflops);

    if (baseline_time > 0) {
        printf(" | %.2fx", (baseline_time) / time_ms);
    }
    printf("\n");
}

int main() {
    printf("MPI and Image recognition \n\n");
    printDeviceInfo();
    printf("\n\nTarget: STL-10 (5 Classes)\n");
    printf("Comparisons: Seq vs CUDA FP32 vs MP (CUDA) vs MP (TC):\n");

    //load data
    std::vector<unsigned char> h_raw_input;
    std::vector<std::string> filenames;
    std::string data_path = find_dataset_folder();
    int batch_size = 0;
    if (data_path.empty()) {
        printf("WARNING: 'STL_image' not found. Using Random Data.\n");
        batch_size = 128;
        h_raw_input.resize(batch_size * INPUT_DIM);
        for (auto& x : h_raw_input) x = rand() % 255;
    }
    else {
        batch_size = load_dataset(data_path, h_raw_input, filenames);
    }

    if (batch_size % 4 != 0) {
        batch_size = (batch_size / 4) * 4;
        printf("Adjusted Batch Size to %d for pipelining.\n", batch_size);
    }
    if (batch_size == 0) return -1;

    //load W & B
    std::vector<float> h_w1, h_b1, h_w2, h_b2;
    load_or_generate_weights("E:/Codez/CUDA_prc/MPI/w1.bin", h_w1, INPUT_DIM * HIDDEN_DIM);
    load_or_generate_weights("E:/Codez/CUDA_prc/MPI/b1.bin", h_b1, HIDDEN_DIM);
    load_or_generate_weights("E:/Codez/CUDA_prc/MPI/w2.bin", h_w2, HIDDEN_DIM * OUTPUT_DIM);
    load_or_generate_weights("E:/Codez/CUDA_prc/MPI/b2.bin", h_b2, OUTPUT_DIM);

    //init inference engine
    InferenceEngine engine(batch_size, h_w1, h_b1, h_w2, h_b2);

    //run comparisons
    std::vector<float> res_cpu(batch_size * OUTPUT_DIM);
    std::vector<float> res_fp32(batch_size * OUTPUT_DIM);
    std::vector<float> res_cuda_mix(batch_size * OUTPUT_DIM);
    std::vector<float> res_tens_mix(batch_size * OUTPUT_DIM);

    printf("\n Benchmarks are running...\n");

    //CPU Baseline
    printf("Running CPU Baseline (Sequential)...\n");
    float t_cpu = engine.run(h_raw_input, res_cpu, InferenceMode::CPU_Baseline);

    //GPU FP32 Baseline
    engine.run(h_raw_input, res_fp32, InferenceMode::GPU_Baseline_FP32);
    printf("Running GPU Baseline (FP32 Custom Kernels)...\n");
    float t_fp32 = engine.run(h_raw_input, res_fp32, InferenceMode::GPU_Baseline_FP32);

    //Mixed (CUDA Cores - Pipelined)
    printf("Running GPU Mixed Precision (CUDA Cores + Pipelined)...\n");
    float t_cuda_mix = engine.run(h_raw_input, res_cuda_mix, InferenceMode::GPU_Mixed_CUDA_Pipelined);

    //Mixed (Tensor Cores - Pipelined)
    printf("Running GPU Mixed Precision (Tensor Cores + Pipelined)...\n");
    float t_tens_mix = engine.run(h_raw_input, res_tens_mix, InferenceMode::GPU_Mixed_Tensor_Pipelined);

    //Results
    printf("\n Performance Metrics \n");
    printf("%-30s | %-10s | %-12s | %-10s | %-8s\n", "Implementation", "Time (ms)", "Img/Sec", "TFLOPS", "Speedup (vs CPU)");
    printf("--------------------------------------------------------------------------------------------------------------\n");
    print_metric_row("CPU Baseline", t_cpu, batch_size, t_cpu);
    print_metric_row("GPU Baseline (FP32)", t_fp32, batch_size, t_cpu);
    print_metric_row("MPI CUDA +Pipelined", t_cuda_mix, batch_size, t_cpu);
    print_metric_row("MPI TCs + Pipelined", t_tens_mix, batch_size, t_cpu);

    //Analysis
    float mse_fp32 = calculate_mse(res_cpu, res_fp32);
    float mse_cuda = calculate_mse(res_cpu, res_cuda_mix);
    float mse_tens = calculate_mse(res_cpu, res_tens_mix);

    printf("\n Accuracy (MSE vs CPU) \n");
    printf("GPU FP32 MSE:        %e\n", mse_fp32);
    printf("MPI CUDA:    %e\n", mse_cuda);
    printf("MPI TCs:  %e\n", mse_tens);

    if (!filenames.empty()) {
        print_predictions(res_tens_mix, filenames, batch_size, 20);
    }

    return 0;
}