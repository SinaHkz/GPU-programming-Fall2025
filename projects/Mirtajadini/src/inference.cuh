#pragma once
#include <vector>
#include <cublas_v2.h>
#include "config.h"

enum class InferenceMode {
    CPU_Baseline,
    GPU_Baseline_FP32,
    GPU_Mixed_CUDA_Pipelined,
    GPU_Mixed_Tensor_Pipelined
};

class InferenceEngine {
public:
    InferenceEngine(int batch_size,
        const std::vector<float>& w1, const std::vector<float>& b1,
        const std::vector<float>& w2, const std::vector<float>& b2);
    ~InferenceEngine();

    float run(const std::vector<unsigned char>& raw_input, std::vector<float>& results, InferenceMode mode);

private:
    int m_batch_size;
    cublasHandle_t m_cublasHandle;

    //host weights (For CPU)
    std::vector<float> h_w1, h_b1, h_w2, h_b2;

    // device pointers
    float* d_w1_fp32, * d_b1_fp32, * d_w2_fp32, * d_b2_fp32;
    float* d_input_fp32, * d_hidden_fp32, * d_output_fp32;

    half* d_w1_f16, * d_w2_f16;
    half* d_input_f16, * d_hidden_f16;

    unsigned char* d_raw_input;

    //Streams
    static const int N_STREAMS = 4;
    cudaStream_t m_streams[N_STREAMS];

    float runCPUBaseline(const std::vector<unsigned char>& raw_input, std::vector<float>& results);
    float runBaselineFP32(const std::vector<unsigned char>& raw_input, std::vector<float>& results);
    float runMixedCudaPipelined(const std::vector<unsigned char>& raw_input, std::vector<float>& results);
    float runMixedTensorPipelined(const std::vector<unsigned char>& raw_input, std::vector<float>& results);
};