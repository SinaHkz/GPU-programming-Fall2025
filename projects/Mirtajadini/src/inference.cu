#include "inference.cuh"
#include "kernels.cuh"
#include <cuda_fp16.h>

InferenceEngine::InferenceEngine(int batch_size,
    const std::vector<float>& w1, const std::vector<float>& b1,
    const std::vector<float>& w2, const std::vector<float>& b2)
    : m_batch_size(batch_size), h_w1(w1), h_b1(b1), h_w2(w2), h_b2(b2)
{
    CHECK_CUBLAS(cublasCreate(&m_cublasHandle));
    CHECK_CUBLAS(cublasSetMathMode(m_cublasHandle, CUBLAS_TENSOR_OP_MATH));
    for (int i = 0; i < N_STREAMS; i++) CHECK_CUDA(cudaStreamCreate(&m_streams[i]));

    //Mem allocation 
    CHECK_CUDA(cudaMalloc(&d_raw_input, batch_size * INPUT_DIM * sizeof(unsigned char)));

    //FP32 W & B
    CHECK_CUDA(cudaMalloc(&d_w1_fp32, w1.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b1_fp32, b1.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w2_fp32, w2.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b2_fp32, b2.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_w1_fp32, w1.data(), w1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b1_fp32, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w2_fp32, w2.data(), w2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b2_fp32, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice));

    //activation buffers (FP32)
    CHECK_CUDA(cudaMalloc(&d_input_fp32, batch_size * INPUT_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hidden_fp32, batch_size * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_fp32, batch_size * OUTPUT_DIM * sizeof(float)));

    //FP16 converted weights 
    CHECK_CUDA(cudaMalloc(&d_w1_f16, w1.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_w2_f16, w2.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_input_f16, batch_size * INPUT_DIM * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_hidden_f16, batch_size * HIDDEN_DIM * sizeof(half)));

    floatToHalfKernel << <(w1.size() + 255) / 256, 256 >> > (d_w1_fp32, d_w1_f16, w1.size());
    floatToHalfKernel << <(w2.size() + 255) / 256, 256 >> > (d_w2_fp32, d_w2_f16, w2.size());
    CHECK_CUDA(cudaDeviceSynchronize());
}

InferenceEngine::~InferenceEngine() {
    cudaFree(d_raw_input);
    cudaFree(d_w1_fp32); cudaFree(d_b1_fp32); cudaFree(d_w2_fp32); cudaFree(d_b2_fp32);
    cudaFree(d_input_fp32); cudaFree(d_hidden_fp32); cudaFree(d_output_fp32);
    cudaFree(d_w1_f16); cudaFree(d_w2_f16); cudaFree(d_input_f16); cudaFree(d_hidden_f16);
    cublasDestroy(m_cublasHandle);
    for (int i = 0; i < N_STREAMS; i++) cudaStreamDestroy(m_streams[i]);
}

float InferenceEngine::run(const std::vector<unsigned char>& raw_input, std::vector<float>& results, InferenceMode mode) {
    switch (mode) {
    case InferenceMode::CPU_Baseline: return runCPUBaseline(raw_input, results);
    case InferenceMode::GPU_Baseline_FP32: return runBaselineFP32(raw_input, results);
    case InferenceMode::GPU_Mixed_CUDA_Pipelined: return runMixedCudaPipelined(raw_input, results);
    case InferenceMode::GPU_Mixed_Tensor_Pipelined: return runMixedTensorPipelined(raw_input, results);
    }
    return 0.0f;
}

// Sequential CPU baseline 
float InferenceEngine::runCPUBaseline(const std::vector<unsigned char>& raw_input, std::vector<float>& results) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    std::vector<float> h_hidden(m_batch_size * HIDDEN_DIM);

    for (int b = 0; b < m_batch_size; ++b) {
        //L1: Input * W1 + B1
        for (int h = 0; h < HIDDEN_DIM; ++h) {
            float sum = 0.0f;
            for (int i = 0; i < INPUT_DIM; ++i) {
                //normalizing
                float pixel = (float)raw_input[b * INPUT_DIM + i];
                float norm = (pixel / 255.0f - 0.5f) / 0.5f;
                sum += norm * h_w1[i * HIDDEN_DIM + h];
            }
            sum += h_b1[h]; //+ B1
            // ReLU
            h_hidden[b * HIDDEN_DIM + h] = (sum > 0.0f) ? sum : 0.0f;
        }

        //L2: Hidden * W2 + B2
        for (int o = 0; o < OUTPUT_DIM; ++o) {
            float sum = 0.0f;
            for (int h = 0; h < HIDDEN_DIM; ++h) {
                sum += h_hidden[b * HIDDEN_DIM + h] * h_w2[h * OUTPUT_DIM + o];
            }
            sum += h_b2[o]; //+ B2
            results[b * OUTPUT_DIM + o] = sum;
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

//GPU Baseline FP32
float InferenceEngine::runBaselineFP32(const std::vector<unsigned char>& raw_input, std::vector<float>& results) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    CHECK_CUDA(cudaMemcpy(d_raw_input, raw_input.data(), raw_input.size(), cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    int total_pixels = m_batch_size * INPUT_DIM;
    preprocessNormalizeFP32 << <(total_pixels + 255) / 256, 256 >> > (d_raw_input, d_input_fp32, total_pixels);

    dim3 block(TILE_SIZE, TILE_SIZE);

    // Layer 1
    dim3 grid1((HIDDEN_DIM + TILE_SIZE - 1) / TILE_SIZE, (m_batch_size + TILE_SIZE - 1) / TILE_SIZE);
    matrixMulTiled << <grid1, block >> > (d_input_fp32, d_w1_fp32, d_hidden_fp32, m_batch_size, HIDDEN_DIM, INPUT_DIM);
    addBias << <(m_batch_size * HIDDEN_DIM + 255) / 256, 256 >> > (d_hidden_fp32, d_b1_fp32, m_batch_size, HIDDEN_DIM);
    reluKernel << <(m_batch_size * HIDDEN_DIM + 255) / 256, 256 >> > (d_hidden_fp32, m_batch_size * HIDDEN_DIM);

    // Layer 2
    dim3 grid2((OUTPUT_DIM + TILE_SIZE - 1) / TILE_SIZE, (m_batch_size + TILE_SIZE - 1) / TILE_SIZE);
    matrixMulTiled << <grid2, block >> > (d_hidden_fp32, d_w2_fp32, d_output_fp32, m_batch_size, OUTPUT_DIM, HIDDEN_DIM);
    addBias << <(m_batch_size * OUTPUT_DIM + 255) / 256, 256 >> > (d_output_fp32, d_b2_fp32, m_batch_size, OUTPUT_DIM);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    CHECK_CUDA(cudaMemcpy(results.data(), d_output_fp32, results.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float ms; cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

//Mixed Precision (CUDA)
float InferenceEngine::runMixedCudaPipelined(const std::vector<unsigned char>& raw_input, std::vector<float>& results) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start, m_streams[0]);

    int chunk_size = m_batch_size / N_STREAMS;
    dim3 block(TILE_SIZE, TILE_SIZE);

    for (int i = 0; i < N_STREAMS; ++i) {
        int offset_raw = i * chunk_size * INPUT_DIM;
        int offset_in = i * chunk_size * INPUT_DIM;
        int offset_hid = i * chunk_size * HIDDEN_DIM;
        int offset_out = i * chunk_size * OUTPUT_DIM;

        //H2D
        CHECK_CUDA(cudaMemcpyAsync(d_raw_input + offset_raw, raw_input.data() + offset_raw,
            chunk_size * INPUT_DIM, cudaMemcpyHostToDevice, m_streams[i]));

        //Normalize (FP16 Output)
        preprocessNormalizeFP16 << <(chunk_size * INPUT_DIM + 255) / 256, 256, 0, m_streams[i] >> > (
            d_raw_input + offset_raw, d_input_f16 + offset_in, chunk_size * INPUT_DIM);

        //Layer 1 (Custom FP16 Kernel)
        dim3 grid1((HIDDEN_DIM + TILE_SIZE - 1) / TILE_SIZE, (chunk_size + TILE_SIZE - 1) / TILE_SIZE);
        matrixMulTiledFP16 << <grid1, block, 0, m_streams[i] >> > (
            d_input_f16 + offset_in, d_w1_f16, d_hidden_fp32 + offset_hid, chunk_size, HIDDEN_DIM, INPUT_DIM);

        //Bias & ReLU (FP32)
        addBias << <(chunk_size * HIDDEN_DIM + 255) / 256, 256, 0, m_streams[i] >> > (d_hidden_fp32 + offset_hid, d_b1_fp32, chunk_size, HIDDEN_DIM);
        reluKernel << <(chunk_size * HIDDEN_DIM + 255) / 256, 256, 0, m_streams[i] >> > (d_hidden_fp32 + offset_hid, chunk_size * HIDDEN_DIM);

        //Convert Hidden to FP16
        floatToHalfKernel << <(chunk_size * HIDDEN_DIM + 255) / 256, 256, 0, m_streams[i] >> > (
            d_hidden_fp32 + offset_hid, d_hidden_f16 + offset_hid, chunk_size * HIDDEN_DIM);

        //Layer 2 (Custom FP16 Kernel)
        dim3 grid2((OUTPUT_DIM + TILE_SIZE - 1) / TILE_SIZE, (chunk_size + TILE_SIZE - 1) / TILE_SIZE);
        matrixMulTiledFP16 << <grid2, block, 0, m_streams[i] >> > (
            d_hidden_f16 + offset_hid, d_w2_f16, d_output_fp32 + offset_out, chunk_size, OUTPUT_DIM, HIDDEN_DIM);

        //Bias (FP32)
        addBias << <(chunk_size * OUTPUT_DIM + 255) / 256, 256, 0, m_streams[i] >> > (d_output_fp32 + offset_out, d_b2_fp32, chunk_size, OUTPUT_DIM);

        //D2H
        CHECK_CUDA(cudaMemcpyAsync(results.data() + offset_out, d_output_fp32 + offset_out,
            chunk_size * OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost, m_streams[i]));
    }

    cudaEventRecord(stop, m_streams[0]);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms; cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

//Mixed Precision (TCs)
float InferenceEngine::runMixedTensorPipelined(const std::vector<unsigned char>& raw_input, std::vector<float>& results) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start, m_streams[0]);

    int chunk_size = m_batch_size / N_STREAMS;
    float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < N_STREAMS; ++i) {
        int offset_raw = i * chunk_size * INPUT_DIM;
        int offset_in = i * chunk_size * INPUT_DIM;
        int offset_hid = i * chunk_size * HIDDEN_DIM;
        int offset_out = i * chunk_size * OUTPUT_DIM;

        //H2D
        CHECK_CUDA(cudaMemcpyAsync(d_raw_input + offset_raw, raw_input.data() + offset_raw,
            chunk_size * INPUT_DIM, cudaMemcpyHostToDevice, m_streams[i]));

        preprocessNormalizeFP16 << <(chunk_size * INPUT_DIM + 255) / 256, 256, 0, m_streams[i] >> > (
            d_raw_input + offset_raw, d_input_f16 + offset_in, chunk_size * INPUT_DIM);

        //Layer 1
        cublasSetStream(m_cublasHandle, m_streams[i]);
        CHECK_CUBLAS(cublasGemmEx(m_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
            HIDDEN_DIM, chunk_size, INPUT_DIM,
            &alpha, d_w1_f16, CUDA_R_16F, HIDDEN_DIM,
            d_input_f16 + offset_in, CUDA_R_16F, INPUT_DIM,
            &beta, d_hidden_fp32 + offset_hid, CUDA_R_32F, HIDDEN_DIM,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        //Bias & ReLU
        addBias << <(chunk_size * HIDDEN_DIM + 255) / 256, 256, 0, m_streams[i] >> > (d_hidden_fp32 + offset_hid, d_b1_fp32, chunk_size, HIDDEN_DIM);
        reluKernel << <(chunk_size * HIDDEN_DIM + 255) / 256, 256, 0, m_streams[i] >> > (d_hidden_fp32 + offset_hid, chunk_size * HIDDEN_DIM);

        //Convert to FP16
        floatToHalfKernel << <(chunk_size * HIDDEN_DIM + 255) / 256, 256, 0, m_streams[i] >> > (
            d_hidden_fp32 + offset_hid, d_hidden_f16 + offset_hid, chunk_size * HIDDEN_DIM);

        //Layer 2 (Tensor Core)
        CHECK_CUBLAS(cublasGemmEx(m_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
            OUTPUT_DIM, chunk_size, HIDDEN_DIM,
            &alpha, d_w2_f16, CUDA_R_16F, OUTPUT_DIM,
            d_hidden_f16 + offset_hid, CUDA_R_16F, HIDDEN_DIM,
            &beta, d_output_fp32 + offset_out, CUDA_R_32F, OUTPUT_DIM,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        //Bias
        addBias << <(chunk_size * OUTPUT_DIM + 255) / 256, 256, 0, m_streams[i] >> > (d_output_fp32 + offset_out, d_b2_fp32, chunk_size, OUTPUT_DIM);

        //D2H
        CHECK_CUDA(cudaMemcpyAsync(results.data() + offset_out, d_output_fp32 + offset_out,
            chunk_size * OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost, m_streams[i]));
    }

    cudaEventRecord(stop, m_streams[0]);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms; cudaEventElapsedTime(&ms, start, stop);
    return ms;
}