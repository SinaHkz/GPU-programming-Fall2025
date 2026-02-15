#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <cstdlib>

#include "conv_backward.h"
#include "tensor.h"
#include "conv_fused.h"
#include "pool_layer.h"
#include "fc_layer.h"  
#include "activations.h"  
#include "optimizer.h"     
#include "loss.h"              
#include "mnist_loader.h"


static inline void ck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::printf("CUDA error: %s | %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}
static inline void kcheck(const char* where) {
    cudaError_t e = cudaPeekAtLastError();
    if (e != cudaSuccess) {
        std::printf("%s Launch Error: %s\n", where, cudaGetErrorString(e));
        std::exit(1);
    }
}
static inline int clamp_threads(int bs) {
    if (bs <= 0 || bs > 1024) return 256;
    return bs;
}

// Init weights
static void init_rand(std::vector<float>& v, float scale = 0.05f) {
    static std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto& x : v) x = dist(rng);
}

// Config
struct TrainConfig {
    std::string train_images, train_labels;
    std::string test_images,  test_labels;
    int   epochs    = 1;
    int   batch     = 64;
    float lr        = 0.01f;
    int   blocksize = 256;
};

// Batch maker
static void make_batch_pinned(const MNISTDataset& ds,
                              const std::vector<int>& perm,
                              int start,
                              int B, int H, int W,
                              float* out_x_pinned,
                              int*   out_y_pinned)
{
    for (int i = 0; i < B; ++i) {
        int idx = perm[start + i];
        out_y_pinned[i] = (int)ds.labels[idx];

        const float* src = ds.images.data() + (size_t)idx * H * W;
        float* dst = out_x_pinned + (size_t)i * H * W;
        std::copy(src, src + (H * W), dst);
    }
}

// Accuracy (CPU)
static int argmaxK(const float* x, int K) {
    int best = 0;
    float v = x[0];
    for (int i = 1; i < K; ++i) {
        if (x[i] > v) { v = x[i]; best = i; }
    }
    return best;
}
static float batch_accuracy_logits(const std::vector<float>& h_logits,
                                   const int* h_labels,
                                   int B, int K)
{
    int correct = 0;
    for (int i = 0; i < B; ++i) {
        int pred = argmaxK(&h_logits[(size_t)i * K], K);
        if (pred == h_labels[i]) correct++;
    }
    return (float)correct / (float)B;
}

// scale kernel (mean dLogits)
__global__ void scale_inplace_kernel(float* x, int n, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= alpha;
}
static void scale_inplace(float* d_ptr, int n, float alpha, int threads, cudaStream_t stream) {
    int t = clamp_threads(threads);
    int blocks = (n + t - 1) / t;
    scale_inplace_kernel<<<blocks, t, 0, stream>>>(d_ptr, n, alpha);
    kcheck("scale_inplace_kernel");
}

// zero parameter grads
static inline void zero_param_grads(
        Tensor& d_dw1, Tensor& d_dw2, Tensor& d_dwfc1, Tensor& d_dwfc2,
        Tensor& d_db1, Tensor& d_db2,
        float* d_dbfc1, float* d_dbfc2,
        int C1, int C2, int K, int FC1_OUT, int FC1_IN, int FC2_OUT, int FC2_IN,
        cudaStream_t stream)
{
    ck(cudaMemsetAsync(d_dw1.data,   0, (size_t)C1 * 1  * K * K * sizeof(float), stream), "zero dw1");
    ck(cudaMemsetAsync(d_dw2.data,   0, (size_t)C2 * C1 * K * K * sizeof(float), stream), "zero dw2");
    ck(cudaMemsetAsync(d_dwfc1.data, 0, (size_t)FC1_OUT * FC1_IN * sizeof(float), stream), "zero dwfc1");
    ck(cudaMemsetAsync(d_dwfc2.data, 0, (size_t)FC2_OUT * FC2_IN * sizeof(float), stream), "zero dwfc2");

    ck(cudaMemsetAsync(d_db1.data,   0, (size_t)C1 * sizeof(float), stream), "zero db1");
    ck(cudaMemsetAsync(d_db2.data,   0, (size_t)C2 * sizeof(float), stream), "zero db2");

    ck(cudaMemsetAsync(d_dbfc1,      0, (size_t)FC1_OUT * sizeof(float), stream), "zero dbfc1");
    ck(cudaMemsetAsync(d_dbfc2,      0, (size_t)FC2_OUT * sizeof(float), stream), "zero dbfc2");
}

void train_mnist(const TrainConfig& cfg)
{
    MNISTDataset train_ds = load_mnist_idx(cfg.train_images, cfg.train_labels);
    MNISTDataset test_ds  = load_mnist_idx(cfg.test_images,  cfg.test_labels);
    (void)test_ds;

    const int H = train_ds.rows;
    const int W = train_ds.cols;
    const int B = cfg.batch;
    const int Kcls = 10;

    const int K = 3, pad = 1, stride = 1;
    const int pool = 2, pool_stride = 2;

    const int C1 = 8, C2 = 16;
    const int FC1_OUT = 128, FC2_OUT = 10;

    const int Hc1 = H,  Wc1 = W;          // 28x28
    const int Hp1 = Hc1/2, Wp1 = Wc1/2;   // 14x14
    const int Hc2 = Hp1, Wc2 = Wp1;       // 14x14
    const int Hp2 = Hc2/2, Wp2 = Wc2/2;   // 7x7

    const int FC1_IN = C2 * Hp2 * Wp2;    // 784
    const int FC2_IN = FC1_OUT;

    const int threads = clamp_threads(cfg.blocksize);

    // Host weights
    std::vector<float> h_w1((size_t)C1 * 1  * K * K), h_b1((size_t)C1);
    std::vector<float> h_w2((size_t)C2 * C1 * K * K), h_b2((size_t)C2);
    std::vector<float> h_wfc1((size_t)FC1_OUT * FC1_IN), h_bfc1((size_t)FC1_OUT);
    std::vector<float> h_wfc2((size_t)FC2_OUT * FC2_IN), h_bfc2((size_t)FC2_OUT);

    init_rand(h_w1);   init_rand(h_b1);
    init_rand(h_w2);   init_rand(h_b2);
    init_rand(h_wfc1); init_rand(h_bfc1);
    init_rand(h_wfc2); init_rand(h_bfc2);

    // Device weights
    Tensor d_w1   {nullptr, C1, 1,  K, K};
    Tensor d_w2   {nullptr, C2, C1, K, K};
    Tensor d_wfc1 {nullptr, FC1_OUT, FC1_IN, 1, 1};
    Tensor d_wfc2 {nullptr, FC2_OUT, FC2_IN, 1, 1};

    float *d_b1=nullptr, *d_b2=nullptr, *d_bfc1=nullptr, *d_bfc2=nullptr;

    ck(cudaMalloc(&d_w1.data,   (size_t)C1*1*K*K*sizeof(float)), "malloc w1");
    ck(cudaMalloc(&d_w2.data,   (size_t)C2*C1*K*K*sizeof(float)), "malloc w2");
    ck(cudaMalloc(&d_wfc1.data, (size_t)FC1_OUT*FC1_IN*sizeof(float)), "malloc wfc1");
    ck(cudaMalloc(&d_wfc2.data, (size_t)FC2_OUT*FC2_IN*sizeof(float)), "malloc wfc2");

    ck(cudaMalloc(&d_b1,   (size_t)C1*sizeof(float)), "malloc b1");
    ck(cudaMalloc(&d_b2,   (size_t)C2*sizeof(float)), "malloc b2");
    ck(cudaMalloc(&d_bfc1, (size_t)FC1_OUT*sizeof(float)), "malloc bfc1");
    ck(cudaMalloc(&d_bfc2, (size_t)FC2_OUT*sizeof(float)), "malloc bfc2");

    ck(cudaMemcpy(d_w1.data,   h_w1.data(),   h_w1.size()*sizeof(float), cudaMemcpyHostToDevice), "cpy w1");
    ck(cudaMemcpy(d_b1,        h_b1.data(),   h_b1.size()*sizeof(float), cudaMemcpyHostToDevice), "cpy b1");
    ck(cudaMemcpy(d_w2.data,   h_w2.data(),   h_w2.size()*sizeof(float), cudaMemcpyHostToDevice), "cpy w2");
    ck(cudaMemcpy(d_b2,        h_b2.data(),   h_b2.size()*sizeof(float), cudaMemcpyHostToDevice), "cpy b2");
    ck(cudaMemcpy(d_wfc1.data, h_wfc1.data(), h_wfc1.size()*sizeof(float), cudaMemcpyHostToDevice), "cpy wfc1");
    ck(cudaMemcpy(d_bfc1,      h_bfc1.data(), h_bfc1.size()*sizeof(float), cudaMemcpyHostToDevice), "cpy bfc1");
    ck(cudaMemcpy(d_wfc2.data, h_wfc2.data(), h_wfc2.size()*sizeof(float), cudaMemcpyHostToDevice), "cpy wfc2");
    ck(cudaMemcpy(d_bfc2,      h_bfc2.data(), h_bfc2.size()*sizeof(float), cudaMemcpyHostToDevice), "cpy bfc2");

    // Grads
    Tensor d_dw1   {nullptr, C1, 1,  K, K};
    Tensor d_dw2   {nullptr, C2, C1, K, K};
    Tensor d_dwfc1 {nullptr, FC1_OUT, FC1_IN, 1, 1};
    Tensor d_dwfc2 {nullptr, FC2_OUT, FC2_IN, 1, 1};

    Tensor d_db1 {nullptr, 1, C1, 1, 1};
    Tensor d_db2 {nullptr, 1, C2, 1, 1};
    float *d_dbfc1=nullptr, *d_dbfc2=nullptr;

    ck(cudaMalloc(&d_dw1.data,   (size_t)C1*1*K*K*sizeof(float)), "malloc dw1");
    ck(cudaMalloc(&d_dw2.data,   (size_t)C2*C1*K*K*sizeof(float)), "malloc dw2");
    ck(cudaMalloc(&d_dwfc1.data, (size_t)FC1_OUT*FC1_IN*sizeof(float)), "malloc dwfc1");
    ck(cudaMalloc(&d_dwfc2.data, (size_t)FC2_OUT*FC2_IN*sizeof(float)), "malloc dwfc2");

    ck(cudaMalloc(&d_db1.data, (size_t)C1*sizeof(float)), "malloc db1");
    ck(cudaMalloc(&d_db2.data, (size_t)C2*sizeof(float)), "malloc db2");
    ck(cudaMalloc(&d_dbfc1,    (size_t)FC1_OUT*sizeof(float)), "malloc dbfc1");
    ck(cudaMalloc(&d_dbfc2,    (size_t)FC2_OUT*sizeof(float)), "malloc dbfc2");

    // Activations (ping-pong input)
    Tensor d_X[2] = { {nullptr, B, 1, H, W}, {nullptr, B, 1, H, W} };

    // conv outputs are POST-ReLU because fused forward
    Tensor d_R1 {nullptr, B, C1, Hc1, Wc1};
    Tensor d_P1 {nullptr, B, C1, Hp1, Wp1};

    Tensor d_R2 {nullptr, B, C2, Hc2, Wc2};
    Tensor d_P2 {nullptr, B, C2, Hp2, Wp2};

    Tensor d_FC1    {nullptr, B, FC1_OUT, 1, 1};
    Tensor d_R3     {nullptr, B, FC1_OUT, 1, 1};
    Tensor d_logits {nullptr, B, FC2_OUT, 1, 1};

    ck(cudaMalloc(&d_X[0].data, (size_t)B*H*W*sizeof(float)), "malloc X0");
    ck(cudaMalloc(&d_X[1].data, (size_t)B*H*W*sizeof(float)), "malloc X1");

    ck(cudaMalloc(&d_R1.data, (size_t)B*C1*Hc1*Wc1*sizeof(float)), "malloc R1");
    ck(cudaMalloc(&d_P1.data, (size_t)B*C1*Hp1*Wp1*sizeof(float)), "malloc P1");
    ck(cudaMalloc(&d_R2.data, (size_t)B*C2*Hc2*Wc2*sizeof(float)), "malloc R2");
    ck(cudaMalloc(&d_P2.data, (size_t)B*C2*Hp2*Wp2*sizeof(float)), "malloc P2");

    ck(cudaMalloc(&d_FC1.data,    (size_t)B*FC1_OUT*sizeof(float)), "malloc FC1");
    ck(cudaMalloc(&d_R3.data,     (size_t)B*FC1_OUT*sizeof(float)), "malloc R3");
    ck(cudaMalloc(&d_logits.data, (size_t)B*FC2_OUT*sizeof(float)), "malloc logits");

    int* d_argmax1=nullptr;
    int* d_argmax2=nullptr;
    ck(cudaMalloc(&d_argmax1, (size_t)B*C1*Hp1*Wp1*sizeof(int)), "malloc argmax1");
    ck(cudaMalloc(&d_argmax2, (size_t)B*C2*Hp2*Wp2*sizeof(int)), "malloc argmax2");

    // Backward buffers
    Tensor d_dLogits{nullptr, B, FC2_OUT, 1, 1};
    Tensor d_dR3    {nullptr, B, FC1_OUT, 1, 1};
    Tensor d_dFC1   {nullptr, B, FC1_OUT, 1, 1};

    Tensor d_dP2    {nullptr, B, C2, Hp2, Wp2};
    Tensor d_dR2    {nullptr, B, C2, Hc2, Wc2};

    Tensor d_dP1    {nullptr, B, C1, Hp1, Wp1};
    Tensor d_dR1    {nullptr, B, C1, Hc1, Wc1};

    Tensor d_dX     {nullptr, B, 1, H, W};

    ck(cudaMalloc(&d_dLogits.data, (size_t)B*FC2_OUT*sizeof(float)), "malloc dLogits");
    ck(cudaMalloc(&d_dR3.data,     (size_t)B*FC1_OUT*sizeof(float)), "malloc dR3");
    ck(cudaMalloc(&d_dFC1.data,    (size_t)B*FC1_OUT*sizeof(float)), "malloc dFC1");

    ck(cudaMalloc(&d_dP2.data, (size_t)B*C2*Hp2*Wp2*sizeof(float)), "malloc dP2");
    ck(cudaMalloc(&d_dR2.data, (size_t)B*C2*Hc2*Wc2*sizeof(float)), "malloc dR2");
    ck(cudaMalloc(&d_dP1.data, (size_t)B*C1*Hp1*Wp1*sizeof(float)), "malloc dP1");
    ck(cudaMalloc(&d_dR1.data, (size_t)B*C1*Hc1*Wc1*sizeof(float)), "malloc dR1");
    ck(cudaMalloc(&d_dX.data,  (size_t)B*H*W*sizeof(float)), "malloc dX");

    // correctness patch for fused Conv+ReLU backward
    Tensor d_dC2 {nullptr, B, C2, Hc2, Wc2};
    Tensor d_dC1 {nullptr, B, C1, Hc1, Wc1};
    ck(cudaMalloc(&d_dC2.data, (size_t)B*C2*Hc2*Wc2*sizeof(float)), "malloc dC2");
    ck(cudaMalloc(&d_dC1.data, (size_t)B*C1*Hc1*Wc1*sizeof(float)), "malloc dC1");

    // Labels + Loss
    int*   d_labels[2] = {nullptr, nullptr};
    float* d_loss=nullptr;
    ck(cudaMalloc(&d_labels[0], (size_t)B*sizeof(int)), "malloc labels0");
    ck(cudaMalloc(&d_labels[1], (size_t)B*sizeof(int)), "malloc labels1");
    ck(cudaMalloc(&d_loss, sizeof(float)), "malloc loss");

    // Pinned host buffers
    float* h_Xpin[2] = {nullptr, nullptr};
    int*   h_ypin[2] = {nullptr, nullptr};
    ck(cudaMallocHost(&h_Xpin[0], (size_t)B*H*W*sizeof(float)), "mallocHost Xpin0");
    ck(cudaMallocHost(&h_Xpin[1], (size_t)B*H*W*sizeof(float)), "mallocHost Xpin1");
    ck(cudaMallocHost(&h_ypin[0], (size_t)B*sizeof(int)),       "mallocHost ypin0");
    ck(cudaMallocHost(&h_ypin[1], (size_t)B*sizeof(int)),       "mallocHost ypin1");

    // Streams + Events (CORRECT PING-PONG)
    cudaStream_t s[2];
    ck(cudaStreamCreate(&s[0]), "create stream0");
    ck(cudaStreamCreate(&s[1]), "create stream1");

    cudaEvent_t h2d_done[2];
    cudaEvent_t buf_free[2];

    ck(cudaEventCreateWithFlags(&h2d_done[0], cudaEventDisableTiming), "event h2d_done0");
    ck(cudaEventCreateWithFlags(&h2d_done[1], cudaEventDisableTiming), "event h2d_done1");

    ck(cudaEventCreateWithFlags(&buf_free[0], cudaEventDisableTiming), "event buf_free0");
    ck(cudaEventCreateWithFlags(&buf_free[1], cudaEventDisableTiming), "event buf_free1");

    ck(cudaEventRecord(buf_free[0], s[0]), "record buf_free0 init");
    ck(cudaEventRecord(buf_free[1], s[1]), "record buf_free1 init");

    std::vector<float> h_logits((size_t)B * Kcls);

    // Permutation
    std::vector<int> perm(train_ds.num);
    for (int i = 0; i < train_ds.num; ++i) perm[i] = i;

    auto run_one_step = [&](int buf, cudaStream_t stream, Tensor& Xdev, int* labels_dev,
                            float& out_batch_loss, float& out_batch_acc,
                            bool do_log_copy)
    {
        // Forward
        conv_forward_fused_bias_relu(&Xdev, &d_w1, d_b1, &d_R1, pad, stride, threads, stream);
        kcheck("conv1_fused");

        maxpool_forward(&d_R1, &d_P1, d_argmax1, pool, pool_stride, threads, stream);
        kcheck("pool1");

        conv_forward_fused_bias_relu(&d_P1, &d_w2, d_b2, &d_R2, pad, stride, threads, stream);
        kcheck("conv2_fused");

        maxpool_forward(&d_R2, &d_P2, d_argmax2, pool, pool_stride, threads, stream);
        kcheck("pool2");

        fc_forward(&d_P2, &d_wfc1, d_bfc1, &d_FC1, threads, stream);
        kcheck("fc1");

        relu_forward(d_FC1, d_R3, threads, stream);
        kcheck("relu_fc");

        fc_forward(&d_R3, &d_wfc2, d_bfc2, &d_logits, threads, stream);
        kcheck("fc2");

        // Loss + dLogits
        ck(cudaMemsetAsync(d_loss, 0, sizeof(float), stream), "zero loss");
        compute_softmax_loss_gradient(
                d_logits.data, labels_dev,
                d_dLogits.data, d_loss,
                B, Kcls, threads, stream
        );
        kcheck("softmax_loss");

        // mean over batch
        scale_inplace(d_dLogits.data, B*Kcls, 1.0f/(float)B, threads, stream);

        // zero grads (dW/db)

        zero_param_grads(
                d_dw1, d_dw2, d_dwfc1, d_dwfc2,
                d_db1, d_db2,
                d_dbfc1, d_dbfc2,
                C1, C2, K, FC1_OUT, FC1_IN, FC2_OUT, FC2_IN,
                stream
        );
        // Backward 
        fc_backward(&d_R3, &d_wfc2, &d_dLogits, &d_dR3, &d_dwfc2, d_dbfc2, threads, stream);
        kcheck("fc2_bwd");

        relu_backward(d_FC1, d_dR3, d_dFC1, threads, stream);
        kcheck("relu_fc_bwd");

        fc_backward(&d_P2, &d_wfc1, &d_dFC1, &d_dP2, &d_dwfc1, d_dbfc1, threads, stream);
        kcheck("fc1_bwd");

        // pool2 backward -> gradient wrt post-ReLU conv2 output
        maxpool_backward(d_dP2, d_argmax2, d_dR2, pool, pool, pool_stride, pool_stride, threads, stream);
        kcheck("pool2_bwd");

        // apply ReLU mask for conv2 (post-ReLU output is d_R2)
        relu_backward(d_R2, d_dR2, d_dC2, threads, stream);
        kcheck("relu2_bwd_mask");

        conv_backward(d_P1, d_w2, d_dC2, d_dP1, d_dw2, d_db2, K, stride, pad, threads, stream);
        kcheck("conv2_bwd");

        // pool1 backward -> gradient wrt post-ReLU conv1 output
        maxpool_backward(d_dP1, d_argmax1, d_dR1, pool, pool, pool_stride, pool_stride, threads, stream);
        kcheck("pool1_bwd");

        // apply ReLU mask for conv1
        relu_backward(d_R1, d_dR1, d_dC1, threads, stream);
        kcheck("relu1_bwd_mask");

        conv_backward(Xdev, d_w1, d_dC1, d_dX, d_dw1, d_db1, K, stride, pad, threads, stream);
        kcheck("conv1_bwd");

        // SGD 
        sgd_update_tensor(d_w1,   d_dw1,   cfg.lr, true, threads, stream); kcheck("sgd_w1");
        sgd_update_tensor(d_w2,   d_dw2,   cfg.lr, true, threads, stream); kcheck("sgd_w2");
        sgd_update_tensor(d_wfc1, d_dwfc1, cfg.lr, true, threads, stream); kcheck("sgd_wfc1");
        sgd_update_tensor(d_wfc2, d_dwfc2, cfg.lr, true, threads, stream); kcheck("sgd_wfc2");

        Tensor b1_t   {d_b1,   1, C1,      1, 1};
        Tensor b2_t   {d_b2,   1, C2,      1, 1};
        Tensor bfc1_t {d_bfc1, 1, FC1_OUT, 1, 1};
        Tensor bfc2_t {d_bfc2, 1, FC2_OUT, 1, 1};

        sgd_update_tensor(b1_t, d_db1, cfg.lr, true, threads, stream); kcheck("sgd_b1");
        sgd_update_tensor(b2_t, d_db2, cfg.lr, true, threads, stream); kcheck("sgd_b2");

        Tensor dbfc1_t{d_dbfc1, 1, FC1_OUT, 1, 1};
        Tensor dbfc2_t{d_dbfc2, 1, FC2_OUT, 1, 1};

        sgd_update_tensor(bfc1_t, dbfc1_t, cfg.lr, true, threads, stream); kcheck("sgd_bfc1");
        sgd_update_tensor(bfc2_t, dbfc2_t, cfg.lr, true, threads, stream); kcheck("sgd_bfc2");

        // Pull loss/logits for logging 
        float h_loss_sum = 0.0f;
        ck(cudaMemcpyAsync(&h_loss_sum, d_loss, sizeof(float), cudaMemcpyDeviceToHost, stream), "cpy loss");

        if (do_log_copy) {
            ck(cudaMemcpyAsync(h_logits.data(), d_logits.data, (size_t)B*Kcls*sizeof(float),
                               cudaMemcpyDeviceToHost, stream), "cpy logits");
        }

        // logging sync
        ck(cudaStreamSynchronize(stream), "sync stream (log)");

        out_batch_loss = h_loss_sum / (float)B;
        if (do_log_copy) out_batch_acc = batch_accuracy_logits(h_logits, h_ypin[buf], B, Kcls);
        else out_batch_acc = 0.0f;
    };

    // Train loop 
    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        std::shuffle(perm.begin(), perm.end(), std::mt19937(123 + epoch));

        float epoch_loss = 0.0f;
        float epoch_acc  = 0.0f;
        int steps = 0;

        const int total_steps = train_ds.num / B;

        make_batch_pinned(train_ds, perm, 0, B, H, W, h_Xpin[0], h_ypin[0]);

        ck(cudaStreamWaitEvent(s[0], buf_free[0], 0), "wait buf_free0 (preload)");
        ck(cudaMemcpyAsync(d_X[0].data, h_Xpin[0], (size_t)B*H*W*sizeof(float),
                           cudaMemcpyHostToDevice, s[0]), "H2D X0");
        ck(cudaMemcpyAsync(d_labels[0], h_ypin[0], (size_t)B*sizeof(int),
                           cudaMemcpyHostToDevice, s[0]), "H2D y0");
        ck(cudaEventRecord(h2d_done[0], s[0]), "record h2d_done0");

        for (int step = 0; step < total_steps; ++step) {
            int buf  = step & 1;
            int next = (step + 1) & 1;

            // schedule next batch H2D
            if (step + 1 < total_steps) {
                int next_start = (step + 1) * B;
                make_batch_pinned(train_ds, perm, next_start, B, H, W, h_Xpin[next], h_ypin[next]);

                ck(cudaStreamWaitEvent(s[next], buf_free[next], 0), "wait buf_free next");
                ck(cudaMemcpyAsync(d_X[next].data, h_Xpin[next], (size_t)B*H*W*sizeof(float),
                                   cudaMemcpyHostToDevice, s[next]), "H2D X next");
                ck(cudaMemcpyAsync(d_labels[next], h_ypin[next], (size_t)B*sizeof(int),
                                   cudaMemcpyHostToDevice, s[next]), "H2D y next");
                ck(cudaEventRecord(h2d_done[next], s[next]), "record h2d_done next");
            }

            ck(cudaStreamWaitEvent(s[buf], h2d_done[buf], 0), "wait h2d_done buf");

            bool do_log = ((steps + 1) % 100 == 0);
            float batch_loss = 0.0f, batch_acc = 0.0f;

            run_one_step(buf, s[buf], d_X[buf], d_labels[buf], batch_loss, batch_acc, do_log);

            // mark buffer free again so future H2D can reuse it
            ck(cudaEventRecord(buf_free[buf], s[buf]), "record buf_free buf");

            epoch_loss += batch_loss;
            if (do_log) epoch_acc += batch_acc;
            steps++;

            if (do_log) {
                std::printf("Epoch %d step %d loss=%.4f acc=%.2f%%\n",
                            epoch, steps, batch_loss, batch_acc * 100.0f);
            }
        }

        float avg_loss = epoch_loss / (steps ? steps : 1);
        int acc_samples = steps / 100;
        float avg_acc = (acc_samples > 0) ? (epoch_acc / (float)acc_samples) : 0.0f;

        std::printf("Epoch %d avg loss=%.4f avg acc(sampled)=%.2f%%\n",
                    epoch, avg_loss, avg_acc * 100.0f);

        ck(cudaDeviceSynchronize(), "sync end epoch");
    }

    // Cleanup 
    ck(cudaStreamDestroy(s[0]), "destroy s0");
    ck(cudaStreamDestroy(s[1]), "destroy s1");

    ck(cudaEventDestroy(h2d_done[0]), "destroy h2d_done0");
    ck(cudaEventDestroy(h2d_done[1]), "destroy h2d_done1");
    ck(cudaEventDestroy(buf_free[0]), "destroy buf_free0");
    ck(cudaEventDestroy(buf_free[1]), "destroy buf_free1");

    ck(cudaFreeHost(h_Xpin[0]), "freeHost X0");
    ck(cudaFreeHost(h_Xpin[1]), "freeHost X1");
    ck(cudaFreeHost(h_ypin[0]), "freeHost y0");
    ck(cudaFreeHost(h_ypin[1]), "freeHost y1");

    cudaFree(d_w1.data);   cudaFree(d_w2.data);   cudaFree(d_wfc1.data);  cudaFree(d_wfc2.data);
    cudaFree(d_b1);        cudaFree(d_b2);        cudaFree(d_bfc1);       cudaFree(d_bfc2);

    cudaFree(d_dw1.data);  cudaFree(d_dw2.data);  cudaFree(d_dwfc1.data); cudaFree(d_dwfc2.data);
    cudaFree(d_db1.data);  cudaFree(d_db2.data);  cudaFree(d_dbfc1);      cudaFree(d_dbfc2);

    cudaFree(d_X[0].data); cudaFree(d_X[1].data);

    cudaFree(d_R1.data);   cudaFree(d_P1.data);
    cudaFree(d_R2.data);   cudaFree(d_P2.data);
    cudaFree(d_FC1.data);  cudaFree(d_R3.data);   cudaFree(d_logits.data);

    cudaFree(d_argmax1);   cudaFree(d_argmax2);

    cudaFree(d_dLogits.data); cudaFree(d_dR3.data); cudaFree(d_dFC1.data);
    cudaFree(d_dP2.data);     cudaFree(d_dR2.data);
    cudaFree(d_dP1.data);     cudaFree(d_dR1.data);
    cudaFree(d_dX.data);

    cudaFree(d_dC2.data);
    cudaFree(d_dC1.data);

    cudaFree(d_labels[0]);
    cudaFree(d_labels[1]);
    cudaFree(d_loss);
}
