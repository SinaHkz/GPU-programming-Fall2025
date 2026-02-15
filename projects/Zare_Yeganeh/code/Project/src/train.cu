// src/train.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <cstdlib>

#include "tensor.h"
#include "conv_layer.h"
//#include "conv_tile.h"
#include "pool_layer.h"
#include "fc_layer.h"
#include "activations.h"
#include "optimizer.h"
#include "loss.h"
#include "mnist_loader.h"


static void ck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::printf("CUDA error: %s | %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}
static void kcheck(const char* where) {
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
static void make_batch(const MNISTDataset& ds,
                       const std::vector<int>& perm,
                       int start,
                       int B, int H, int W,
                       std::vector<float>& out_x,
                       std::vector<int>& out_y)
{
    out_x.resize((size_t)B * H * W);
    out_y.resize((size_t)B);

    for (int i = 0; i < B; ++i) {
        int idx = perm[start + i];
        out_y[i] = (int)ds.labels[idx];
        const float* src = ds.images.data() + (size_t)idx * H * W;
        float* dst = out_x.data() + (size_t)i * H * W;
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
                                   const std::vector<int>& h_labels,
                                   int B, int K)
{
    int correct = 0;
    for (int i = 0; i < B; ++i) {
        int pred = argmaxK(&h_logits[(size_t)i * K], K);
        if (pred == h_labels[i]) correct++;
    }
    return (float)correct / (float)B;
}

// scale kernel (for dLogits mean) 
__global__ void scale_inplace_kernel(float* x, int n, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= alpha;
}
static void scale_inplace(float* d_ptr, int n, float alpha, int threads) {
    int t = clamp_threads(threads);
    int blocks = (n + t - 1) / t;
    scale_inplace_kernel<<<blocks, t>>>(d_ptr, n, alpha);
    kcheck("scale_inplace_kernel");
}

// zero parameter grads
static inline void zero_param_grads(
        Tensor& d_dw1, Tensor& d_dw2, Tensor& d_dwfc1, Tensor& d_dwfc2,
        Tensor& d_db1, Tensor& d_db2,
        float* d_dbfc1, float* d_dbfc2)
{
    ck(cudaMemset(d_dw1.data,   0, (size_t)numel(d_dw1)   * sizeof(float)), "zero dw1");
    ck(cudaMemset(d_dw2.data,   0, (size_t)numel(d_dw2)   * sizeof(float)), "zero dw2");
    ck(cudaMemset(d_dwfc1.data, 0, (size_t)numel(d_dwfc1) * sizeof(float)), "zero dwfc1");
    ck(cudaMemset(d_dwfc2.data, 0, (size_t)numel(d_dwfc2) * sizeof(float)), "zero dwfc2");

    ck(cudaMemset(d_db1.data,   0, (size_t)numel(d_db1)   * sizeof(float)), "zero db1");
    ck(cudaMemset(d_db2.data,   0, (size_t)numel(d_db2)   * sizeof(float)), "zero db2");

    ck(cudaMemset(d_dbfc1,      0, (size_t)128 * sizeof(float)), "zero dbfc1");
    ck(cudaMemset(d_dbfc2,      0, (size_t)10  * sizeof(float)), "zero dbfc2");
}
//evaluate
static float evaluate_mnist(
        const MNISTDataset& test_ds,
        int B, int H, int W, int Kcls,
        const Tensor& d_w1, const float* d_b1,
        const Tensor& d_w2, const float* d_b2,
        const Tensor& d_wfc1, const float* d_bfc1,
        const Tensor& d_wfc2, const float* d_bfc2,
        Tensor& d_X,
        Tensor& d_C1o, Tensor& d_R1, Tensor& d_P1,
        Tensor& d_C2o, Tensor& d_R2, Tensor& d_P2,
        Tensor& d_FC1, Tensor& d_R3, Tensor& d_logits,
        int* d_argmax1, int* d_argmax2,
        int pad, int stride, int pool, int pool_stride,
        int threads
){
    std::vector<int> perm(test_ds.num);
    for (int i = 0; i < test_ds.num; ++i) perm[i] = i;

    std::vector<float> h_batch_x;
    std::vector<int>   h_batch_y;
    std::vector<float> h_logits((size_t)B * Kcls);

    float acc_sum = 0.0f;
    int steps = 0;

    for (int start = 0; start + B <= test_ds.num; start += B) {
        make_batch(test_ds, perm, start, B, H, W, h_batch_x, h_batch_y);

        ck(cudaMemcpy(d_X.data, h_batch_x.data(), (size_t)B*H*W*sizeof(float),
                      cudaMemcpyHostToDevice), "eval: cpy X");

        // Forward only
        conv_forward(&d_X,  &d_w1, d_b1, &d_C1o, pad, stride, threads); kcheck("eval conv1");
        relu_forward(d_C1o, d_R1, threads);                             kcheck("eval relu1");
        maxpool_forward(&d_R1, &d_P1, d_argmax1, pool, pool_stride, threads); kcheck("eval pool1");

        conv_forward(&d_P1, &d_w2, d_b2, &d_C2o, pad, stride, threads); kcheck("eval conv2");
        relu_forward(d_C2o, d_R2, threads);                             kcheck("eval relu2");
        maxpool_forward(&d_R2, &d_P2, d_argmax2, pool, pool_stride, threads); kcheck("eval pool2");

        fc_forward(&d_P2, &d_wfc1, d_bfc1, &d_FC1, threads);            kcheck("eval fc1");
        relu_forward(d_FC1, d_R3, threads);                             kcheck("eval relu3");
        fc_forward(&d_R3, &d_wfc2, d_bfc2, &d_logits, threads);         kcheck("eval fc2");

        ck(cudaMemcpy(h_logits.data(), d_logits.data, (size_t)B*Kcls*sizeof(float),
                      cudaMemcpyDeviceToHost), "eval: cpy logits");

        float acc = batch_accuracy_logits(h_logits, h_batch_y, B, Kcls);

        acc_sum += acc;
        steps++;
    }

    return acc_sum / (steps ? steps : 1);
}

void train_mnist(const TrainConfig& cfg)
{
    MNISTDataset train_ds = load_mnist_idx(cfg.train_images, cfg.train_labels);
    MNISTDataset test_ds  = load_mnist_idx(cfg.test_images,  cfg.test_labels);
    (void)test_ds;

    const int H = train_ds.rows;   // 28
    const int W = train_ds.cols;   // 28
    const int B = cfg.batch;       // batch
    const int Kcls = 10;

    const int K = 3, pad = 1, stride = 1;
    const int pool = 2, pool_stride = 2;
    const int C1 = 8, C2 = 16;
    const int FC1_OUT = 128, FC2_OUT = 10;

    const int Hc1 = H, Wc1 = W;           // 28x28
    const int Hp1 = Hc1 / 2, Wp1 = Wc1/2; // 14x14
    const int Hc2 = Hp1, Wc2 = Wp1;       // 14x14
    const int Hp2 = Hc2 / 2, Wp2 = Wc2/2; // 7x7

    const int FC1_IN = C2 * Hp2 * Wp2;    // 16*7*7=784
    const int FC2_IN = FC1_OUT;           // 128

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
    Tensor d_w1   {nullptr, C1, 1,    K, K};
    Tensor d_w2   {nullptr, C2, C1,   K, K};
    Tensor d_wfc1 {nullptr, FC1_OUT, FC1_IN, 1, 1};
    Tensor d_wfc2 {nullptr, FC2_OUT, FC2_IN, 1, 1};

    float *d_b1=nullptr, *d_b2=nullptr, *d_bfc1=nullptr, *d_bfc2=nullptr;

    ck(cudaMalloc(&d_w1.data,   (size_t)numel(d_w1)*sizeof(float)), "malloc w1");
    ck(cudaMalloc(&d_w2.data,   (size_t)numel(d_w2)*sizeof(float)), "malloc w2");
    ck(cudaMalloc(&d_wfc1.data, (size_t)numel(d_wfc1)*sizeof(float)), "malloc wfc1");
    ck(cudaMalloc(&d_wfc2.data, (size_t)numel(d_wfc2)*sizeof(float)), "malloc wfc2");

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
    Tensor d_dw1   {nullptr, C1, 1,    K, K};
    Tensor d_dw2   {nullptr, C2, C1,   K, K};
    Tensor d_dwfc1 {nullptr, FC1_OUT, FC1_IN, 1, 1};
    Tensor d_dwfc2 {nullptr, FC2_OUT, FC2_IN, 1, 1};

    Tensor d_db1 {nullptr, 1, C1, 1, 1};
    Tensor d_db2 {nullptr, 1, C2, 1, 1};
    float *d_dbfc1=nullptr, *d_dbfc2=nullptr;

    ck(cudaMalloc(&d_dw1.data,   (size_t)numel(d_dw1)*sizeof(float)), "malloc dw1");
    ck(cudaMalloc(&d_dw2.data,   (size_t)numel(d_dw2)*sizeof(float)), "malloc dw2");
    ck(cudaMalloc(&d_dwfc1.data, (size_t)numel(d_dwfc1)*sizeof(float)), "malloc dwfc1");
    ck(cudaMalloc(&d_dwfc2.data, (size_t)numel(d_dwfc2)*sizeof(float)), "malloc dwfc2");

    ck(cudaMalloc(&d_db1.data, (size_t)C1*sizeof(float)), "malloc db1");
    ck(cudaMalloc(&d_db2.data, (size_t)C2*sizeof(float)), "malloc db2");
    ck(cudaMalloc(&d_dbfc1,    (size_t)FC1_OUT*sizeof(float)), "malloc dbfc1");
    ck(cudaMalloc(&d_dbfc2,    (size_t)FC2_OUT*sizeof(float)), "malloc dbfc2");

    // Activations 
    Tensor d_X     {nullptr, B, 1, H, W};

    Tensor d_C1o   {nullptr, B, C1, Hc1, Wc1};
    Tensor d_R1    {nullptr, B, C1, Hc1, Wc1};
    Tensor d_P1    {nullptr, B, C1, Hp1, Wp1};

    Tensor d_C2o   {nullptr, B, C2, Hc2, Wc2};
    Tensor d_R2    {nullptr, B, C2, Hc2, Wc2};
    Tensor d_P2    {nullptr, B, C2, Hp2, Wp2};

    Tensor d_FC1   {nullptr, B, FC1_OUT, 1, 1};
    Tensor d_R3    {nullptr, B, FC1_OUT, 1, 1};
    Tensor d_logits{nullptr, B, FC2_OUT, 1, 1};

    ck(cudaMalloc(&d_X.data,      (size_t)numel(d_X)*sizeof(float)), "malloc X");

    ck(cudaMalloc(&d_C1o.data,    (size_t)numel(d_C1o)*sizeof(float)), "malloc C1o");
    ck(cudaMalloc(&d_R1.data,     (size_t)numel(d_R1)*sizeof(float)), "malloc R1");
    ck(cudaMalloc(&d_P1.data,     (size_t)numel(d_P1)*sizeof(float)), "malloc P1");

    ck(cudaMalloc(&d_C2o.data,    (size_t)numel(d_C2o)*sizeof(float)), "malloc C2o");
    ck(cudaMalloc(&d_R2.data,     (size_t)numel(d_R2)*sizeof(float)), "malloc R2");
    ck(cudaMalloc(&d_P2.data,     (size_t)numel(d_P2)*sizeof(float)), "malloc P2");

    ck(cudaMalloc(&d_FC1.data,    (size_t)numel(d_FC1)*sizeof(float)), "malloc FC1");
    ck(cudaMalloc(&d_R3.data,     (size_t)numel(d_R3)*sizeof(float)), "malloc R3");
    ck(cudaMalloc(&d_logits.data, (size_t)numel(d_logits)*sizeof(float)), "malloc logits");

    int* d_argmax1=nullptr;
    int* d_argmax2=nullptr;
    ck(cudaMalloc(&d_argmax1, (size_t)numel(d_P1)*sizeof(int)), "malloc argmax1");
    ck(cudaMalloc(&d_argmax2, (size_t)numel(d_P2)*sizeof(int)), "malloc argmax2");

    // Backward buffers 
    Tensor d_dLogits{nullptr, B, FC2_OUT, 1, 1};
    Tensor d_dR3    {nullptr, B, FC1_OUT, 1, 1};
    Tensor d_dFC1   {nullptr, B, FC1_OUT, 1, 1};

    Tensor d_dP2    {nullptr, B, C2, Hp2, Wp2};
    Tensor d_dR2    {nullptr, B, C2, Hc2, Wc2};
    Tensor d_dC2o   {nullptr, B, C2, Hc2, Wc2};

    Tensor d_dP1    {nullptr, B, C1, Hp1, Wp1};
    Tensor d_dR1    {nullptr, B, C1, Hc1, Wc1};
    Tensor d_dC1o   {nullptr, B, C1, Hc1, Wc1};

    Tensor d_dX     {nullptr, B, 1, H, W};

    ck(cudaMalloc(&d_dLogits.data, (size_t)numel(d_dLogits)*sizeof(float)), "malloc dLogits");
    ck(cudaMalloc(&d_dR3.data,     (size_t)numel(d_dR3)*sizeof(float)), "malloc dR3");
    ck(cudaMalloc(&d_dFC1.data,    (size_t)numel(d_dFC1)*sizeof(float)), "malloc dFC1");

    ck(cudaMalloc(&d_dP2.data,     (size_t)numel(d_dP2)*sizeof(float)), "malloc dP2");
    ck(cudaMalloc(&d_dR2.data,     (size_t)numel(d_dR2)*sizeof(float)), "malloc dR2");
    ck(cudaMalloc(&d_dC2o.data,    (size_t)numel(d_dC2o)*sizeof(float)), "malloc dC2o");

    ck(cudaMalloc(&d_dP1.data,     (size_t)numel(d_dP1)*sizeof(float)), "malloc dP1");
    ck(cudaMalloc(&d_dR1.data,     (size_t)numel(d_dR1)*sizeof(float)), "malloc dR1");
    ck(cudaMalloc(&d_dC1o.data,    (size_t)numel(d_dC1o)*sizeof(float)), "malloc dC1o");

    ck(cudaMalloc(&d_dX.data,      (size_t)numel(d_dX)*sizeof(float)), "malloc dX");

    // Labels + Loss 
    int*   d_labels=nullptr;
    float* d_loss=nullptr; // scalar
    ck(cudaMalloc(&d_labels, (size_t)B*sizeof(int)), "malloc labels");
    ck(cudaMalloc(&d_loss,   sizeof(float)), "malloc loss scalar");

    // CPU buffers
    std::vector<int> perm(train_ds.num);
    for (int i = 0; i < train_ds.num; ++i) perm[i] = i;

    std::vector<float> h_batch_x;
    std::vector<int>   h_batch_y;
    std::vector<float> h_logits((size_t)B * Kcls);

    // Train loop 
    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        std::shuffle(perm.begin(), perm.end(), std::mt19937(123 + epoch));

        float epoch_loss = 0.0f;
        float epoch_acc  = 0.0f;
        int steps = 0;

        for (int start = 0; start + B <= train_ds.num; start += B) {
            make_batch(train_ds, perm, start, B, H, W, h_batch_x, h_batch_y);

            ck(cudaMemcpy(d_X.data, h_batch_x.data(), (size_t)numel(d_X)*sizeof(float),
                          cudaMemcpyHostToDevice), "cpy X");
            ck(cudaMemcpy(d_labels, h_batch_y.data(), (size_t)B*sizeof(int),
                          cudaMemcpyHostToDevice), "cpy labels");

            // Forward 
            conv_forward(&d_X,  &d_w1, d_b1, &d_C1o, pad, stride, threads); kcheck("conv1_fwd");
            relu_forward(d_C1o, d_R1, threads);                             kcheck("relu1");
            maxpool_forward(&d_R1, &d_P1, d_argmax1, pool, pool_stride, threads); kcheck("pool1");

            conv_forward(&d_P1, &d_w2, d_b2, &d_C2o, pad, stride, threads); kcheck("conv2_fwd");
            relu_forward(d_C2o, d_R2, threads);                             kcheck("relu2");
            maxpool_forward(&d_R2, &d_P2, d_argmax2, pool, pool_stride, threads); kcheck("pool2");

            fc_forward(&d_P2, &d_wfc1, d_bfc1, &d_FC1, threads);            kcheck("fc1_fwd");
            relu_forward(d_FC1, d_R3, threads);                             kcheck("relu3");
            fc_forward(&d_R3, &d_wfc2, d_bfc2, &d_logits, threads);         kcheck("fc2_fwd");

            // Accuracy 
            ck(cudaMemcpy(h_logits.data(), d_logits.data, (size_t)B*Kcls*sizeof(float),
                          cudaMemcpyDeviceToHost), "cpy logits");
            float acc = batch_accuracy_logits(h_logits, h_batch_y, B, Kcls);

            // Loss + dLogits 
            ck(cudaMemset(d_loss, 0, sizeof(float)), "zero loss scalar");

            compute_softmax_loss_gradient(
                    d_logits.data, d_labels,
                    d_dLogits.data, d_loss,
                    B, Kcls, threads
            );
            ck(cudaDeviceSynchronize(), "sync after softmax_loss");
            kcheck("softmax_loss");

            // Make dLogits MEAN
            scale_inplace(d_dLogits.data, B * Kcls, 1.0f / (float)B, threads);
            ck(cudaDeviceSynchronize(), "sync after scale_inplace");
            kcheck("scale_inplace");

            float h_loss_sum = 0.0f;
            ck(cudaMemcpy(&h_loss_sum, d_loss, sizeof(float),
                          cudaMemcpyDeviceToHost), "copy loss scalar");
            float batch_loss = h_loss_sum / (float)B;

            epoch_loss += batch_loss;
            epoch_acc  += acc;
            steps++;

            // ZERO grads BEFORE backward 
            zero_param_grads(d_dw1, d_dw2, d_dwfc1, d_dwfc2, d_db1, d_db2, d_dbfc1, d_dbfc2);

            // Backward 
            fc_backward(&d_R3, &d_wfc2, &d_dLogits, &d_dR3,  &d_dwfc2, d_dbfc2, threads); kcheck("fc2_bwd");
            relu_backward(d_FC1, d_dR3, d_dFC1, threads);                                 kcheck("relu3_bwd");
            fc_backward(&d_P2, &d_wfc1, &d_dFC1, &d_dP2,  &d_dwfc1, d_dbfc1, threads);    kcheck("fc1_bwd");

            maxpool_backward(d_dP2, d_argmax2, d_dR2, pool, pool, pool_stride, pool_stride, threads); kcheck("pool2_bwd");
            relu_backward(d_C2o, d_dR2, d_dC2o, threads);                                             kcheck("relu2_bwd");
            conv_backward(d_P1, d_w2, d_dC2o, d_dP1, d_dw2, d_db2, K, stride, pad, threads);           kcheck("conv2_bwd");

            maxpool_backward(d_dP1, d_argmax1, d_dR1, pool, pool, pool_stride, pool_stride, threads); kcheck("pool1_bwd");
            relu_backward(d_C1o, d_dR1, d_dC1o, threads);                                             kcheck("relu1_bwd");
            conv_backward(d_X, d_w1, d_dC1o, d_dX, d_dw1, d_db1, K, stride, pad, threads);             kcheck("conv1_bwd");

            // SGD 
            sgd_update_tensor(d_w1,   d_dw1,   cfg.lr, true, threads); kcheck("sgd_w1");
            sgd_update_tensor(d_w2,   d_dw2,   cfg.lr, true, threads); kcheck("sgd_w2");
            sgd_update_tensor(d_wfc1, d_dwfc1, cfg.lr, true, threads); kcheck("sgd_wfc1");
            sgd_update_tensor(d_wfc2, d_dwfc2, cfg.lr, true, threads); kcheck("sgd_wfc2");

            Tensor b1_t   {d_b1,   1, C1,      1, 1};
            Tensor b2_t   {d_b2,   1, C2,      1, 1};
            Tensor bfc1_t {d_bfc1, 1, FC1_OUT, 1, 1};
            Tensor bfc2_t {d_bfc2, 1, FC2_OUT, 1, 1};

            sgd_update_tensor(b1_t, d_db1, cfg.lr, true, threads); kcheck("sgd_b1");
            sgd_update_tensor(b2_t, d_db2, cfg.lr, true, threads); kcheck("sgd_b2");

            Tensor dbfc1_t{d_dbfc1, 1, FC1_OUT, 1, 1};
            Tensor dbfc2_t{d_dbfc2, 1, FC2_OUT, 1, 1};

            sgd_update_tensor(bfc1_t, dbfc1_t, cfg.lr, true, threads); kcheck("sgd_bfc1");
            sgd_update_tensor(bfc2_t, dbfc2_t, cfg.lr, true, threads); kcheck("sgd_bfc2");

            ck(cudaDeviceSynchronize(), "sync step");

            if (steps % 100 == 0) {
                std::printf("Epoch %d step %d loss=%.4f acc=%.2f%%\n",
                            epoch, steps, batch_loss, acc * 100.0f);
            }
        }

        float avg_loss = epoch_loss / (steps ? steps : 1);
        float avg_acc  = epoch_acc  / (steps ? steps : 1);
        std::printf("Epoch %d avg loss=%.4f avg acc=%.2f%%\n",
                    epoch, avg_loss, avg_acc * 100.0f);

        float test_acc = evaluate_mnist(
                test_ds,
                B, H, W, Kcls,
                d_w1, d_b1,
                d_w2, d_b2,
                d_wfc1, d_bfc1,
                d_wfc2, d_bfc2,
                d_X,
                d_C1o, d_R1, d_P1,
                d_C2o, d_R2, d_P2,
                d_FC1, d_R3, d_logits,
                d_argmax1, d_argmax2,
                pad, stride, pool, pool_stride,
                threads
        );
        std::printf("Epoch %d TEST acc=%.2f%%\n", epoch, test_acc * 100.0f);

    }


    // Free 
    cudaFree(d_w1.data);   cudaFree(d_w2.data);   cudaFree(d_wfc1.data); cudaFree(d_wfc2.data);
    cudaFree(d_b1);        cudaFree(d_b2);        cudaFree(d_bfc1);      cudaFree(d_bfc2);

    cudaFree(d_dw1.data);  cudaFree(d_dw2.data);  cudaFree(d_dwfc1.data); cudaFree(d_dwfc2.data);
    cudaFree(d_db1.data);  cudaFree(d_db2.data);  cudaFree(d_dbfc1);      cudaFree(d_dbfc2);

    cudaFree(d_X.data);
    cudaFree(d_C1o.data);  cudaFree(d_R1.data);   cudaFree(d_P1.data);
    cudaFree(d_C2o.data);  cudaFree(d_R2.data);   cudaFree(d_P2.data);
    cudaFree(d_FC1.data);  cudaFree(d_R3.data);   cudaFree(d_logits.data);

    cudaFree(d_argmax1);   cudaFree(d_argmax2);

    cudaFree(d_dLogits.data); cudaFree(d_dR3.data); cudaFree(d_dFC1.data);
    cudaFree(d_dP2.data);     cudaFree(d_dR2.data); cudaFree(d_dC2o.data);
    cudaFree(d_dP1.data);     cudaFree(d_dR1.data); cudaFree(d_dC1o.data);
    cudaFree(d_dX.data);

    cudaFree(d_labels);
    cudaFree(d_loss);
}
