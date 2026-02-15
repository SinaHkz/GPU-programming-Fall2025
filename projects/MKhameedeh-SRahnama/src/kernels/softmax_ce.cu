#include "gpu/kernels/softmax_ce.h"

#include <cuda_runtime.h>

#include "gpu/core/profiler.h"
#include "gpu/utils/cuda_check.h"
#include "gpu/utils/cuda_profiled.h"

namespace gpu {

namespace {

struct TmpBuffers {
  int* labels{nullptr};
  int* correct{nullptr};
  float* loss{nullptr};
  int labels_cap{0};
  int correct_cap{0};
  int loss_cap{0};

  ~TmpBuffers() {
    if (labels) cudaFree(labels);
    if (correct) cudaFree(correct);
    if (loss) cudaFree(loss);
  }

  int* ensure_labels(int n) {
    if (n <= labels_cap) return labels;
    if (labels) cudaFree(labels);
    GPU_CUDA_CHECK(cudaMalloc(&labels, n * sizeof(int)));
    labels_cap = n;
    return labels;
  }

  float* ensure_loss(int n) {
    if (n <= loss_cap) return loss;
    if (loss) cudaFree(loss);
    GPU_CUDA_CHECK(cudaMalloc(&loss, n * sizeof(float)));
    loss_cap = n;
    return loss;
  }

  int* ensure_correct(int n) {
    if (n <= correct_cap) return correct;
    if (correct) cudaFree(correct);
    GPU_CUDA_CHECK(cudaMalloc(&correct, n * sizeof(int)));
    correct_cap = n;
    return correct;
  }
};

TmpBuffers& tmp() {
  static TmpBuffers t;
  return t;
}

}  // namespace

__global__ void softmax_ce_bwd_kernel(const float* logits, const int* labels, float* grad, float* loss, int n,
                                     int classes) {
  const int sample = blockIdx.x;
  if (sample >= n) return;

  __shared__ float maxv;
  __shared__ float denom;

  if (threadIdx.x == 0) {
    float m = -1e20f;
    for (int c = 0; c < classes; ++c) m = fmaxf(m, logits[sample * classes + c]);
    maxv = m;

    float s = 0.0f;
    for (int c = 0; c < classes; ++c) s += expf(logits[sample * classes + c] - maxv);
    denom = s;

    if (loss) {
      const int y = labels[sample];
      const float p = expf(logits[sample * classes + y] - maxv) / denom;
      loss[sample] = -logf(fmaxf(p, 1e-12f));
    }
  }
  __syncthreads();

  // Gradient: (p - onehot)/n
  const int y = labels[sample];
  for (int c = threadIdx.x; c < classes; c += blockDim.x) {
    const float p = expf(logits[sample * classes + c] - maxv) / denom;
    const float g = (p - (c == y ? 1.0f : 0.0f)) / static_cast<float>(n);
    grad[sample * classes + c] = g;
  }
}

__global__ void argmax_acc_kernel(const float* logits, const int* labels, int* correct, int n, int classes) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int best = 0;
  float bestv = logits[i * classes];
  for (int c = 1; c < classes; ++c) {
    const float v = logits[i * classes + c];
    if (v > bestv) {
      bestv = v;
      best = c;
    }
  }
  correct[i] = (best == labels[i]) ? 1 : 0;
}

float softmax_cross_entropy_backward(const Tensor& logits, const std::vector<int>& labels, Tensor& grad_logits,
                                     bool compute_loss) {
  Profiler::ScopedGpuTimer t("softmax_cross_entropy_backward");
  const int n = logits.dim(0);
  const int classes = logits.dim(1);
  if (static_cast<int>(labels.size()) != n) return 0.0f;

  int* d_labels = tmp().ensure_labels(n);
  float* d_loss = tmp().ensure_loss(n);
  GPU_CUDA_CHECK(cudaMemcpyProfiled(d_labels, labels.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  const int threads = 256;
  GPU_CUDA_CHECK(cudaMemset(d_loss, 0, n * sizeof(float)));
  Profiler::instance().record_kernel_launch("softmax_ce_bwd_kernel", (const void*)softmax_ce_bwd_kernel, dim3(n),
                                            dim3(threads));
  softmax_ce_bwd_kernel<<<n, threads>>>(logits.data(), d_labels, grad_logits.data(), d_loss, n, classes);
  GPU_CUDA_CHECK(cudaGetLastError());
  if (!compute_loss) return 0.0f;

  std::vector<float> host_loss(static_cast<size_t>(n));
  GPU_CUDA_CHECK(cudaMemcpyProfiled(host_loss.data(), d_loss, n * sizeof(float), cudaMemcpyDeviceToHost));
  float sum = 0.0f;
  for (float v : host_loss) sum += v;
  return sum / static_cast<float>(n);
}

int argmax_accuracy(const Tensor& logits, const std::vector<int>& labels) {
  Profiler::ScopedGpuTimer t("argmax_accuracy");
  const int n = logits.dim(0);
  const int classes = logits.dim(1);
  if (static_cast<int>(labels.size()) != n) return 0;

  int* d_labels = tmp().ensure_labels(n);
  int* d_correct = tmp().ensure_correct(n);
  GPU_CUDA_CHECK(cudaMemcpyProfiled(d_labels, labels.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  Profiler::instance().record_kernel_launch("argmax_acc_kernel", (const void*)argmax_acc_kernel, dim3((n + 255) / 256),
                                            dim3(256));
  argmax_acc_kernel<<<(n + 255) / 256, 256>>>(logits.data(), d_labels, d_correct, n, classes);
  GPU_CUDA_CHECK(cudaGetLastError());

  std::vector<int> host_correct(static_cast<size_t>(n));
  GPU_CUDA_CHECK(cudaMemcpyProfiled(host_correct.data(), d_correct, n * sizeof(int), cudaMemcpyDeviceToHost));
  int sum = 0;
  for (int v : host_correct) sum += v;
  return sum;
}

}  // namespace gpu
