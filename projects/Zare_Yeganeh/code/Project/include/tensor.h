#ifndef TENSOR_H
#define TENSOR_H

#pragma once

struct Tensor {
    float* data;
    int N, C, H, W;
};

static inline int numel(const Tensor& t) {
return t.N * t.C * t.H * t.W;
}
#endif