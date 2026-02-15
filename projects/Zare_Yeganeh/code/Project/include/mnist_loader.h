#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct MNISTDataset {
    int num = 0;
    int rows = 0;
    int cols = 0;
    std::vector<float> images;     // num * rows * cols in [0,1]
    std::vector<uint8_t> labels;
};

MNISTDataset load_mnist_idx(const std::string& image_path,
                            const std::string& label_path);

#endif