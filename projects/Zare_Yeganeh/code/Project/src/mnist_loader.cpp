#include "mnist_loader.h"
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <iostream>

static uint32_t read_u32_be(std::ifstream& f) {
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    if (!f) throw std::runtime_error("Failed to read 4 bytes (file too short?)");
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

MNISTDataset load_mnist_idx(const std::string& image_path,
                            const std::string& label_path)
{
    MNISTDataset ds;

    std::ifstream fi(image_path, std::ios::binary);
    if (!fi) throw std::runtime_error("Cannot open image file: " + image_path);

    std::ifstream fl(label_path, std::ios::binary);
    if (!fl) throw std::runtime_error("Cannot open label file: " + label_path);

    uint32_t magic_img = read_u32_be(fi);
    uint32_t num_img   = read_u32_be(fi);
    uint32_t rows      = read_u32_be(fi);
    uint32_t cols      = read_u32_be(fi);

    uint32_t magic_lbl = read_u32_be(fl);
    uint32_t num_lbl   = read_u32_be(fl);

    std::cout << "[MNIST] images: magic=" << magic_img
              << " num=" << num_img << " rows=" << rows << " cols=" << cols << "\n";
    std::cout << "[MNIST] labels: magic=" << magic_lbl
              << " num=" << num_lbl << "\n";

    if (magic_img != 2051) throw std::runtime_error("Bad image magic (expected 2051). File is not MNIST images.");
    if (magic_lbl != 2049) throw std::runtime_error("Bad label magic (expected 2049). File is not MNIST labels.");
    if (num_img != num_lbl) throw std::runtime_error("Count mismatch: num_images != num_labels");
    if (rows == 0 || cols == 0) throw std::runtime_error("Invalid image dimensions");

    ds.num  = (int)num_img;
    ds.rows = (int)rows;
    ds.cols = (int)cols;

    ds.labels.resize(ds.num);
    fl.read(reinterpret_cast<char*>(ds.labels.data()), ds.num);
    if (!fl) throw std::runtime_error("Failed reading labels (file too short?)");

    size_t img_bytes = (size_t)ds.num * ds.rows * ds.cols;
    std::vector<uint8_t> tmp(img_bytes);
    fi.read(reinterpret_cast<char*>(tmp.data()), (std::streamsize)tmp.size());
    if (!fi) throw std::runtime_error("Failed reading images (file too short?)");

    ds.images.resize(img_bytes);
    for (size_t i = 0; i < img_bytes; ++i) ds.images[i] = tmp[i] / 255.0f;

    return ds;
}
