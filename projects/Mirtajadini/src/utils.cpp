#include "utils.h"
#include "config.h"
#include <iostream>
#include <fstream>
#include <random>
#include <io.h> 
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" 

const char* STL10_CLASS_NAMES[5] = {
    "Airplane", "Bird", "Car", "Dog", "Cat"
};

void load_or_generate_weights(const std::string& filename, std::vector<float>& vec, int size) {
    vec.resize(size);
    std::ifstream file(filename, std::ios::binary);

    if (file.is_open()) {
        file.seekg(0, std::ios::end);
        size_t fsize = file.tellg();
        file.seekg(0, std::ios::beg);
        if (fsize == size * sizeof(float)) {
            file.read(reinterpret_cast<char*>(vec.data()), fsize);
            printf("Loaded %s (%zu bytes)\n", filename.c_str(), fsize);
            return;
        }
        else {
            printf("WARNING: Size mismatch for %s. Expected %d floats.\n", filename.c_str(), size);
        }
    }
    printf("Generating RANDOM weights for %s...\n", filename.c_str());
    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dis(-0.05f, 0.05f);
    for (int i = 0; i < size; ++i) vec[i] = dis(gen);
}

std::string find_dataset_folder() {
    std::vector<std::string> candidates = { "STL_image", "STL_test", "../STL_test", "../../STL_test", "C:/STL_test" };
    for (const auto& path : candidates) {
        std::string check = path + "/*.png";
        struct _finddata_t c_file;
        intptr_t hFile = _findfirst(check.c_str(), &c_file);
        if (hFile != -1L) {
            _findclose(hFile);
            return path;
        }
    }
    return "";
}

int load_dataset(const std::string& folder_path, std::vector<unsigned char>& host_input, std::vector<std::string>& filenames) {
    if (folder_path.empty()) return 0;
    std::string search_mask = folder_path + "\\*.png";
    struct _finddata_t c_file;
    intptr_t hFile;
    int count = 0;
    if ((hFile = _findfirst(search_mask.c_str(), &c_file)) != -1L) {
        do { filenames.push_back(folder_path + "\\" + c_file.name); count++; } while (_findnext(hFile, &c_file) == 0);
        _findclose(hFile);
    }
    if (count == 0) return 0;

    printf("Found %d images in '%s'\n", count, folder_path.c_str());
    host_input.resize(count * INPUT_DIM);
    for (int i = 0; i < count; ++i) {
        int w, h, c;
        unsigned char* img = stbi_load(filenames[i].c_str(), &w, &h, &c, 3);
        if (img) { memcpy(&host_input[i * INPUT_DIM], img, INPUT_DIM); stbi_image_free(img); }
        else { memset(&host_input[i * INPUT_DIM], 0, INPUT_DIM); }
    }
    return count;
}

float calculate_mse(const std::vector<float>& ref, const std::vector<float>& target) {
    double sum_sq_diff = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        float diff = ref[i] - target[i];
        sum_sq_diff += diff * diff;
    }
    return (float)(sum_sq_diff / ref.size());
}

void print_predictions(const std::vector<float>& results, const std::vector<std::string>& filenames, int batch_size, int top_k_to_show) {
    int count = std::min(batch_size, top_k_to_show);
    printf("\n Classifying top %d: \n", count);
    for (int i = 0; i < count; ++i) {
        int max_idx = 0; float max_val = results[i * OUTPUT_DIM];
        for (int k = 1; k < OUTPUT_DIM; ++k) {
            if (results[i * OUTPUT_DIM + k] > max_val) { max_val = results[i * OUTPUT_DIM + k]; max_idx = k; }
        }
        std::string fname = (i < filenames.size()) ? filenames[i] : "RandomInput";
        size_t last_slash = fname.find_last_of("\\/");
        if (last_slash != std::string::npos) fname = fname.substr(last_slash + 1);
        printf("Image [%-15s] -> Predicted: %-10s (Score: %.2f)\n", fname.c_str(), STL10_CLASS_NAMES[max_idx], max_val);
    }
}