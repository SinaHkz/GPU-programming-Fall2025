#pragma once
#include <vector>
#include <string>

void load_or_generate_weights(const std::string& filename, std::vector<float>& vec, int size);

//folder for PNGs and load them
int load_dataset(const std::string& folder_path, std::vector<unsigned char>& host_input, std::vector<std::string>& filenames);

std::string find_dataset_folder();

float calculate_mse(const std::vector<float>& ref, const std::vector<float>& target);

void print_predictions(const std::vector<float>& results, const std::vector<std::string>& filenames, int batch_size, int top_k_to_show);
