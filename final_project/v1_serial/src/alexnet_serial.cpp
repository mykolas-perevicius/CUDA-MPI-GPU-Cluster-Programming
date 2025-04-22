#include <vector>
#include <iostream>
#include <chrono>    // For timing
#include <algorithm> // For std::min, std::swap
#include <cmath>     // For std::max, std::pow // LRN might use pow
#include <string>    // For std::string
#include <cstdlib>   // For rand(), RAND_MAX
#include <cstddef>   // For size_t
#include <limits>    // For numeric_limits if needed by layers

// Include necessary headers ONCE correctly
#include "../include/alexnet.hpp" // Correct relative path for LayerParams
#include "../include/layers.hpp"  // Correct relative path for serial layer functions

// --- Helper function definitions belong here (or a separate utils.cpp) ---
// --- DO NOT duplicate these definitions in layers_serial.cpp ---

// Helper to calculate output dimensions (using reference parameters)
void calculate_conv_output_dims(int& outH, int& outW, int H, int W, int F, int S, int P) {
    if (S <= 0) {
        std::cerr << "Error: Stride (S) must be positive." << std::endl;
        outH = 0; outW = 0; return;
    }
    outH = (H - F + 2 * P) / S + 1;
    outW = (W - F + 2 * P) / S + 1;
}

// Helper to calculate pooling output dimensions
void calculate_pool_output_dims(int& outH, int& outW, int H, int W, int F, int S) {
    if (S <= 0) {
        std::cerr << "Error: Stride (S) must be positive." << std::endl;
        outH = 0; outW = 0; return;
    }
    outH = (H - F) / S + 1; // Standard pooling, assumes P=0
    outW = (W - F) / S + 1;
}

// Helper to initialize input data
void initializeData(std::vector<float>& data, size_t size) {
    data.resize(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.1f;
    }
}

// Helper to initialize weights
void initializeWeights(std::vector<float>& weights, size_t size) {
     weights.resize(size);
    for (size_t i = 0; i < size; ++i) {
        weights[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f) * 0.02f;
    }
}

// Helper to initialize biases
void initializeBiases(std::vector<float>& biases, size_t size) {
    biases.assign(size, 0.1f);
}

// Helper to print dimensions
void printDimensions(const std::string& layer_name, int H, int W, int C) {
    std::cout << "  [" << layer_name << "] Dimensions: H=" << H << ", W=" << W << ", C=" << C << std::endl;
}

// --- AlexNet Forward Pass Implementation ---
// --- Definition belongs ONLY here ---

void alexnetForwardPass(
    std::vector<float>& input_data,
    const LayerParams& paramsConv1,
    const LayerParams& paramsConv2,
    int H, int W, int C)
{
    std::cout << "Starting AlexNet Serial Forward Pass..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<float> buffer1, buffer2;
    std::vector<float>* current_input = &input_data;
    std::vector<float>* current_output = &buffer1;
    current_output->clear();

    int currentH = H;
    int currentW = W;
    int currentC = C;
    int nextH, nextW, nextC;

    // --- Block 1 ---
    printDimensions("Input", currentH, currentW, currentC);

    // Conv1
    std::cout << "Applying Conv1..." << std::endl;
    calculate_conv_output_dims(nextH, nextW, currentH, currentW, paramsConv1.F, paramsConv1.S, paramsConv1.P);
    nextC = paramsConv1.K;
    size_t next_size_conv1 = static_cast<size_t>(nextH) * nextW * nextC; // Cast first dim to size_t
    current_output->resize(next_size_conv1);
    serialConvLayer(*current_output, *current_input, paramsConv1.weights, paramsConv1.biases,
                    currentH, currentW, currentC, paramsConv1.K, paramsConv1.F, paramsConv1.S, paramsConv1.P);
    std::swap(current_input, current_output);
    currentH = nextH; currentW = nextW; currentC = nextC;
    printDimensions("After Conv1", currentH, currentW, currentC);

    // ReLU1
    std::cout << "Applying ReLU1..." << std::endl;
    serialReluLayer(*current_input);
    printDimensions("After ReLU1", currentH, currentW, currentC);

    // Pool1
    std::cout << "Applying MaxPool1..." << std::endl;
    calculate_pool_output_dims(nextH, nextW, currentH, currentW, paramsConv1.F_pool, paramsConv1.S_pool);
    nextC = currentC;
    size_t next_size_pool1 = static_cast<size_t>(nextH) * nextW * nextC;
    current_output->resize(next_size_pool1);
    serialMaxPoolLayer(*current_output, *current_input,
                       currentH, currentW, currentC, paramsConv1.F_pool, paramsConv1.S_pool);
    std::swap(current_input, current_output);
    currentH = nextH; currentW = nextW; currentC = nextC;
    printDimensions("After Pool1", currentH, currentW, currentC);

    // --- Block 2 ---

    // Conv2
    std::cout << "Applying Conv2..." << std::endl;
    calculate_conv_output_dims(nextH, nextW, currentH, currentW, paramsConv2.F, paramsConv2.S, paramsConv2.P);
    nextC = paramsConv2.K;
    size_t next_size_conv2 = static_cast<size_t>(nextH) * nextW * nextC;
    // Resize buffer if necessary
    if (current_output->size() != next_size_conv2) {
         current_output->resize(next_size_conv2);
    }
    serialConvLayer(*current_output, *current_input, paramsConv2.weights, paramsConv2.biases,
                    currentH, currentW, currentC, paramsConv2.K, paramsConv2.F, paramsConv2.S, paramsConv2.P);
    std::swap(current_input, current_output);
    currentH = nextH; currentW = nextW; currentC = nextC;
    printDimensions("After Conv2", currentH, currentW, currentC);

    // ReLU2
    std::cout << "Applying ReLU2..." << std::endl;
    serialReluLayer(*current_input);
    printDimensions("After ReLU2", currentH, currentW, currentC);

    // Pool2
    std::cout << "Applying MaxPool2..." << std::endl;
    calculate_pool_output_dims(nextH, nextW, currentH, currentW, paramsConv2.F_pool, paramsConv2.S_pool);
    nextC = currentC;
    size_t next_size_pool2 = static_cast<size_t>(nextH) * nextW * nextC;
    if (current_output->size() != next_size_pool2) {
        current_output->resize(next_size_pool2);
    }
    serialMaxPoolLayer(*current_output, *current_input,
                       currentH, currentW, currentC, paramsConv2.F_pool, paramsConv2.S_pool);
    std::swap(current_input, current_output);
    currentH = nextH; currentW = nextW; currentC = nextC;
    printDimensions("After Pool2", currentH, currentW, currentC);

    // LRN2
    std::cout << "Applying LRN2..." << std::endl;
    nextH = currentH; nextW = currentW; nextC = currentC; // LRN doesn't change dimensions
    size_t next_size_lrn2 = static_cast<size_t>(nextH) * nextW * nextC;
    if (current_output->size() != next_size_lrn2) {
        current_output->resize(next_size_lrn2);
    }
    serialLRNLayer(*current_output, *current_input,
                   currentH, currentW, currentC, paramsConv2.N_lrn, paramsConv2.alpha, paramsConv2.beta, paramsConv2.k_lrn);
    std::swap(current_input, current_output);
    printDimensions("After LRN2", currentH, currentW, currentC);

    // --- Copy final result back to input_data if necessary ---
    if (current_input != &input_data) {
         std::cout << "Copying final result back to original buffer..." << std::endl;
         input_data = *current_input; // Vector assignment handles copy/resize
    } else {
         std::cout << "Final result is already in the original buffer." << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "AlexNet Serial Forward Pass completed in " << duration.count() << " ms" << std::endl;

    // Output first few values
    std::cout << "Final Output (first 10 values): ";
    // Use size_t for loop counter when comparing with vector::size()
    size_t print_count = std::min(static_cast<size_t>(10), input_data.size());
    for(size_t i = 0; i < print_count; ++i) {
        std::cout << input_data[i] << (i == print_count - 1 ? "" : " ");
    }
    std::cout << (input_data.size() > 10 ? "..." : "") << std::endl;
}