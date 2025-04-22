#include "alexnet.hpp"
#include "layers.hpp"
#include <vector>
#include <iostream>
#include <chrono> // For timing

// Helper to calculate output dimensions
void calculate_conv_output_dims(int& outH, int& outW, int H, int W, int F, int S, int P) {
    outH = (H - F + 2 * P) / S + 1;
    outW = (W - F + 2 * P) / S + 1;
}

void calculate_pool_output_dims(int& outH, int& outW, int H, int W, int F, int S) {
    outH = (H - F) / S + 1;
    outW = (W - F) / S + 1;
}


void alexnetForwardPass(
    std::vector<float>& input_data, // Gets modified in-place by ReLU/LRN
    const LayerParams& paramsConv1,
    const LayerParams& paramsConv2,
    int H, int W, int C)
{
    std::cout << "Starting AlexNet Serial Forward Pass..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // --- Layer Buffers (using vectors for simplicity) ---
    // We reuse buffers where possible, but need separate ones when dimensions change significantly
    std::vector<float> buffer1, buffer2;
    std::vector<float>* current_input = &input_data;
    std::vector<float>* current_output = &buffer1; // Start with buffer1 as output

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
    serialConvLayer(*current_output, *current_input, paramsConv1.weights, paramsConv1.biases,
                    currentH, currentW, currentC, paramsConv1.K, paramsConv1.F, paramsConv1.S, paramsConv1.P);
    std::swap(current_input, current_output); // Output becomes input for next layer
    currentH = nextH; currentW = nextW; currentC = nextC;
    printDimensions("After Conv1", currentH, currentW, currentC);

    // ReLU1
    std::cout << "Applying ReLU1..." << std::endl;
    serialReluLayer(*current_input); // In-place
    printDimensions("After ReLU1", currentH, currentW, currentC);

    // Pool1
    std::cout << "Applying MaxPool1..." << std::endl;
    calculate_pool_output_dims(nextH, nextW, currentH, currentW, paramsConv1.F_pool, paramsConv1.S_pool); // Need pool params
    nextC = currentC;
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
     // Ensure the non-active buffer is large enough or resize
    if (current_output->size() < nextH * nextW * nextC) {
        current_output->resize(nextH * nextW * nextC);
    }
    serialConvLayer(*current_output, *current_input, paramsConv2.weights, paramsConv2.biases,
                    currentH, currentW, currentC, paramsConv2.K, paramsConv2.F, paramsConv2.S, paramsConv2.P);
    std::swap(current_input, current_output);
    currentH = nextH; currentW = nextW; currentC = nextC;
    printDimensions("After Conv2", currentH, currentW, currentC);


    // ReLU2
    std::cout << "Applying ReLU2..." << std::endl;
    serialReluLayer(*current_input); // In-place
    printDimensions("After ReLU2", currentH, currentW, currentC);


    // Pool2
    std::cout << "Applying MaxPool2..." << std::endl;
    calculate_pool_output_dims(nextH, nextW, currentH, currentW, paramsConv2.F_pool, paramsConv2.S_pool); // Need pool params
    nextC = currentC;
     // Ensure the non-active buffer is large enough or resize
    if (current_output->size() < nextH * nextW * nextC) {
        current_output->resize(nextH * nextW * nextC);
    }
    serialMaxPoolLayer(*current_output, *current_input,
                       currentH, currentW, currentC, paramsConv2.F_pool, paramsConv2.S_pool);
    std::swap(current_input, current_output);
    currentH = nextH; currentW = nextW; currentC = nextC;
    printDimensions("After Pool2", currentH, currentW, currentC);


    // LRN2
    std::cout << "Applying LRN2..." << std::endl;
    // LRN output dims are same as input
    nextH = currentH; nextW = currentW; nextC = currentC;
    // Ensure the non-active buffer is large enough or resize
    if (current_output->size() < nextH * nextW * nextC) {
        current_output->resize(nextH * nextW * nextC);
    }
    serialLRNLayer(*current_output, *current_input,
                   currentH, currentW, currentC, paramsConv2.N_lrn, paramsConv2.alpha, paramsConv2.beta, paramsConv2.k_lrn); // Need LRN params
    std::swap(current_input, current_output);
    // currentH = nextH; currentW = nextW; currentC = nextC; // Dims don't change
    printDimensions("After LRN2", currentH, currentW, currentC);

    // --- Copy final result back to input_data if necessary ---
    // If the final result ended up in buffer1 or buffer2, copy it back
    if (current_input != &input_data) {
         std::cout << "Copying final result back to original buffer..." << std::endl;
         input_data = *current_input; // Copy vector content
    }


    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "AlexNet Serial Forward Pass completed in " << duration.count() << " ms" << std::endl;

    // Output first few values of the final result for basic check
    std::cout << "Final Output (first 10 values): ";
    for(int i = 0; i < std::min((size_t)10, input_data.size()); ++i) {
        std::cout << input_data[i] << " ";
    }
    std::cout << "..." << std::endl;
}


// Helper function implementations
void initializeData(std::vector<float>& data, size_t size) {
    data.resize(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.1f; // Small random values
    }
}

void initializeWeights(std::vector<float>& weights, size_t size) {
     weights.resize(size);
    for (size_t i = 0; i < size; ++i) {
        weights[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f) * 0.02f; // Small random weights around 0
    }
}
void initializeBiases(std::vector<float>& biases, size_t size) {
    biases.assign(size, 0.1f); // Initialize biases to a small constant value
}


void printDimensions(const std::string& layer_name, int H, int W, int C) {
    std::cout << "  [" << layer_name << "] Dimensions: H=" << H << ", W=" << W << ", C=" << C << std::endl;
}