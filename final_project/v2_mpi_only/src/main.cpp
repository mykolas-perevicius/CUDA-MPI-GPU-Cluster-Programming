#include <iostream>
#include <vector>
#include <cstdlib> // For rand(), srand()
#include <ctime>   // For time()
#include <stdexcept> // For exceptions
#include <cstddef> // Include for size_t

#include "alexnet.hpp" // Include the header for the forward pass function

int main(int argc, char* argv[]) {
    // --- Basic Setup ---
    srand(time(0)); // Seed random number generator

    std::cout << "--- AlexNet Serial CPU Version (V1) ---" << std::endl;

    // --- Define Network Dimensions and Parameters (EXAMPLE VALUES) ---
    // Input Image
    int H = 227, W = 227, C = 3; // Example AlexNet input size

    // Conv1 Parameters
    LayerParams paramsConv1;
    paramsConv1.K = 96;  // Num filters
    paramsConv1.F = 11;  // Filter size
    paramsConv1.S = 4;   // Stride
    paramsConv1.P = 0;   // Padding
    // Pool1 (often associated with Conv1 params)
    paramsConv1.F_pool = 3;
    paramsConv1.S_pool = 2;

    // Conv2 Parameters
    LayerParams paramsConv2;
    paramsConv2.K = 256; // Num filters
    paramsConv2.F = 5;   // Filter size
    paramsConv2.S = 1;   // Stride
    paramsConv2.P = 2;   // Padding
    // Pool2
    paramsConv2.F_pool = 3;
    paramsConv2.S_pool = 2;
    // LRN2
    paramsConv2.N_lrn = 5;
    paramsConv2.alpha = 0.0001f;
    paramsConv2.beta = 0.75f;
    paramsConv2.k_lrn = 2.0f;


    // --- Allocate and Initialize Data & Weights (Host Memory) ---
    std::cout << "Initializing data and parameters..." << std::endl;

    // Input Data
    std::vector<float> h_inputData;
    initializeData(h_inputData, static_cast<size_t>(H) * W * C); // Use size_t

    // Conv1 Weights & Biases
    initializeWeights(paramsConv1.weights, static_cast<size_t>(paramsConv1.K) * C * paramsConv1.F * paramsConv1.F);
    initializeBiases(paramsConv1.biases, static_cast<size_t>(paramsConv1.K));


    // Conv2 Weights & Biases
    // Input channels to Conv2 is output channels of Conv1/Pool1
    int C_conv2_input = paramsConv1.K;
    initializeWeights(paramsConv2.weights, static_cast<size_t>(paramsConv2.K) * C_conv2_input * paramsConv2.F * paramsConv2.F);
    initializeBiases(paramsConv2.biases, static_cast<size_t>(paramsConv2.K));

    std::cout << "Initialization complete." << std::endl;


    // --- Perform Forward Pass ---
    try {
        alexnetForwardPass(h_inputData, paramsConv1, paramsConv2, H, W, C);
    } catch (const std::exception& e) {
        std::cerr << "Error during forward pass: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "--- Serial execution finished ---" << std::endl;

    return 0;
}