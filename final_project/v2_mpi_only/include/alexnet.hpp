#ifndef ALEXNET_HPP
#define ALEXNET_HPP

#include <vector>
#include <string>
#include <cstddef> // Include for size_t

// Structure to hold layer parameters
struct LayerParams {
    // Convolution Params
    std::vector<float> weights;
    std::vector<float> biases;
    int K, F, S, P; // K=NumFilters, F=FilterSize, S=Stride, P=Padding

    // Pooling Params (Associated with the preceding Conv layer for convenience)
    int F_pool; // Pooling Filter Size
    int S_pool; // Pooling Stride

    // LRN Params (Associated with the layer needing LRN)
    int N_lrn;  // LRN Window Size
    float alpha;
    float beta;
    float k_lrn;
};

// Corrected function prototype name
void alexnetForwardPass(
    std::vector<float>& input_data, // Input image data (flattened)
    const LayerParams& paramsConv1, // Includes Pool1 params
    const LayerParams& paramsConv2, // Includes Pool2 & LRN2 params
    int H, int W, int C // Initial image dimensions
);

// Helper function to initialize data (e.g., random)
void initializeData(std::vector<float>& data, size_t size);
void initializeWeights(std::vector<float>& weights, size_t size);
void initializeBiases(std::vector<float>& biases, size_t size);

// Helper function to print dimensions (for debugging)
void printDimensions(const std::string& layer_name, int H, int W, int C);

#endif // ALEXNET_HPP