#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <vector> // Use vectors for simplicity, or raw pointers if preferred

// --- Serial Layer Function Prototypes ---

// Naive Serial Convolution Layer
void serialConvLayer(
    std::vector<float>& output,
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& biases, // Added biases
    int H, int W, int C, // Input dimensions (Height, Width, Channels)
    int K, // Number of filters (Output channels)
    int F, // Filter size (FxF)
    int S, // Stride
    int P  // Padding
);

// Serial ReLU Activation Layer (in-place or out-of-place)
void serialReluLayer(std::vector<float>& data); // Example: In-place

// Serial Max Pooling Layer
void serialMaxPoolLayer(
    std::vector<float>& output,
    const std::vector<float>& input,
    int H, int W, int C, // Input dimensions
    int F, // Filter size (pooling window size)
    int S  // Stride
);

// Serial Local Response Normalization (LRN) Layer
void serialLRNLayer(
    std::vector<float>& output,
    const std::vector<float>& input,
    int H, int W, int C, // Input dimensions
    int N,        // Size of the normalization window (across channels)
    float alpha,  // LRN parameter
    float beta,   // LRN parameter
    float k       // LRN parameter
);

#endif // LAYERS_HPP