#include <vector>
#include <cmath>     // For std::max, std::pow, std::fmax? (check usage)
#include <algorithm> // For std::max, std::min
#include <limits>    // For std::numeric_limits
#include <cstddef>   // For size_t
#include <iostream>  // For error checking (optional)

// Include necessary headers
#include "layers.hpp"  // Should declare the functions being implemented here
#include "alexnet.hpp" // Needed for LayerParams if used directly (conv layer signature needs adjustment)

// --- 3D Index Helper ---
// Make it static or put in an anonymous namespace if only used in this file
namespace { // Anonymous namespace limits scope to this file
    inline size_t idx3D(int h, int w, int c, int W, int C) {
        // Add checks for h, w, c boundaries if needed for robustness
        return (static_cast<size_t>(h) * W + w) * C + c;
    }
    // Helper to calculate output dimensions (needed by layers implementation)
    // Can be defined here (static/anon namespace) or declared in layers.hpp and defined once elsewhere.
    // Let's keep it local here for now.
    inline int calculateConvOutputDim(int D, int F, int P, int S) {
        if (S <= 0) return 0; // Avoid division by zero
        return (D - F + 2 * P) / S + 1;
    }
    inline int calculatePoolOutputDim(int D, int F, int S) {
        if (S <= 0) return 0; // Avoid division by zero
        return (D - F) / S + 1; // Assumes P=0 for pool
    }
} // End anonymous namespace


// --- Serial Layer Function Implementations ---

// Naive Serial Convolution Layer
// Signature matches layers.hpp
void serialConvLayer(
    std::vector<float>& output,
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& biases,
    int H, int W, int C,
    int K, int F, int S, int P)
{
    int Ho = calculateConvOutputDim(H, F, P, S);
    int Wo = calculateConvOutputDim(W, F, P, S);

    // Pre-check output size (optional but good practice)
    if (output.size() != static_cast<size_t>(Ho) * Wo * K) {
         std::cerr << "Warning: Output vector size mismatch in serialConvLayer. Resizing." << std::endl;
         output.resize(static_cast<size_t>(Ho) * Wo * K);
    }

    // Parallelizing this loop is the main goal of MPI/CUDA versions
    for (int k = 0; k < K; ++k) { // For each output channel (filter)
        for (int ho = 0; ho < Ho; ++ho) { // For each output row
            for (int wo = 0; wo < Wo; ++wo) { // For each output column
                float sum = biases[k]; // Start with bias
                // Apply filter
                for (int c = 0; c < C; ++c) { // For each input channel
                    for (int fh = 0; fh < F; ++fh) { // For each filter row
                        for (int fw = 0; fw < F; ++fw) { // For each filter column
                            int hi = ho * S - P + fh; // Input row index
                            int wi = wo * S - P + fw; // Input column index

                            // Check bounds (convolution with padding)
                            if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                                size_t input_idx = idx3D(hi, wi, c, W, C);
                                // Weight index: OutputChannel(k), InputChannel(c), FilterRow(fh), FilterCol(fw)
                                size_t weight_idx = (((static_cast<size_t>(k) * C + c) * F + fh) * F) + fw;
                                sum += input[input_idx] * weights[weight_idx];
                            }
                            // else: contribution is 0 (implicitly, due to padding)
                        }
                    }
                }
                output[idx3D(ho, wo, k, Wo, K)] = sum; // Store result
            }
        }
    }
}

// Serial ReLU Activation Layer (in-place)
// Signature matches layers.hpp
void serialReluLayer(std::vector<float>& data) {
    for (float& val : data) {
        val = std::max(0.0f, val);
    }
    // Can be optimized using std::transform or OpenMP for trivial parallelization on CPU
}

// Serial Max Pooling Layer
// Signature matches layers.hpp
void serialMaxPoolLayer(
    std::vector<float>& output,
    const std::vector<float>& input,
    int H, int W, int C,
    int F, // Filter size (pooling window size)
    int S) // Stride
{
    int Ho = calculatePoolOutputDim(H, F, S);
    int Wo = calculatePoolOutputDim(W, F, S);

    // Pre-check output size
     if (output.size() != static_cast<size_t>(Ho) * Wo * C) {
         std::cerr << "Warning: Output vector size mismatch in serialMaxPoolLayer. Resizing." << std::endl;
         output.resize(static_cast<size_t>(Ho) * Wo * C);
     }

    for (int c = 0; c < C; ++c) { // For each channel (pooling is usually done per-channel)
        for (int ho = 0; ho < Ho; ++ho) {
            for (int wo = 0; wo < Wo; ++wo) {
                float max_val = -std::numeric_limits<float>::infinity();
                // Find max in the FxF window
                for (int fh = 0; fh < F; ++fh) {
                    for (int fw = 0; fw < F; ++fw) {
                        int hi = ho * S + fh;
                        int wi = wo * S + fw;
                        // Bounds check (should ideally not be needed if Ho, Wo calculated correctly)
                        if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                            max_val = std::max(max_val, input[idx3D(hi, wi, c, W, C)]);
                        }
                    }
                }
                output[idx3D(ho, wo, c, Wo, C)] = max_val;
            }
        }
    }
}

// Serial Local Response Normalization (LRN) Layer
// Signature matches layers.hpp
void serialLRNLayer(
    std::vector<float>& output,
    const std::vector<float>& input,
    int H, int W, int C,
    int N, float alpha, float beta, float k)
{
     // Pre-check output size
     if (output.size() != input.size()) {
         std::cerr << "Warning: Output vector size mismatch in serialLRNLayer. Resizing." << std::endl;
         output.resize(input.size());
     }
     if (N <= 0) {
         std::cerr << "Error: LRN window size (N) must be positive." << std::endl;
         // Optionally copy input to output or handle error differently
         std::copy(input.begin(), input.end(), output.begin());
         return;
     }

    int half_N = N / 2;
    float alpha_over_N = alpha / static_cast<float>(N); // Precompute

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int c = 0; c < C; ++c) { // For each channel at this (h, w) position
                // Calculate the sum of squares across the channel window
                float sum_sq = 0.0f;
                int c_start = std::max(0, c - half_N);
                int c_end = std::min(C - 1, c + half_N);
                for (int i = c_start; i <= c_end; ++i) {
                    float val = input[idx3D(h, w, i, W, C)];
                    sum_sq += val * val;
                }

                // Calculate the normalization factor
                float norm_factor = std::pow(k + alpha_over_N * sum_sq, beta);

                // Apply normalization
                size_t current_idx = idx3D(h, w, c, W, C);
                output[current_idx] = input[current_idx] / norm_factor;
            }
        }
    }
}