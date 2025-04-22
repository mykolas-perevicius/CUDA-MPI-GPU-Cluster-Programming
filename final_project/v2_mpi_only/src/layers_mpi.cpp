#include "layers.hpp"
#include <vector>
#include <cmath>      // For pow, std::max
#include <iostream>   // For potential debugging output
#include <algorithm>  // For std::max, std::min
#include <limits>     // Include for std::numeric_limits
#include <cstddef>    // Include for size_t

// Helper function for indexing flattened 3D arrays (H, W, C)
inline int get_idx(int h, int w, int c, int W, int C) {
    return (h * W + w) * C + c;
}

// Helper function for indexing flattened 4D arrays (K, H, W, C) - typical for weights
// K = filter index (output channel), c = input channel, fh/fw = filter row/col
inline int get_weight_idx(int k, int c, int fh, int fw, int C_in, int F_size) {
     // Stride through filters -> Stride through input channels -> Stride through filter rows -> Stride through filter columns
     return k * (C_in * F_size * F_size) + c * (F_size * F_size) + fh * F_size + fw;
}


// --- Serial Layer Implementations ---

void serialConvLayer(
    std::vector<float>& output,
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& biases,
    int H, int W, int C, int K, int F, int S, int P)
{
    int outH = (H - F + 2 * P) / S + 1;
    int outW = (W - F + 2 * P) / S + 1;
    // Check for invalid dimensions
    if (outH <= 0 || outW <= 0) {
        // Handle error: throw exception, print message, etc.
        std::cerr << "Error: Invalid output dimensions in Conv layer." << std::endl;
        output.clear(); // Indicate error state
        return;
    }
    output.assign(static_cast<size_t>(outH) * outW * K, 0.0f); // Use size_t for allocation

    for (int k = 0; k < K; ++k) { // For each output channel (filter)
        for (int oh = 0; oh < outH; ++oh) { // For each output row
            for (int ow = 0; ow < outW; ++ow) { // For each output column
                float sum = 0.0f;
                // Apply the filter
                for (int c = 0; c < C; ++c) { // For each input channel
                    for (int fh = 0; fh < F; ++fh) { // For each filter row
                        for (int fw = 0; fw < F; ++fw) { // For each filter column
                            // Calculate input coordinates (considering padding and stride)
                            int ih = oh * S + fh - P;
                            int iw = ow * S + fw - P;

                            // Check bounds (handle padding implicitly by skipping out-of-bounds reads)
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int input_idx = get_idx(ih, iw, c, W, C);
                                // Pass C (input channels) and F (filter size) to weight index func
                                int weight_idx = get_weight_idx(k, c, fh, fw, C, F);
                                sum += input[static_cast<size_t>(input_idx)] * weights[static_cast<size_t>(weight_idx)];
                            }
                        }
                    }
                }
                // Add bias
                sum += biases[static_cast<size_t>(k)]; // Use size_t for index
                // Pass K (output channels) to output index func
                int output_idx = get_idx(oh, ow, k, outW, K);
                output[static_cast<size_t>(output_idx)] = sum; // Use size_t for index
            }
        }
    }
}


void serialReluLayer(std::vector<float>& data) {
    for (size_t i = 0; i < data.size(); ++i) { // Use size_t for loop counter
        data[i] = std::max(0.0f, data[i]);
    }
}


void serialMaxPoolLayer(
    std::vector<float>& output,
    const std::vector<float>& input,
    int H, int W, int C, int F, int S)
{
    int outH = (H - F) / S + 1;
    int outW = (W - F) / S + 1;
    // Check for invalid dimensions
    if (outH <= 0 || outW <= 0) {
        std::cerr << "Error: Invalid output dimensions in Pool layer." << std::endl;
        output.clear();
        return;
    }
    output.assign(static_cast<size_t>(outH) * outW * C, 0.0f); // Use size_t

    for (int c = 0; c < C; ++c) { // For each channel
        for (int oh = 0; oh < outH; ++oh) { // For each output row
            for (int ow = 0; ow < outW; ++ow) { // For each output column
                // Use lowest possible float value instead of negative infinity for better portability
                float max_val = std::numeric_limits<float>::lowest();
                // Find max in the pooling window
                for (int fh = 0; fh < F; ++fh) { // For each window row
                    for (int fw = 0; fw < F; ++fw) { // For each window column
                        int ih = oh * S + fh;
                        int iw = ow * S + fw;
                        // Check bounds (should always be within bounds if H, W calculated correctly)
                        if (ih < H && iw < W) { // No need for >= 0 check if stride/filter ok
                             int input_idx = get_idx(ih, iw, c, W, C);
                             max_val = std::max(max_val, input[static_cast<size_t>(input_idx)]); // Use size_t
                        }
                    }
                }
                 // Pass C (channels) to output index func
                 int output_idx = get_idx(oh, ow, c, outW, C);
                 output[static_cast<size_t>(output_idx)] = max_val; // Use size_t
            }
        }
    }
}


void serialLRNLayer(
    std::vector<float>& output,
    const std::vector<float>& input,
    int H, int W, int C, int N, float alpha, float beta, float k_lrn) // Renamed k to k_lrn
{
    output.assign(static_cast<size_t>(H) * W * C, 0.0f); // Use size_t
    int half_N = N / 2;

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int c = 0; c < C; ++c) {
                // Calculate the sum of squares in the neighborhood across channels
                float sum_sq = 0.0f;
                // Correct loop bounds using std::max/min
                for (int i = std::max(0, c - half_N); i <= std::min(C - 1, c + half_N); ++i) {
                    int neighbor_idx = get_idx(h, w, i, W, C);
                    float val = input[static_cast<size_t>(neighbor_idx)]; // Use size_t
                    sum_sq += val * val;
                }

                // Apply LRN formula
                int current_idx = get_idx(h, w, c, W, C);
                float denominator = pow(k_lrn + (alpha / static_cast<float>(N)) * sum_sq, beta); // Cast N to float for division
                if (denominator == 0.0f) {
                     // Handle potential division by zero if needed
                     output[static_cast<size_t>(current_idx)] = 0.0f; // Or some other strategy
                } else {
                    output[static_cast<size_t>(current_idx)] = input[static_cast<size_t>(current_idx)] / denominator;
                }
            }
        }
    }
}