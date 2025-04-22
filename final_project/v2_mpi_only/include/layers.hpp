// final_project/include/layers.hpp
#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <cuda_runtime.h>

// --- Layer Parameter Structures ---
struct ConvLayerParams {
    int inputChannels;
    int outputChannels;
    int kernelSize;  // Assumes square kernels (R = S)
    int stride;
    int padding;
    int inputHeight;
    int inputWidth;
    int outputHeight;
    int outputWidth;
};

struct PoolLayerParams {
    int poolSize;    // e.g., 3
    int stride;      // e.g., 2
    int inputChannels;
    int inputHeight;
    int inputWidth;
    int outputHeight;
    int outputWidth;
};

struct LRNLayerParams {
    int channels;    // Number of feature map channels
    int height;
    int width;
    int localSize;   // Typically 5 in AlexNet
    float alpha;     // Typically 1e-4
    float beta;      // Typically 0.75
    float k;         // Typically 2.0
};

// --- Existing Launchers for Block1 (Conv1->ReLU->Pool1) ---
// (Assumed to be implemented elsewhere)

// --- New Declarations for Block2: Conv2 -> ReLU2 -> Pool2 -> LRN2 ---
void launch_conv2d_forward_conv2(
    const float* d_input, float* d_output,
    const float* d_weights, const float* d_biases,
    const ConvLayerParams& params,
// CUDA? // CUDA?     cudaStream_t stream = 0);

// CUDA? // CUDA? void launch_relu_forward_conv2(float* d_data, int N, cudaStream_t stream = 0);

void launch_maxpool_forward2(
    const float* d_input, float* d_output,
    const PoolLayerParams& params,
// CUDA? // CUDA?     cudaStream_t stream = 0);

void launch_lrn_forward(
    const float* d_input, float* d_output,
    const LRNLayerParams& params,
// CUDA? // CUDA?     cudaStream_t stream = 0);

#endif // LAYERS_HPP
