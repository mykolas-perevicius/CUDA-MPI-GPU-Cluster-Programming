#ifndef LAYERS_CUDA_HPP
#define LAYERS_CUDA_HPP

#include <vector>

// Launches conv kernel: output size (Ho×Wo×K)
void cudaConvLayer(
    float* d_output,
    const float* d_input,
    const float* d_weights,
    const float* d_biases,
    int H, int W, int C,
    const int K, const int F, const int S, const int P);

// Elementwise ReLU in‐place
void cudaReluLayer(float* d_data, int N);

// Max‐pool in one kernel
void cudaMaxPoolLayer(
    float* d_output,
    const float* d_input,
    int H, int W, int C,
    int F_pool, int S_pool);

// Local response normalization
void cudaLRNLayer(
    float* d_output,
    const float* d_input,
    int H, int W, int C,
    int N, float alpha, float beta, float k);

#endif // LAYERS_CUDA_HPP
