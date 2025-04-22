#ifndef ALEXNET_CUDA_HPP
#define ALEXNET_CUDA_HPP

#include <vector>

// Holds layer parameters (same as V1)
struct LayerParams {
    std::vector<float> weights;
    std::vector<float> biases;
    int K, F, S, P;      // Conv params
    int F_pool, S_pool;  // Pooling
    int N_lrn;           // LRN window
    float alpha, beta, k_lrn;
};

// Runs full forward pass on GPU
// input: flattened H×W×C
// out: flattened final feature map
void alexnetForwardPassCUDA(
    const std::vector<float>& input_host,
    const LayerParams& paramsConv1,
    const LayerParams& paramsConv2,
    int H, int W, int C,
    std::vector<float>& output_host);

#endif // ALEXNET_CUDA_HPP
