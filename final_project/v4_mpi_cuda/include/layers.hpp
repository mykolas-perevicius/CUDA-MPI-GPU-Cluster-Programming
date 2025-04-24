#ifndef LAYERS_MPI_CUDA_HPP
#define LAYERS_MPI_CUDA_HPP

#include <cstddef> // Include for size_t

// This header primarily declares the CUDA kernel *launcher* functions.
// The actual kernels (__global__) are defined in layers_mpi_cuda.cu.
// These signatures are identical to V3.
// **Declarations Only** - Definitions are in layers_mpi_cuda.cu

// Launches conv kernel: output size (Ho×Wo×K)
// Operates entirely on device memory pointers.
void cudaConvLayer(
    float* d_output,
    const float* d_input,
    const float* d_weights,
    const float* d_biases,
    int H, int W, int C, // Input dimensions (of d_input)
    const int K, const int F, const int S, const int P); // Layer params

// Elementwise ReLU in‐place on device memory
void cudaReluLayer(float* d_data, size_t N); // N = total number of elements

// Max‐pool kernel launcher on device memory
void cudaMaxPoolLayer(
    float* d_output,
    const float* d_input,
    int H, int W, int C, // Input dimensions (of d_input)
    int F_pool, int S_pool); // Pooling params

// Local response normalization kernel launcher on device memory
void cudaLRNLayer(
    float* d_output,
    const float* d_input,
    int H, int W, int C, // Input dimensions (of d_input)
    int N, float alpha, float beta, float k); // LRN params

#endif // LAYERS_MPI_CUDA_HPP