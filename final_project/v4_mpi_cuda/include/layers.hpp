// final_project/v4_mpi_cuda/include/layers.hpp
#ifndef LAYERS_MPI_CUDA_HPP
#define LAYERS_MPI_CUDA_HPP

#include <cstddef> // Include for size_t

// CUDA Kernel Launcher Declarations
// These are host functions defined in layers_mpi_cuda.cu
// Ensure these exactly match the definitions.

void cudaConvLayer(
    float* d_output,
    const float* d_input,
    const float* d_weights,
    const float* d_biases,
    int H, int W, int C,
    const int K, const int F, const int S, const int P);

void cudaReluLayer(
    float* d_data,
    size_t N); // Use size_t consistently

void cudaMaxPoolLayer(
    float* d_output,
    const float* d_input,
    int H, int W, int C,
    int F_pool, int S_pool);

void cudaLRNLayer(
    float* d_output,
    const float* d_input,
    int H, int W, int C,
    int N, float alpha, float beta, float k);

#endif // LAYERS_MPI_CUDA_HPP