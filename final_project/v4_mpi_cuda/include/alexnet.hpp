// final_project/v4_mpi_cuda/include/alexnet.hpp
#ifndef ALEXNET_MPI_CUDA_HPP
#define ALEXNET_MPI_CUDA_HPP

#include <vector>
#include <string> // For error messages
#include <cstddef> // For size_t

// Holds layer parameters (same as V1/V3)
// Host vectors store the master copy, will be copied to device.
struct LayerParams {
    std::vector<float> weights;
    std::vector<float> biases;
    int K, F, S, P;      // Conv params
    int F_pool, S_pool;  // Pooling
    int N_lrn;           // LRN window
    float alpha, beta, k_lrn;
};

// Function to run the forward pass using MPI and CUDA
// Takes the local portion of input data on host, performs computation on GPU,
// returns the local portion of the final output on host.
// **Declaration Only** - Definition is in alexnet_mpi_cuda.cu
// *** ADDED original total height H argument ***
void alexnetForwardPassMPI_CUDA(
    const std::vector<float>& h_localInput, // Local input data slice on host
    int localH,          // Height of the local input slice
    int H, int W, int C, // *** ADDED H *** Original Dimensions (H, W, C)
    const LayerParams& p1, // Params for Block 1 (Conv1, Pool1)
    const LayerParams& p2, // Params for Block 2 (Conv2, Pool2, LRN2)
    std::vector<float>& h_localOutput, // Local output slice on host (resized by function)
    int rank, int size   // MPI rank and size for halo logic
);

// Helper inline functions for calculating output dimensions
inline int convOutDim(int D, int F, int P, int S) {
    if (S <= 0) return 0; // Avoid division by zero
    // Ensure filter fits, considering padding
    if (F > D + 2 * P) return 0;
    return (D + 2*P - F) / S + 1;
}

inline int poolOutDim(int D, int F, int S) {
     if (S <= 0) return 0; // Avoid division by zero
    // Standard pooling, assumes P=0
    // Handle edge case where input dim is less than filter dim
    if (D < F) return 0;
    return (D - F) / S + 1;
}

#endif // ALEXNET_MPI_CUDA_HPP