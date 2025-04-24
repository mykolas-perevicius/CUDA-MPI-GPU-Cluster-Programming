#include <cstdio>         // For fprintf, stderr in CUDA_CHECK
#include <cstdlib>        // For exit in CUDA_CHECK
#include <cmath>          // For fmaxf, powf, max, min
#include <cuda_runtime.h>
#include <mpi.h>          // Needed for MPI_Abort in CUDA_CHECK
#include <cstddef>        // Include for size_t

#include "../include/layers.hpp" // Function prototypes being implemented

// Added definitions for convOutDim and poolOutDim for CUDA kernels
__host__ __device__ inline int convOutDim(int D, int F, int P, int S) {
    return (D + 2 * P - F) / S + 1;
}

__host__ __device__ inline int poolOutDim(int D, int F, int S) {
    return (D - F) / S + 1;
}

// Macro to check CUDA calls - ABORTS using MPI for coordinated shutdown
#define CUDA_CHECK(call)                                    \
  do {                                                      \
    cudaError_t err = call;                                 \
    if(err != cudaSuccess) {                                \
      int rank_for_error; MPI_Comm_rank(MPI_COMM_WORLD, &rank_for_error); \
      fprintf(stderr,                                       \
        "[Rank %d] CUDA error in %s:%d: %s (%d)\n",         \
         rank_for_error, __FILE__, __LINE__, cudaGetErrorString(err), err); \
      fflush(stderr); /* Ensure message prints before abort */ \
      MPI_Abort(MPI_COMM_WORLD, err);                       \
    }                                                       \
  } while(0)

// --- Helper Functions (Device) ---
// Optional: Define __device__ helpers if needed by kernels, e.g., index calculation

// --- Kernel Implementations ---
// These kernels are identical to V3, with the pooling index fix

// Conv kernel (one thread per output element: Ho * Wo * K)
__global__ void convKernel(
    float* __restrict__ out, const float* __restrict__ in,
    const float* __restrict__ w, const float* __restrict__ b,
    int H, int W, int C, int K, int F, int S, int P,
    int Ho, int Wo)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = Ho * Wo * K;

    if (idx >= total_outputs) return;

    // Calculate output coordinates (k, y, x) from global thread index
    int k = idx % K;          // Output channel index
    int temp = idx / K;
    int x = temp % Wo;        // Output column index
    int y = temp / Wo;        // Output row index

    float sum = 0.0f; // Use float for intermediate sum

    // Corresponding input top-left corner coordinates
    int start_y = y * S - P;
    int start_x = x * S - P;

    // Iterate over input channels and filter dimensions
    for (int c = 0; c < C; ++c) {
        for (int fy = 0; fy < F; ++fy) {
            int current_y = start_y + fy;
            for (int fx = 0; fx < F; ++fx) {
                int current_x = start_x + fx;

                // Check bounds: only process valid input pixels
                if (current_y >= 0 && current_y < H && current_x >= 0 && current_x < W) {
                    // Calculate 1D indices
                    // Input index: (batch=0 implicitly) height, width, channel
                    // Use size_t for intermediate calculations to avoid overflow with large dimensions
                    size_t in_idx = ((size_t)current_y * W + current_x) * C + c;
                    // Weight index: output_channel, input_channel, filter_y, filter_x
                    size_t w_idx = (((size_t)k * C + c) * F + fy) * F + fx;

                    // Accumulate product
                    sum += in[in_idx] * w[w_idx];
                }
                // Pixels outside bounds (due to padding) contribute 0, so no `else` needed
            }
        }
    }

    // Add bias (once per output element) and store result
    out[idx] = sum + b[k];
}


// ReLU kernel (elementwise, in-place)
__global__ void reluKernel(float* data, size_t N) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i] = fmaxf(0.0f, data[i]);
    }
}

// Max-pooling kernel (one thread per output element: Hp * Wp * C)
__global__ void poolKernel(
    float* __restrict__ out, const float* __restrict__ in,
    int H, int W, int C,  // Input dimensions
    int Fp, int Sp,       // Pooling parameters (Filter size, Stride)
    int Hp, int Wp)       // Output dimensions
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = Hp * Wp * C;

    if (idx >= total_outputs) return;

    // Calculate output coordinates (c, y, x)
    int c = idx % C;
    int temp = idx / C;
    // *** Corrected y calculation ***
    int y = temp / Wp; // Integer division gives row index
    int x = temp % Wp; // Modulo gives column index


    // Find top-left corner of the pooling window in the input
    int start_y = y * Sp;
    int start_x = x * Sp;

    float max_val = -INFINITY; // Initialize with negative infinity

    // Iterate over the pooling window (Fp x Fp)
    for (int fy = 0; fy < Fp; ++fy) {
        int current_y = start_y + fy;
        // Ensure we don't go out of input bounds vertically
        if (current_y >= H) continue;

        for (int fx = 0; fx < Fp; ++fx) {
            int current_x = start_x + fx;
             // Ensure we don't go out of input bounds horizontally
             if (current_x >= W) continue;

            // Calculate 1D index in the input tensor
            // Use size_t for intermediate calculations
            size_t in_idx = ((size_t)current_y * W + current_x) * C + c;

            // Update maximum value
            max_val = fmaxf(max_val, in[in_idx]);
        }
    }

    // Store the maximum value found in the output tensor
    out[idx] = max_val;
}


// LRN kernel (naive cross-channel)
__global__ void lrnKernel(
    float* __restrict__ out, const float* __restrict__ in,
    int H, int W, int C, // Input dimensions
    int N_lrn, float alpha, float beta, float k) // LRN parameters
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = H * W * C;

    if (idx >= total_elements) return;

    // Calculate coordinates (c, y, x)
    int c = idx % C;
    int temp = idx / C;
    int x = temp % W;
    int y = temp / W;

    // Calculate the sum of squares within the local neighborhood across channels
    float sum_sq = 0.0f;
    int half_N = N_lrn / 2;
    int c_start = max(0, c - half_N);
    int c_end = min(C - 1, c + half_N); // Inclusive end

    for (int channel_i = c_start; channel_i <= c_end; ++channel_i) {
         // Use size_t for intermediate calculations
        size_t neighbor_idx = ((size_t)y * W + x) * C + channel_i;
        float val = in[neighbor_idx];
        sum_sq += val * val;
    }

    // Calculate the normalization factor
    // Using the common implementation approach (alpha not divided by N)
     float scale = k + alpha * sum_sq;

    // Apply normalization: out = in / (scale ^ beta)
    // Avoid division by zero or negative scale if beta is fractional
    if (scale <= 0.0f && beta > 0.0f && beta != 1.0f) {
         // Handle error case: e.g., set output to 0 or clamp scale
         out[idx] = 0.0f; // Or potentially in[idx] depending on desired behavior
    } else {
         out[idx] = in[idx] * powf(scale, -beta); // Use powf for float exponentiation
    }
}


// --- Kernel Launcher Functions ---
// These are host functions that set up grid/block dimensions and launch kernels.
// ** Function Definitions **

// Definition for cudaConvLayer
void cudaConvLayer(
    float* d_output,
    const float* d_input,
    const float* d_weights,
    const float* d_biases,
    int H, int W, int C,
    const int K, const int F, const int S, const int P)
{
    // Calculate output dimensions using helper (robust check)
    int Ho = convOutDim(H, F, P, S);
    int Wo = convOutDim(W, F, P, S);

    if (Ho <= 0 || Wo <= 0 || K <= 0) {
         // Output dimensions are zero or negative, or no filters. Nothing to compute.
         // Ensure output buffer is zeroed if necessary, or just return.
         // If d_output was allocated based on these dims, it might be nullptr or size 0.
         // Optionally print a warning:
         // fprintf(stderr, "Warning: ConvLayer output size is zero or negative (Ho=%d, Wo=%d, K=%d). Skipping kernel launch.\n", Ho, Wo, K);
         return;
     }

    int total_outputs = Ho * Wo * K;
    int threads_per_block = 256; // Common choice, tune based on GPU architecture
    // Calculate grid size, ensuring it covers all outputs
    int blocks_per_grid = (total_outputs + threads_per_block - 1) / threads_per_block;

    convKernel<<<blocks_per_grid, threads_per_block>>>(
        d_output, d_input, d_weights, d_biases,
        H, W, C, K, F, S, P, Ho, Wo);
    CUDA_CHECK(cudaGetLastError()); // Check for launch configuration errors
    // Optional: Synchronize if needed immediately after kernel
    // CUDA_CHECK(cudaDeviceSynchronize());
}

// Definition for cudaReluLayer
void cudaReluLayer(float* d_data, size_t N) {
     if (N == 0 || d_data == nullptr) return; // Nothing to do
    int threads_per_block = 256;
    // Use size_t for N, cast carefully for grid calculation if N is very large
    // Standard approach assumes N fits within reasonable limits for grid calculation
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    reluKernel<<<blocks_per_grid, threads_per_block>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}

// Definition for cudaMaxPoolLayer
void cudaMaxPoolLayer(
    float* d_output,
    const float* d_input,
    int H, int W, int C,
    int F_pool, int S_pool)
{
     // Calculate output dimensions using helper (robust check)
    int Hp = poolOutDim(H, F_pool, S_pool);
    int Wp = poolOutDim(W, F_pool, S_pool);


    if (Hp <= 0 || Wp <= 0 || C <= 0) {
         // Output dimensions are zero or negative, or no channels. Nothing to compute.
         // fprintf(stderr, "Warning: MaxPoolLayer output size is zero or negative (Hp=%d, Wp=%d, C=%d). Skipping kernel launch.\n", Hp, Wp, C);
         return;
     }

    int total_outputs = Hp * Wp * C;
    int threads_per_block = 256;
    int blocks_per_grid = (total_outputs + threads_per_block - 1) / threads_per_block;

    poolKernel<<<blocks_per_grid, threads_per_block>>>(
        d_output, d_input,
        H, W, C,
        F_pool, S_pool,
        Hp, Wp);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}

// Definition for cudaLRNLayer
void cudaLRNLayer(
    float* d_output,
    const float* d_input,
    int H, int W, int C,
    int N, float alpha, float beta, float k)
{
    if (H <= 0 || W <= 0 || C <= 0 || N <= 0) {
         // Invalid dimensions or LRN window size.
         // fprintf(stderr, "Warning: LRNLayer input size or N is zero or negative. Skipping kernel launch.\n");
         // If output must match input size, consider copying input to output here.
         if (d_output != d_input && H*W*C > 0) { // Avoid self-copy
             CUDA_CHECK(cudaMemcpy(d_output, d_input, (size_t)H*W*C*sizeof(float), cudaMemcpyDeviceToDevice));
         }
         return;
     }
    int total_elements = H * W * C;
    int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    lrnKernel<<<blocks_per_grid, threads_per_block>>>(
        d_output, d_input,
        H, W, C, N, alpha, beta, k);
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}