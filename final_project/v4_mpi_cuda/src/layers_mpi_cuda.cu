// final_project/v4_mpi_cuda/src/layers_mpi_cuda.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <mpi.h>
#include <cstddef>
#include <algorithm> // Include for std::max, std::min used by device kernels

#include "../include/layers.hpp"
#include "../include/alexnet.hpp" // Include alexnet.hpp for convOutDim/poolOutDim

// Macro to check CUDA calls - ABORTS using MPI for coordinated shutdown
#define CUDA_CHECK(call)                                    \
  do {                                                      \
    cudaError_t err = call;                                 \
    if(err != cudaSuccess) {                                \
      int rank_for_error=-1; MPI_Comm_rank(MPI_COMM_WORLD, &rank_for_error); \
      fprintf(stderr, "[Rank %d] CUDA error in %s:%d: %s (%d)\n", rank_for_error, __FILE__, __LINE__, cudaGetErrorString(err), err); \
      fflush(stderr); MPI_Abort(MPI_COMM_WORLD, err); } } while(0)


// --- Kernel Implementations ---

__global__ void convKernel(
    float* __restrict__ out, const float* __restrict__ in,
    const float* __restrict__ w, const float* __restrict__ b,
    int H, int W, int C, int K, int F, int S, int P,
    int Ho, int Wo)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = Ho * Wo * K; if (idx >= total_outputs) return;
    int k = idx % K; int temp = idx / K; int x = temp % Wo; int y = temp / Wo;
    float sum = b[k]; // Initialize with bias
    int start_y = y * S - P; int start_x = x * S - P;
    for (int c = 0; c < C; ++c) {
        for (int fy = 0; fy < F; ++fy) { int current_y = start_y + fy;
            for (int fx = 0; fx < F; ++fx) { int current_x = start_x + fx;
                if (current_y >= 0 && current_y < H && current_x >= 0 && current_x < W) {
                    size_t in_idx = ((size_t)current_y * W + current_x) * C + c;
                    size_t w_idx = (((size_t)k * C + c) * F + fy) * F + fx;
                    sum += in[in_idx] * w[w_idx];
                }
            }
        }
    } out[idx] = sum;
}

__global__ void reluKernel(float* data, size_t N) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N) { data[i] = fmaxf(0.0f, data[i]); }
}

__global__ void poolKernel(
    float* __restrict__ out, const float* __restrict__ in,
    int H, int W, int C, int Fp, int Sp, int Hp, int Wp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = Hp * Wp * C; if (idx >= total_outputs) return;
    int c = idx % C; int temp = idx / C; int y = temp / Wp; int x = temp % Wp;
    int start_y = y * Sp; int start_x = x * Sp;
    float max_val = -INFINITY;
    for (int fy = 0; fy < Fp; ++fy) { int current_y = start_y + fy; if (current_y >= H) continue;
        for (int fx = 0; fx < Fp; ++fx) { int current_x = start_x + fx; if (current_x >= W) continue;
            size_t in_idx = ((size_t)current_y * W + current_x) * C + c;
            max_val = fmaxf(max_val, in[in_idx]);
        }
    } out[idx] = max_val;
}

__global__ void lrnKernel(
    float* __restrict__ out, const float* __restrict__ in,
    int H, int W, int C, int N_lrn, float alpha, float beta, float k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = H * W * C; if (idx >= total_elements) return;
    int c = idx % C; int temp = idx / C; int x = temp % W; int y = temp / W;
    float sum_sq = 0.0f; int half_N = N_lrn / 2;
    // Use std:: namespace for max/min inside kernel requires including <algorithm> and potentially modifying build flags if not already handled
    // Using standard max/min functions which should be available in CUDA context
    int c_start = max(0, c - half_N); int c_end = min(C - 1, c + half_N);
    for (int channel_i = c_start; channel_i <= c_end; ++channel_i) {
        size_t neighbor_idx = ((size_t)y * W + x) * C + channel_i;
        float val = in[neighbor_idx]; sum_sq += val * val;
    }
    float scale = k + alpha * sum_sq;
    if (scale <= 0.0f && beta > 0.0f && beta != 1.0f) { out[idx] = 0.0f; }
    else { out[idx] = in[idx] * powf(scale, -beta); }
}


// --- Kernel Launcher Functions ---

void cudaConvLayer(
    float* d_output, const float* d_input, const float* d_weights, const float* d_biases,
    int H, int W, int C, const int K, const int F, const int S, const int P)
{
    int Ho = convOutDim(H, F, P, S); int Wo = convOutDim(W, F, P, S);
    if (Ho <= 0 || Wo <= 0 || K <= 0) { return; }
    int total_outputs = Ho * Wo * K; int threads = 256; int blocks = (total_outputs + threads - 1) / threads;
    convKernel<<<blocks, threads>>>(d_output, d_input, d_weights, d_biases, H, W, C, K, F, S, P, Ho, Wo);
    CUDA_CHECK(cudaGetLastError());
}

void cudaReluLayer(float* d_data, size_t N) {
     if (N == 0 || d_data == nullptr) return;
    int threads = 256; size_t blocks = (N + threads - 1) / threads;
    // Ensure blocks does not exceed device limits (though unlikely for typical N)
    if (blocks > 2147483647) blocks = 2147483647; // Max grid dim for older devices
    reluKernel<<< (unsigned int)blocks, threads >>>(d_data, N); // Cast blocks for safety
    CUDA_CHECK(cudaGetLastError());
}

void cudaMaxPoolLayer(
    float* d_output, const float* d_input,
    int H, int W, int C, int F_pool, int S_pool)
{
    int Hp = poolOutDim(H, F_pool, S_pool); int Wp = poolOutDim(W, F_pool, S_pool);
    if (Hp <= 0 || Wp <= 0 || C <= 0) { return; }
    int total_outputs = Hp * Wp * C; int threads = 256; int blocks = (total_outputs + threads - 1) / threads;
    poolKernel<<<blocks, threads>>>(d_output, d_input, H, W, C, F_pool, S_pool, Hp, Wp);
    CUDA_CHECK(cudaGetLastError());
}

void cudaLRNLayer(
    float* d_output, const float* d_input,
    int H, int W, int C, int N, float alpha, float beta, float k)
{
    if (H <= 0 || W <= 0 || C <= 0 || N <= 0) {
         if (d_output != d_input && H*W*C > 0) { CUDA_CHECK(cudaMemcpy(d_output, d_input, (size_t)H*W*C*sizeof(float), cudaMemcpyDeviceToDevice)); }
         return;
     }
    int total_elements = H * W * C; int threads = 256; int blocks = (total_elements + threads - 1) / threads;
    lrnKernel<<<blocks, threads>>>(d_output, d_input, H, W, C, N, alpha, beta, k);
    CUDA_CHECK(cudaGetLastError());
}