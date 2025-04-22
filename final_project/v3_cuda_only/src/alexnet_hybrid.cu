// final_project/src/alexnet_hybrid.cu
#include <cuda_runtime.h>
#include <iostream>
#include "alexnet.hpp"
#include "layers.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Dimension Constants for Block1 (assumed implemented elsewhere)
// For example, after Conv1->ReLU->MaxPool1:
static const int POOL1_C_OUT = 96;
static const int POOL1_H_OUT = 27;
static const int POOL1_W_OUT = 27;

// Dimensions for Block2 (Conv2 -> ReLU2 -> Pool2 -> LRN2)
static const int CONV2_K = 256;
static const int CONV2_R = 5;
static const int CONV2_S = 5;
static const int CONV2_STRIDE = 1;
static const int CONV2_PAD = 2;
static const int CONV2_H_OUT = 27; // (27 - 5 + 2*2) / 1 + 1 = 27
static const int CONV2_W_OUT = 27; // Same as above

// Pool2 parameters: Pool size 3, stride 2, output dims: floor((27-3)/2)+1 = 13
static const int POOL2_POOL_SIZE = 3;
static const int POOL2_STRIDE = 2;
static const int POOL2_H_OUT = 13;
static const int POOL2_W_OUT = 13;
static const int POOL2_C_OUT = CONV2_K; // 256

// LRN2 parameters: Applies on Pool2 output (256,13,13)
static const int LRN2_LOCAL_SIZE = 5;
static const float LRN2_ALPHA = 1.0e-4f;
static const float LRN2_BETA  = 0.75f;
static const float LRN2_K     = 2.0f;

// The alexnetForward function now includes Block1 and Block2
// For simplicity, we assume Block1 (Conv1->ReLU->Pool1) is already computed and its result is in d_pool1_out.
// In this demonstration, we'll allocate d_pool1_out and fill it with a placeholder value.
void alexnetForward(
    const float* h_input_batch, float* h_output_batch,
    int N, int C, int H, int W,
    int outputSize)
{
    // Assume batch size N=1 per rank for simplicity.
    size_t input_bytes = (size_t)N * C * H * W * sizeof(float);
    size_t pool1_out_bytes = (size_t)N * POOL1_C_OUT * POOL1_H_OUT * POOL1_W_OUT * sizeof(float);
    size_t conv2_out_bytes = (size_t)N * CONV2_K * CONV2_H_OUT * CONV2_W_OUT * sizeof(float);
    size_t pool2_out_bytes = (size_t)N * POOL2_C_OUT * POOL2_H_OUT * POOL2_W_OUT * sizeof(float);
    size_t lrn2_out_bytes  = pool2_out_bytes;

    float *d_input_batch = nullptr;
    float *d_pool1_out   = nullptr;
    float *d_conv2_out   = nullptr;
    float *d_pool2_out   = nullptr;
    float *d_lrn2_out    = nullptr;

    // Allocate memory for Block1 output (placeholder)
    CUDA_CHECK(cudaMalloc((void**)&d_input_batch, input_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_pool1_out, pool1_out_bytes));
    // For demonstration, fill d_pool1_out with zeros (in practice, use Conv1->ReLU->Pool1)
    CUDA_CHECK(cudaMemset(d_pool1_out, 0, pool1_out_bytes));

    // Allocate memory for Block2 intermediate outputs
    CUDA_CHECK(cudaMalloc((void**)&d_conv2_out, conv2_out_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_pool2_out, pool2_out_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_lrn2_out,  lrn2_out_bytes));

    // Allocate and initialize Conv2 weights and biases will be done in main (weights copied to device elsewhere)
    // In this function, we assume d_conv2_weights and d_conv2_biases are available externally.
    // For this demonstration, we'll assume they have been copied into device memory by main via a broadcast.
    // Instead, here we use placeholders for d_conv2_weights and d_conv2_biases.
    float *d_conv2_weights = nullptr;
    float *d_conv2_biases  = nullptr;
    size_t conv2_weights_bytes = (size_t)CONV2_K * POOL1_C_OUT * CONV2_R * CONV2_S * sizeof(float);
    size_t conv2_biases_bytes  = (size_t)CONV2_K * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&d_conv2_weights, conv2_weights_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_conv2_biases, conv2_biases_bytes));
// MPI?     // (In real implementation, weights are copied from host via MPI_Bcast.)

    // Create a CUDA stream (optional)
    cudaStream_t stream = 0;

    // Start timing the Block2 operations
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));

    // Set up Conv2 parameters (input to Conv2 is output of Pool1)
    ConvLayerParams conv2_params;
    conv2_params.inputChannels  = POOL1_C_OUT;  // 96 from pool1 output
    conv2_params.outputChannels = CONV2_K;      // 256
    conv2_params.kernelSize     = CONV2_R;        // 5
    conv2_params.stride         = CONV2_STRIDE;   // 1
    conv2_params.padding        = CONV2_PAD;      // 2
    conv2_params.inputHeight    = POOL1_H_OUT;    // 27
    conv2_params.inputWidth     = POOL1_W_OUT;    // 27
    conv2_params.outputHeight   = CONV2_H_OUT;    // 27
    conv2_params.outputWidth    = CONV2_W_OUT;    // 27

    launch_conv2d_forward_conv2(
        d_pool1_out, d_conv2_out,
        d_conv2_weights, d_conv2_biases,
        conv2_params, stream);
    CUDA_CHECK(cudaGetLastError());

    // ReLU2 on conv2 output (in-place)
    int conv2_num_elements = N * CONV2_K * CONV2_H_OUT * CONV2_W_OUT;
    launch_relu_forward_conv2(d_conv2_out, conv2_num_elements, stream);
    CUDA_CHECK(cudaGetLastError());

    // Pool2 parameters
    PoolLayerParams pool2_params;
    pool2_params.poolSize = POOL2_POOL_SIZE;  // 3
    pool2_params.stride   = POOL2_STRIDE;       // 2
    pool2_params.inputChannels = CONV2_K;       // 256
    pool2_params.inputHeight = CONV2_H_OUT;       // 27
    pool2_params.inputWidth  = CONV2_W_OUT;       // 27
    pool2_params.outputHeight = POOL2_H_OUT;       // 13
    pool2_params.outputWidth  = POOL2_W_OUT;       // 13

    launch_maxpool_forward2(d_conv2_out, d_pool2_out, pool2_params, stream);
    CUDA_CHECK(cudaGetLastError());

    // LRN2 parameters
    LRNLayerParams lrn2_params;
    lrn2_params.channels = POOL2_C_OUT;   // 256
    lrn2_params.height   = POOL2_H_OUT;     // 13
    lrn2_params.width    = POOL2_W_OUT;     // 13
    lrn2_params.localSize = LRN2_LOCAL_SIZE; // 5
    lrn2_params.alpha   = LRN2_ALPHA;       // 1e-4
    lrn2_params.beta    = LRN2_BETA;        // 0.75
    lrn2_params.k       = LRN2_K;           // 2.0

    launch_lrn_forward(d_pool2_out, d_lrn2_out, lrn2_params, stream);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float kernel_time_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, start, stop));
    std::cout << "Kernel execution time (Block2: Conv2->LRN2): " << kernel_time_ms << " ms." << std::endl;

    // Copy final output (from LRN2) back to host
    size_t final_output_bytes = (size_t)N * POOL2_C_OUT * POOL2_H_OUT * POOL2_W_OUT * sizeof(float);
    CUDA_CHECK(cudaMemcpy(h_output_batch, d_lrn2_out, final_output_bytes, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_input_batch));
    CUDA_CHECK(cudaFree(d_pool1_out));
    CUDA_CHECK(cudaFree(d_conv2_out));
    CUDA_CHECK(cudaFree(d_pool2_out));
    CUDA_CHECK(cudaFree(d_lrn2_out));
    CUDA_CHECK(cudaFree(d_conv2_weights));
    CUDA_CHECK(cudaFree(d_conv2_biases));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
