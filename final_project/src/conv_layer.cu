#include <cuda_runtime.h>
#include <stdio.h>
#include "conv_layer.cuh"

#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// Tiled convolution kernel: computes one output pixel per thread.
// - Input is assumed to be in C x H x W order.
// - Kernel is in K x C x R x S order.
// - Output is in K x outH x outW order.
__global__ void convTiledKernel(const float* __restrict__ input,
                                const float* __restrict__ kernel,
                                const float* __restrict__ bias,
                                float* __restrict__ output,
                                int C, int H, int W,
                                int K, int R, int S,
                                int stride,
                                int outH, int outW)
{
    int out_x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int out_y = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    int filter = blockIdx.z;  // One output channel per block in the Z-dimension.

    if (out_x >= outW || out_y >= outH)
        return;

    float value = bias[filter]; // Start with the bias for this filter.

    // Perform convolution for one output pixel.
    for (int c = 0; c < C; c++) {
        for (int r = 0; r < R; r++) {
            for (int s = 0; s < S; s++) {
                int in_y = out_y * stride + r;
                int in_x = out_x * stride + s;
                if (in_y < H && in_x < W) {
                    int inputIdx = c * (H * W) + in_y * W + in_x;
                    int kernelIdx = filter * (C * R * S) + c * (R * S) + r * S + s;
                    value += input[inputIdx] * kernel[kernelIdx];
                }
            }
        }
    }

    // Apply ReLU activation.
    value = value > 0.0f ? value : 0.0f;

    int outputIdx = filter * (outH * outW) + out_y * outW + out_x;
    output[outputIdx] = value;
}

// Host function that sets up device memory, launches the kernel, and copies results back.
void runConvolution(const float* h_input, const float* h_kernel, const float* h_bias, 
                    float* h_output, int C, int H, int W, int K, int R, int S, int stride)
{
    int outH = (H - R) / stride + 1;
    int outW = (W - S) / stride + 1;
    int inputSize = C * H * W;
    int kernelSize = K * C * R * S;
    int outputSize = K * outH * outW;

    float *d_input, *d_kernel, *d_bias, *d_output;
    cudaMalloc((void**)&d_input, inputSize * sizeof(float));
    cudaMalloc((void**)&d_kernel, kernelSize * sizeof(float));
    cudaMalloc((void**)&d_bias, K * sizeof(float));
    cudaMalloc((void**)&d_output, outputSize * sizeof(float));

    cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridDim((outW + TILE_WIDTH - 1) / TILE_WIDTH,
                 (outH + TILE_HEIGHT - 1) / TILE_HEIGHT,
                 K);

    convTiledKernel<<<gridDim, blockDim>>>(d_input, d_kernel, d_bias, d_output,
                                             C, H, W, K, R, S, stride, outH, outW);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_bias);
    cudaFree(d_output);
}
