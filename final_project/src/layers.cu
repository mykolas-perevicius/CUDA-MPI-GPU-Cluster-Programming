// final_project/src/layers.cu
#include "layers.hpp"
#include <stdio.h>
#include <math.h>

// ---------------------------------------------------------------------
// Existing kernels for Conv (used for block1), ReLU, and MaxPool
// ---------------------------------------------------------------------
__global__ void conv2d_forward_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    const float* __restrict__ weights, const float* __restrict__ biases,
    int C, int H, int W,
    int K, int R, int S,
    int stride, int padding,
    int H_out, int W_out)
{
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;
    if (k < K && h_out < H_out && w_out < W_out) {
        float acc = biases[k];
        int h_in_start = h_out * stride - padding;
        int w_in_start = w_out * stride - padding;
        for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
                for (int s_ = 0; s_ < S; ++s_) {
                    int h_in = h_in_start + r;
                    int w_in = w_in_start + s_;
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        int input_idx = c * (H * W) + h_in * W + w_in;
                        int weight_idx = k * (C * R * S) + c * (R * S) + r * S + s_;
                        acc += input[input_idx] * weights[weight_idx];
                    }
                }
            }
        }
        int output_idx = k * (H_out * W_out) + h_out * W_out + w_out;
        output[output_idx] = acc;
    }
}

__global__ void relu_forward_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = fmaxf(0.f, data[idx]);
    }
}

__global__ void maxpool_forward_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int C, int H_in, int W_in,
    int poolSize, int stride,
    int H_out, int W_out)
{
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    if (c < C && h_out < H_out && w_out < W_out) {
        int h_in_start = h_out * stride;
        int w_in_start = w_out * stride;
        float max_val = -INFINITY;
        for (int r = 0; r < poolSize; ++r) {
            for (int s_ = 0; s_ < poolSize; ++s_) {
                int h_in = h_in_start + r;
                int w_in = w_in_start + s_;
                if (h_in < H_in && w_in < W_in) {
                    int input_idx = c * (H_in * W_in) + h_in * W_in + w_in;
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
        int output_idx = c * (H_out * W_out) + h_out * W_out + w_out;
        output[output_idx] = max_val;
    }
}

// ---------------------------------------------------------------------
// New Kernels for Block2: Conv2, ReLU2, Pool2, and LRN2
// ---------------------------------------------------------------------

// Conv2: We reuse conv2d_forward_kernel for Conv2
// (launch_conv2d_forward_conv2 will call conv2d_forward_kernel with parameters for Conv2)

void launch_conv2d_forward_conv2(
    const float* d_input, float* d_output,
    const float* d_weights, const float* d_biases,
    const ConvLayerParams& params,
    cudaStream_t stream)
{
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks(
        (params.outputWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (params.outputHeight + threadsPerBlock.y - 1) / threadsPerBlock.y,
        params.outputChannels
    );
    conv2d_forward_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>( 
        d_input, d_output, d_weights, d_biases,
        params.inputChannels, params.inputHeight, params.inputWidth,
        params.outputChannels, params.kernelSize, params.kernelSize,
        params.stride, params.padding, params.outputHeight, params.outputWidth
    );
}

// ReLU2: use same relu_forward_kernel but separate launcher
void launch_relu_forward_conv2(float* d_data, int N, cudaStream_t stream) {
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    relu_forward_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_data, N);
}

// Pool2: New maxpool kernel using PoolLayerParams
__global__ void maxpool_forward_kernel2(
    const float* __restrict__ input, float* __restrict__ output,
    PoolLayerParams params)
{
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    if (c < params.inputChannels && h_out < params.outputHeight && w_out < params.outputWidth) {
        int h_in_start = h_out * params.stride;
        int w_in_start = w_out * params.stride;
        float max_val = -INFINITY;
        for (int r = 0; r < params.poolSize; ++r) {
            for (int s_ = 0; s_ < params.poolSize; ++s_) {
                int h_in = h_in_start + r;
                int w_in = w_in_start + s_;
                if (h_in < params.inputHeight && w_in < params.inputWidth) {
                    int input_idx = c * (params.inputHeight * params.inputWidth) + h_in * params.inputWidth + w_in;
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
        int output_idx = c * (params.outputHeight * params.outputWidth) + h_out * params.outputWidth + w_out;
        output[output_idx] = max_val;
    }
}

void launch_maxpool_forward2(
    const float* d_input, float* d_output,
    const PoolLayerParams& params,
    cudaStream_t stream)
{
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks(
        (params.outputWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (params.outputHeight + threadsPerBlock.y - 1) / threadsPerBlock.y,
        params.inputChannels
    );
    maxpool_forward_kernel2<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_input, d_output, params
    );
}

// LRN2: Naive cross-channel Local Response Normalization kernel
__global__ void lrn_forward_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    LRNLayerParams params)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    if (c < params.channels && y < params.height && x < params.width) {
        int idx = c * (params.height * params.width) + y * params.width + x;
        int halfWindow = (params.localSize - 1) / 2;
        int c_start = max(0, c - halfWindow);
        int c_end = min(params.channels - 1, c + halfWindow);
        float accum = 0.0f;
        for (int c2 = c_start; c2 <= c_end; c2++) {
            int idx2 = c2 * (params.height * params.width) + y * params.width + x;
            float val = input[idx2];
            accum += val * val;
        }
        float denom = powf(params.k + params.alpha * accum, params.beta);
        output[idx] = input[idx] / denom;
    }
}

void launch_lrn_forward(
    const float* d_input, float* d_output,
    const LRNLayerParams& params,
    cudaStream_t stream)
{
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks(
        (params.width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (params.height + threadsPerBlock.y - 1) / threadsPerBlock.y,
        params.channels
    );
    lrn_forward_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_input, d_output, params);
}
