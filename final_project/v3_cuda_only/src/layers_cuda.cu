#include <cstdio>         // for fprintf, stderr
#include <cstdlib>        // for exit
#include <cmath>          // for fmaxf, powf
#include <cuda_runtime.h>
#include "../include/layers.hpp"

// Macro to check CUDA calls
#define CUDA_CHECK(call)                                    \
  do {                                                      \
    cudaError_t err = call;                                 \
    if(err != cudaSuccess) {                                \
      std::fprintf(stderr,                                  \
        "CUDA error %s:%d '%s'\n",                         \
        __FILE__, __LINE__, cudaGetErrorString(err));       \
      std::exit(1);                                         \
    }                                                       \
  } while(0)

// Conv kernel (one thread per output element)
__global__ void convKernel(
    float* out, const float* in, const float* w, const float* b,
    int H, int W, int C, int K, int F, int S, int P,
    int Ho, int Wo)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Ho*Wo*K) return;

    int k = idx % K;
    int tmp = idx / K;
    int x = tmp % Wo;
    int y = tmp / Wo;

    float sum = 0.0f;
    for(int c=0;c<C;++c)
    for(int fy=0;fy<F;++fy)
    for(int fx=0;fx<F;++fx) {
        int in_y = y*S + fy - P;
        int in_x = x*S + fx - P;
        if(in_y>=0 && in_y<H && in_x>=0 && in_x<W) {
            int in_idx = ((in_y*W)+in_x)*C + c;
            int w_idx  = ((k*C + c)*F + fy)*F + fx;
            sum += in[in_idx] * w[w_idx];
        }
    }
    out[idx] = sum + b[k];
}

void cudaConvLayer(
    float* d_output,
    const float* d_input,
    const float* d_weights,
    const float* d_biases,
    int H, int W, int C,
    const int K, const int F, const int S, const int P)
{
    int Ho = (H + 2*P - F)/S + 1;
    int Wo = (W + 2*P - F)/S + 1;
    int total = Ho*Wo*K;
    int block = 256, grid = (total+block-1)/block;
    convKernel<<<grid,block>>>(d_output,d_input,d_weights,d_biases,
        H,W,C,K,F,S,P,Ho,Wo);
    CUDA_CHECK(cudaGetLastError());
}

// ReLU
__global__ void reluKernel(float* data, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N) data[i] = fmaxf(0.0f, data[i]);
}

void cudaReluLayer(float* d_data, int N) {
    int block = 256, grid = (N+block-1)/block;
    reluKernel<<<grid,block>>>(d_data,N);
    CUDA_CHECK(cudaGetLastError());
}

// Max‐pool
__global__ void poolKernel(
    float* out, const float* in,
    int H, int W, int C,
    int Fp, int Sp,
    int Hp, int Wp)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>=Hp*Wp*C) return;

    int c = idx % C;
    int tmp = idx / C;
    int x = tmp % Wp;
    int y = tmp / Wp;

    float mx = -1e20f;
    for(int fy=0;fy<Fp;++fy)
    for(int fx=0;fx<Fp;++fx){
        int in_y = y*Sp + fy;
        int in_x = x*Sp + fx;
        int in_idx = ((in_y*W)+in_x)*C + c;
        mx = fmaxf(mx, in[in_idx]);
    }
    out[idx] = mx;
}

void cudaMaxPoolLayer(
    float* d_output,
    const float* d_input,
    int H, int W, int C,
    int F_pool, int S_pool)
{
    int Hp = (H - F_pool)/S_pool + 1;
    int Wp = (W - F_pool)/S_pool + 1;
    int total = Hp*Wp*C;
    int block = 256, grid = (total+block-1)/block;
    poolKernel<<<grid,block>>>(d_output,d_input,
        H,W,C,F_pool,S_pool,Hp, Wp);
    CUDA_CHECK(cudaGetLastError());
}

// LRN (naive cross‐channel)
__global__ void lrnKernel(
    float* out, const float* in,
    int H, int W, int C, int N,
    float alpha, float beta, float k)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>=H*W*C) return;

    int c = idx % C;
    int tmp = idx / C;
    int x = tmp % W;
    int y = tmp / W;

    float sum = 0.0f;
    int half = N/2;
    for(int cc = max(0,c-half); cc<=min(C-1,c+half); ++cc) {
        int ii = ((y*W)+x)*C + cc;
        sum += in[ii]*in[ii];
    }
    out[idx] = in[idx] / powf(k + alpha*sum, beta);
}

void cudaLRNLayer(
    float* d_output,
    const float* d_input,
    int H, int W, int C,
    int N, float alpha, float beta, float k)
{
    int total = H*W*C;
    int block = 256, grid = (total+block-1)/block;
    lrnKernel<<<grid,block>>>(d_output,d_input,
        H,W,C,N,alpha,beta,k);
    CUDA_CHECK(cudaGetLastError());
}
